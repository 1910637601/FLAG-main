from utils import *
from datetime import datetime, timedelta
from collections import defaultdict
from contextlib import nullcontext
from torch import optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from dataset.utils import *
from losses import *


def _compute_loss(model, sde, batch, config):
    mode = config['mode']
    if mode == "graph_fixed":
        return graph_fixed_loss(model, sde, batch, config)
    elif mode == "repa_graph_fixed":
        return repa_graph_fixed_loss(model, sde, batch, config)
    elif mode == "repa_cell_graph_fixed":
        return repa_cell_graph_fixed_loss(model, sde, batch, config)
    elif mode == "proto_graph_fixed":
        return proto_graph_fixed_loss(model, sde, batch, config)
    raise ValueError(f"mode {mode} not recognized")


class _NullWriter:
    def add_scalar(self, *a, **k): pass
    def close(self): pass


def _setup_dist(config):
    """torchrun sets WORLD_SIZE/RANK/LOCAL_RANK. Without torchrun: single process, config['device']."""
    world = int(os.environ.get('WORLD_SIZE', 1))
    if world > 1:
        local_rank = int(os.environ['LOCAL_RANK'])
        # long timeout: validation (diffusion sampling) runs on rank 0 only while others wait
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group('nccl', timeout=timedelta(hours=3))
            return True, dist.get_rank(), world, f'cuda:{local_rank}'
        dist.init_process_group('gloo', timeout=timedelta(hours=3))       # CPU debugging
        return True, dist.get_rank(), world, 'cpu'
    return False, 0, 1, config['device']


def train_graph_latent_diffusion(config):
    distributed, rank, world, device = _setup_dist(config)
    is_main = rank == 0
    if is_main:
        print(f"start train_graph_latent_diffusion pipeline ... (world_size={world})")

    seeding(config['seed'])
    mode = config['mode']

    data_config = config['data_config']
    model_config = config['model_config']
    train_config = config['train_config']

    result_folder = get_result_folder(config)
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(result_folder, "runs", current_time)
    if is_main:
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        logger = create_logger(log_dir)
        logger.info(f"Experiment directory created at {result_folder}")
    else:
        writer = _NullWriter()
        logger = logging.getLogger('silent')
        logger.addHandler(logging.NullHandler())
        logger.propagate = False

    train_dataset, selected_genes = load_dataset(data_config, mode='train')
    gene_size = len(selected_genes)
    valid_dataset, _ = load_dataset(data_config, mode='valid') if is_main else (None, None)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world, rank=rank, shuffle=True,
                                       seed=config['seed']) if distributed else None
    train_loader = make_loader(train_dataset, batch_size=train_config['batch_size'], shuffle=True,
                               sampler=train_sampler)
    valid_loader = make_loader(valid_dataset, batch_size=train_config['batch_size'], shuffle=False) if is_main else None

    prepare_model_config(config, selected_genes)
    if is_main:
        save_config(os.path.join(result_folder, 'config.yaml'), config)

    model = load_model(model_config, gene_size).to(device)        # identical init on every rank (same seed)
    logger.info(f"#params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    ema = EMA(model, decay=0.999)
    if distributed:
        train_model = DDP(model, device_ids=[torch.cuda.current_device()] if 'cuda' in str(device) else None,
                          find_unused_parameters=True)
        seeding(config['seed'] + rank)                            # different diffusion noise per rank
    else:
        train_model = model
    sde = load_sde(config['sde_config'])
    sampler_fn = load_sampler_fn(config['sde_config'])

    opt = optim.AdamW(model.parameters(), lr=train_config['lr'], weight_decay=train_config['wd'])

    # keep the effective number of tiles per optimizer step equal to gradient_accumulate_every
    accum_every = max(1, int(round(train_config['gradient_accumulate_every'] / world)))
    logger.info(f"world_size={world}, per-rank grad accumulation={accum_every}, "
                f"effective tiles/step={accum_every * world * train_config['batch_size']}")

    best_pcc = -1
    epochs_no_improve = 0
    patience = train_config['patience']
    val_num_batches = int(train_config.get('val_num_batches', 5))   # -1 -> whole validation set
    val_steps = int(train_config.get('val_steps', 20))
    checkpoint_path = None

    accum_step = 0

    for epoch in range(train_config['epochs']):
        model.train()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        pbar = tqdm(train_loader, desc=f"epoch {epoch}|", disable=not is_main)
        total_loss = 0
        epoch_logs = defaultdict(float)
        for batch in pbar:
            for k in batch:
                try:
                    batch[k] = batch[k].to(device)
                except:
                    continue
            accum_step += 1
            do_step = accum_step % accum_every == 0
            ctx = train_model.no_sync() if (distributed and not do_step) else nullcontext()
            with ctx:
                loss, logs, _ = _compute_loss(train_model, sde, batch, config)
                (loss / accum_every).backward()

            if do_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config['grad_clip'])
                opt.step()
                opt.zero_grad()
                ema.update()
                writer.add_scalar(f"Train-Step/loss", float(loss), accum_step)

            if is_main:
                pbar.set_postfix({"Loss": float(loss), **{k: float(v) for k, v in logs.items()}})
            total_loss += float(loss)
            for k, v in logs.items():
                epoch_logs[k] += float(v)

        n_batches = max(len(train_loader), 1)
        msg = " | ".join(f"{k}: {v / n_batches:.5f}" for k, v in epoch_logs.items())
        logger.info(f"Epoch={epoch} | Average Training Loss (rank0): {total_loss / n_batches:.5f} | {msg}")
        writer.add_scalar(f"Train-Epoch/total_loss", total_loss / n_batches, epoch)
        for k, v in epoch_logs.items():
            writer.add_scalar(f"Train-Epoch/{k}", v / n_batches, epoch)

        stop = False
        if epoch % train_config['eval_interval'] == 0:
            if is_main:
                model.eval()
                ema.apply_shadow()
                with torch.no_grad():
                    all_gt_genes, all_pred_genes, all_prior_genes = [], [], []
                    total_val_batches = len(valid_loader)
                    sample_num = total_val_batches if val_num_batches < 0 else min(val_num_batches, total_val_batches)
                    subset_indices = set(random.sample(range(total_val_batches), sample_num))

                    for i, batch in enumerate(tqdm(valid_loader, desc=f"Quick Val (Sampled {sample_num})")):
                        if i not in subset_indices:
                            continue
                        for k in batch:
                            try:
                                batch[k] = batch[k].to(device)
                            except:
                                continue
                        pred_x = sampler_fn(config, ema.model, sde, batch['cond_x'], batch['cond_e'], batch['mask'],
                                            steps=val_steps, mode=mode).squeeze(0).cpu()
                        all_gt_genes.append(batch['x_gt'].squeeze(0).cpu().numpy())
                        all_pred_genes.append(pred_x.numpy())
                        if mode == 'proto_graph_fixed':
                            prior = ema.model.compute_prior(batch['cond_x'], batch['cond_e'], batch['mask'])
                            all_prior_genes.append(prior['mu'].cpu().numpy())
                ema.restore()
                model.train()

                val_metrics = compute_metrics(all_pred_genes, all_gt_genes)
                valid_count = len(selected_genes) - val_metrics['nan_count']
                logger.info(f"Epoch={epoch} | Quick Validation Metrics: MSE={val_metrics['MSE']:.5f}, "
                            f"MAE={val_metrics['MAE']:.5f}, "
                            f"nan_count={val_metrics['nan_count']:.5f}, "
                            f"PCC-10={val_metrics['PCC-10']:.5f}, "
                            f"PCC-50={val_metrics['PCC-50']:.5f}, "
                            f"PCC-{valid_count}={val_metrics['PCC-all']:.5f}")
                writer.add_scalar(f"Validation/MSE", val_metrics['MSE'], epoch)
                writer.add_scalar(f"Validation/MAE", val_metrics['MAE'], epoch)
                writer.add_scalar(f"Validation/PCC-all", val_metrics['PCC-all'], epoch)

                if all_prior_genes:
                    pm = compute_metrics(all_prior_genes, all_gt_genes)
                    logger.info(f"Epoch={epoch} | Prior-only (cell-type mixture) Metrics: MSE={pm['MSE']:.5f}, "
                                f"PCC-50={pm['PCC-50']:.5f}, PCC-all={pm['PCC-all']:.5f}")
                    writer.add_scalar(f"Validation/prior_MSE", pm['MSE'], epoch)
                    writer.add_scalar(f"Validation/prior_PCC-all", pm['PCC-all'], epoch)

                val_pcc = val_metrics['PCC-all']
                checkpoint_path = save_checkpoint(model, ema, opt, result_folder, epoch)
                if val_pcc > best_pcc:
                    best_pcc = val_pcc
                    epochs_no_improve = 0
                    save_checkpoint(model, ema, opt, result_folder, epoch, name='best.pt')
                    logger.info(f"New best PCC-all={best_pcc:.5f} at epoch {epoch} -> best.pt")
                else:
                    epochs_no_improve += 1
                    logger.info(f"No improvement in validation pcc, epochs without improvement: "
                                f"{epochs_no_improve}/{patience}")
                if epochs_no_improve >= patience:
                    logger.info(f"Early stopping triggered after {epochs_no_improve} epochs without improvement")
                    stop = True

            if distributed:
                flag = torch.tensor([1 if stop else 0], device=device)
                dist.broadcast(flag, src=0)
                stop = bool(flag.item())
        if stop:
            break

    writer.close()
    logger.info(f"Training completed. Best validation pcc: {best_pcc:.5f}, Last checkpoint: {checkpoint_path}")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
