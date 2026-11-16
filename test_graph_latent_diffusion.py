from utils import *
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from dataset.utils import *


def test_graph_latent_diffusion(config):
    print("start test_graph_latent_diffusion pipeline ...")
    seeding(config['seed'])
    device = config['device']
    mode = config['mode']

    data_config = config['data_config']
    model_config = config['model_config']
    test_config = config['test_config']

    result_folder = get_result_folder(config)
    os.makedirs(result_folder, exist_ok=True)

    # Initialize TensorBoard
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(result_folder, "runs", current_time)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    logger = create_logger(log_dir)
    logger.info(f"Experiment directory created at {result_folder}")

    test_dataset, selected_genes = load_dataset(data_config, mode='test')
    gene_size = len(selected_genes)

    test_loader = make_loader(test_dataset, batch_size=1, shuffle=False)
    prepare_model_config(config, selected_genes)
    model = load_model(model_config, gene_size).to(device)
    ema = EMA(model, decay=0.999)
    ckpt_path = os.path.join(result_folder, test_config['ckpt'])
    print("loading ckpt from: ", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    ema.shadow = ckpt['ema']
    model.eval()

    sde = load_sde(config['sde_config'])
    sampler_fn = load_sampler_fn(config['sde_config'])
    steps = int(test_config.get('steps', 100))
    is_proto = (mode == 'proto_graph_fixed')
    sampler_kwargs = {}
    if is_proto and 't_start' in test_config:
        sampler_kwargs['t_start'] = float(test_config['t_start'])

    all_gt_genes, all_pred_genes, all_prior_genes, all_w = [], [], [], []
    all_barcodes, all_slides = [], []

    ema.apply_shadow()
    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            for k in batch:
                try:
                    batch[k] = batch[k].to(device)
                except:
                    continue
            all_gt_genes.append(batch['x_gt'].squeeze(0).cpu().numpy())
            all_barcodes.append(np.asarray(batch['barcodes']))
            all_slides.append(np.asarray(batch['slides']))
            batch_pred_genes = 0
            for ix in range(test_config['sample_num']):
                pred_x = sampler_fn(config, ema.model, sde, batch['cond_x'], batch['cond_e'], batch['mask'],
                                    steps=steps, mode=mode, **sampler_kwargs).squeeze(0).cpu()
                pred_x[pred_x < 0] = 0
                batch_pred_genes += pred_x
            all_pred_genes.append((batch_pred_genes / test_config['sample_num']).numpy())
            if is_proto:
                prior = ema.model.compute_prior(batch['cond_x'], batch['cond_e'], batch['mask'])
                all_prior_genes.append(prior['mu'].cpu().numpy())
                all_w.append(prior['w'].cpu().numpy())
    elapsed = time.time() - t0
    ema.restore()
    logger.info(f"Inference time: {elapsed:.1f}s for {len(test_loader)} tiles "
                f"({test_config['sample_num']} samples x {steps} steps)")

    tag = os.path.basename(ckpt_path)[:-3]
    save_results(all_barcodes, all_pred_genes, all_slides, selected_genes, os.path.join(result_folder, f"preds_{tag}"))
    if is_proto:
        save_results(all_barcodes, all_prior_genes, all_slides, selected_genes, os.path.join(result_folder, f"prior_{tag}"))
        save_results(all_barcodes, all_w, all_slides, ema.model.cell_types, os.path.join(result_folder, f"celltype_{tag}"))

    if test_config['mode'] == 'eval':
        val_metrics = compute_metrics(all_pred_genes, all_gt_genes)
        nan_count = val_metrics['nan_count']
        valid_count = len(selected_genes) - nan_count
        logger.info(f"Test Metrics: MSE={val_metrics['MSE']:.5f}, "
                    f"MAE={val_metrics['MAE']:.5f}, "
                    f"nan_count={val_metrics['nan_count']:.5f},"
                    f"PCC-10={val_metrics['PCC-10']:.5f}, "
                    f"PCC-50={val_metrics['PCC-50']:.5f}, "
                    f"PCC-{valid_count}={val_metrics['PCC-all']:.5f}")

        eval_result = {
            'MSE': [val_metrics['MSE']],
            'MAE': [val_metrics['MAE']],
            'PCC-10': [val_metrics['PCC-10']],
            'PCC-50': [val_metrics['PCC-50']],
            f'PCC-{valid_count}': [val_metrics['PCC-all']],
            'time_s': [elapsed],
        }
        if is_proto:
            pm = compute_metrics(all_prior_genes, all_gt_genes)
            logger.info(f"Prior-only Metrics: MSE={pm['MSE']:.5f}, PCC-10={pm['PCC-10']:.5f}, "
                        f"PCC-50={pm['PCC-50']:.5f}, PCC-all={pm['PCC-all']:.5f}")
            eval_result.update({'prior_MSE': [pm['MSE']], 'prior_PCC-50': [pm['PCC-50']],
                                'prior_PCC-all': [pm['PCC-all']]})
        metric_path = ckpt_path[:-3] + '_eval.csv'
        pd.DataFrame(eval_result).to_csv(metric_path, index=False)
        logger.info(f"metrics saved to {metric_path}")
    else:
        raise ValueError(f"Test mode {test_config['mode']} not supported")
