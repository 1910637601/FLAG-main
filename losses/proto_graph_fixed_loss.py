"""
Loss for FLAG-Proto (mode: 'proto_graph_fixed').

L = L_dsm(EDM-weighted)                     diffusion around the cell-type prior
  + lambda_align * L_repa                   FLAG's GFM (Geneformer) alignment, unchanged
  + lambda_prior * L_nb(mu_count, counts)   prior mean explains the observed counts (CPNN)
  + lambda_comp  * KL(W_deconv || w_hat)    composition grounded by deconvolution (CPNN L_R / DUET)
  + lambda_lib   * MSE(log_lib)             library size predicted from histology
  + lambda_proto * ||log T - log T0||^2     prototypes stay close to scRNA reference (CPNN L_R)
"""
import torch
import torch.nn.functional as F


def _repa_loss(zs, batch, cells, G):
    if zs is None or 'llm_gene_level_embeddings' not in batch:
        return None
    z = zs[0]
    z_tilde = batch['llm_gene_level_embeddings'].reshape(cells, G, -1)
    valid = batch['llm_gene_level_mask'].reshape(cells, G) > 0.5
    if z_tilde.shape[-1] != z.shape[-1]:
        raise ValueError(f"GFM embedding dim {z_tilde.shape[-1]} != projector z_dim {z.shape[-1]}")
    cos = (F.normalize(z, dim=-1) * F.normalize(z_tilde, dim=-1)).sum(-1)
    v = cos[valid]
    return (-v).mean() if v.numel() > 0 else torch.zeros([], device=cos.device)


def _nb_nll(y, mu, theta, eps=1e-8):
    """Negative binomial NLL, mean over entries. y, mu: (N,G); theta: (G,)"""
    theta = theta.unsqueeze(0)
    log_theta_mu = torch.log(theta + mu + eps)
    ll = (torch.lgamma(y + theta) - torch.lgamma(theta) - torch.lgamma(y + 1.0)
          + theta * (torch.log(theta + eps) - log_theta_mu)
          + y * (torch.log(mu + eps) - log_theta_mu))
    return -ll.mean()


def proto_graph_fixed_loss(model, sde, batch, config):
    lc = config['train_config'].get('proto_loss', {})
    lam_align = float(lc.get('lambda_align', 0.5))
    lam_prior = float(lc.get('lambda_prior', 1.0))
    lam_comp = float(lc.get('lambda_comp', 1.0))
    lam_lib = float(lc.get('lambda_lib', 0.1))
    lam_proto = float(lc.get('lambda_proto', 1.0))
    prior_loss_type = lc.get('prior_loss', 'nb')

    x = batch["x_gt"]                    # (1, N, G) log2(1+counts)
    cond_x = batch["cond_x"]
    cond_e = batch["cond_e"]
    mask = batch["mask"]
    e = batch["e_gt"]
    B, N, G = x.shape
    assert B == 1, "FLAG pipeline assumes batch_size=1 (one tile per step)"
    device = x.device

    t_min = config['sde_config']['t_min']
    t = t_min + (1 - t_min) * torch.rand(B, device=device)
    xt, _, sigma = sde.perturb(x, t)
    et = torch.zeros_like(e)

    _, zs, aux = model(xt, et, t, cond_x, cond_e, mask)
    model = model.module if hasattr(model, 'module') else model     # unwrap DDP for attribute access
    x0 = x.squeeze(0)
    m = mask.squeeze(0).float().unsqueeze(-1)                       # (N,1)
    msum = m.sum().clamp_min(1.0)

    # ---- EDM-weighted denoising loss (O(1) at every noise level) ----
    s = aux['sigma']
    sd = model.sigma_data
    weight = (s ** 2 + sd ** 2) / (s * sd) ** 2
    loss_x = (weight * (aux['D'] - x0) ** 2 * m).sum() / msum / G
    loss = loss_x
    logs = {'loss_x': loss_x.detach()}

    # ---- GFM alignment (FLAG) ----
    if lam_align > 0:
        la = _repa_loss(zs, batch, N, G)
        if la is not None:
            loss = loss + lam_align * la
            logs['loss_gene_embed'] = la.detach()

    if model.use_prior:
        # ---- prior likelihood ----
        if lam_prior > 0:
            if prior_loss_type == 'nb':
                counts = torch.clamp(torch.round(torch.pow(2.0, x0) - 1.0), min=0.0)
                theta = aux['theta']
                lp = _nb_nll(counts * m, aux['mu_count'] * m + (1 - m), theta)
            else:  # mse in log2 space
                lp = (((aux['mu'] - x0) ** 2) * m).sum() / msum / G
            loss = loss + lam_prior * lp
            logs['loss_prior'] = lp.detach()
            with torch.no_grad():
                logs['prior_mse'] = ((((aux['mu'] - x0) ** 2) * m).sum() / msum / G)

        # ---- composition supervision (deconvolution) ----
        if lam_comp > 0 and 'w_comp' in batch:
            w_t = batch['w_comp'].squeeze(0)                         # (N, C)
            wm = batch['w_mask'].squeeze(0).float() * m.squeeze(-1)
            if wm.sum() > 0:
                logq = F.log_softmax(aux['w_logits'], dim=-1)
                kl = (w_t * (torch.log(w_t.clamp_min(1e-8)) - logq)).sum(-1)
                lw = (kl * wm).sum() / wm.sum()
                loss = loss + lam_comp * lw
                logs['loss_comp'] = lw.detach()

        # ---- library size ----
        if lam_lib > 0 and 'log_lib' in batch:
            ll_t = batch['log_lib'].squeeze(0)
            mm = m.squeeze(-1)
            ll = (((aux['log_lib'] - ll_t) ** 2) * mm).sum() / mm.sum().clamp_min(1.0)
            loss = loss + lam_lib * ll
            logs['loss_lib'] = ll.detach()

        # ---- prototype regulariser ----
        if lam_proto > 0 and model.log_T.requires_grad:
            lr_ = ((model.log_T - model.log_T0) ** 2).mean()
            loss = loss + lam_proto * lr_
            logs['loss_proto'] = lr_.detach()

    return loss, logs, aux
