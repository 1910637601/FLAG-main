import torch


@torch.no_grad()
def proto_ode_sampler(config, model, sde, cond_x, cond_e, mask, steps=100, mode='proto_graph_fixed',
                      t_start=1.0, return_aux=False):
    """Heun probability-flow ODE sampler for FLAG-Proto.

    Differences from graph_ode_sampler:
      * starts from N(mu, sigma(t_start)^2) where mu is the histology-only cell-type prior
      * model returns (score, zs, aux)
      * t_start < 1 gives a shorter SDEdit-style trajectory around the prior (fewer steps)
    """
    device = cond_x.device
    B, N, G = cond_x.size(0), cond_x.size(1), config['model_config']['node_dim']
    e = torch.zeros(B, N, N, 1, device=device)

    prior = model.compute_prior(cond_x, cond_e, mask)
    mu = prior['mu'].unsqueeze(0) if model.use_prior else torch.zeros(B, N, G, device=device)

    t_end = config['sde_config']['t_min']
    ts = torch.linspace(float(t_start), t_end, steps + 1, device=device)
    x = mu + torch.randn(B, N, G, device=device) * sde.sigma(ts[0])

    def score(x_, t_):
        s, _, _ = model(x_, e, t_, cond_x, cond_e, mask)
        return s.unsqueeze(0)

    for k in range(steps):
        t = ts[k].expand(B)
        t_next = ts[k + 1].expand(B)
        dt = (t_next - t).view(B, 1, 1)
        g = sde.diffusion(t).view(B, 1, 1)
        d1 = -0.5 * g ** 2 * score(x, t)
        x_euler = x + d1 * dt
        g2 = sde.diffusion(t_next).view(B, 1, 1)
        d2 = -0.5 * g2 ** 2 * score(x_euler, t_next)
        x = x + 0.5 * (d1 + d2) * dt

    # Tweedie denoising at t_end
    t_final = ts[-1].expand(B)
    sigma_end = sde.sigma(t_final).view(B, 1, 1)
    x = x + sigma_end ** 2 * score(x, t_final)
    if return_aux:
        return x, prior
    return x
