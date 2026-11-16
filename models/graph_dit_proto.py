"""
FLAG + Cell-type Prototype Prior (FLAG-Proto).

Idea (borrowed from CPNN / DUET / GC-MoE and adapted to FLAG's diffusion setting):
    1. A spot is a mixture of cell types. Cell-type prototypes T (C x G, fraction-of-library
       expression profiles) are estimated offline from a scRNA-seq reference.
    2. A deterministic "prior head" predicts, from histology only (no noisy x_t),
         - cell-type composition  w_hat (N x C)       [supervised by spot deconvolution]
         - spot library size      log_lib (N)          [supervised by true total counts]
       and builds a biologically grounded prior mean
         lambda_g = lib * sum_c w_c T_cg * exp(a_g) + softplus(b_g)     (count space, CPNN eq.1)
         mu_g     = log2(1 + lambda_g)                                    (same space as targets)
       a_g / b_g are gene-wise modality corrections (scRNA -> Visium capture gap).
    3. The diffusion model denoises around this prior with EDM-style preconditioning:
         D(x_t) = mu + c_skip(s) (x_t - mu) + c_out(s) F(c_in(s)(x_t - mu), cond)
       so at high noise the denoiser falls back to the cell-type prior, and the network only
       has to learn the residual. The returned score is (D - x_t) / sigma^2, so the original
       samplers keep working; sampling starts from N(mu, sigma_max^2).

The rest (spatial graph encoder, gene-token DiT, GFM/REPA alignment) is inherited from FLAG.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.graph_dit_repa import GraphDitRepa


def _load_prototypes(path, gene_list, eps=1e-8):
    """Load prototypes npz and align them to the target gene order.

    npz keys: T (C x G_all, rows sum to 1), genes (G_all,), cell_types (C,)
    Returns T0 (C x G) float32, cell_types list, found mask (G,) bool.
    """
    data = np.load(path, allow_pickle=True)
    T_all = np.asarray(data['T'], dtype=np.float64)
    genes = [str(g).upper() for g in data['genes']]
    cell_types = [str(c) for c in data['cell_types']]
    g2i = {g: i for i, g in enumerate(genes)}

    G = len(gene_list)
    C = T_all.shape[0]
    T0 = np.zeros((C, G), dtype=np.float64)
    found = np.zeros(G, dtype=bool)
    for j, g in enumerate(gene_list):
        i = g2i.get(str(g).upper())
        if i is not None:
            T0[:, j] = T_all[:, i]
            found[j] = True
    # genes absent from the scRNA reference: give them the smallest observed value so the
    # prior is tiny but non-zero; the modality correction (a_g, b_g) can still lift them.
    pos = T_all[T_all > 0]
    floor = float(pos.min()) if pos.size > 0 else eps
    T0[:, ~found] = floor
    T0 = np.clip(T0, floor * 1e-3 + eps, None)
    return T0.astype(np.float32), cell_types, found


class GraphDitProto(GraphDitRepa):
    def __init__(self, model_config):
        super().__init__(model_config)
        cfg = model_config
        H = self.hidden_dim
        G = self.node_dim
        img_dim = cfg['node_cond_dim']

        self.use_prior = bool(cfg.get('use_prior', True))
        self.detach_prior = bool(cfg.get('detach_prior_in_dsm', True))
        self.use_img_cond = bool(cfg.get('use_img_cond', False))
        self.use_proto_gene_embed = bool(cfg.get('use_proto_gene_embed', True))
        self.sigma_data = float(cfg.get('sigma_data', 1.0))
        self.sigma_min = float(cfg['sigma_min'])
        self.sigma_max = float(cfg['sigma_max'])
        self._log_r = math.log(self.sigma_max / self.sigma_min)

        # ---------------- prototypes ----------------
        gene_list = cfg['gene_list']
        assert len(gene_list) == G, f"gene_list ({len(gene_list)}) != node_dim ({G})"
        T0, cell_types, found = _load_prototypes(cfg['prototype_path'], gene_list)
        self.cell_types = cell_types
        C = T0.shape[0]
        self.num_cell_types = C
        print(f"[GraphDitProto] {C} cell types, {int(found.sum())}/{G} target genes found in scRNA reference")

        self.register_buffer('log_T0', torch.from_numpy(np.log(T0)))          # (C, G)
        self.register_buffer('gene_found', torch.from_numpy(found))
        self.log_T = nn.Parameter(torch.from_numpy(np.log(T0)).clone(),
                                  requires_grad=bool(cfg.get('update_prototypes', True)))
        # gene-wise modality correction (CPNN alpha / beta)
        self.mod_a = nn.Parameter(torch.zeros(G))
        self.mod_b = nn.Parameter(torch.full((G,), -5.0))                      # softplus(-5) ~ 0.0067 counts
        # NB dispersion for the prior likelihood
        self.log_theta = nn.Parameter(torch.full((G,), math.log(10.0)))

        # ---------------- prior head (histology only) ----------------
        ph = int(cfg.get('prior_hidden', 512))
        self.prior_mlp = nn.Sequential(
            nn.LayerNorm(2 * img_dim),
            nn.Linear(2 * img_dim, ph), nn.GELU(), nn.Dropout(cfg.get('dropout', 0.1)),
            nn.Linear(ph, ph), nn.GELU(),
        )
        self.comp_head = nn.Linear(ph, C)
        self.lib_head = nn.Linear(ph, 1)
        nn.init.zeros_(self.lib_head.weight)
        nn.init.constant_(self.lib_head.bias, float(cfg.get('log_lib_init', 8.0)))

        # ---------------- prior -> diffusion conditioning ----------------
        self.prior_embed = nn.Sequential(nn.Linear(1, H), nn.SiLU(), nn.Linear(H, H))   # per-gene prior value
        self.comp_embed = nn.Sequential(nn.Linear(C, H), nn.SiLU(), nn.Linear(H, H))    # per-spot composition
        nn.init.zeros_(self.prior_embed[-1].weight); nn.init.zeros_(self.prior_embed[-1].bias)
        nn.init.zeros_(self.comp_embed[-1].weight); nn.init.zeros_(self.comp_embed[-1].bias)
        if self.use_proto_gene_embed:
            # gene identity enriched by its cell-type specificity profile (GC-MoE flavour)
            self.proto_gene_embed = nn.Linear(C, H)
            nn.init.zeros_(self.proto_gene_embed.weight); nn.init.zeros_(self.proto_gene_embed.bias)

    # ------------------------------------------------------------------
    def sigma(self, t):
        return self.sigma_min * torch.exp(self._log_r * t)

    def prototypes(self):
        return torch.exp(self.log_T)                                            # (C, G)

    def compute_prior(self, cond_x, cond_e, mask=None):
        """Histology-only prior. cond_x: (1,N,Dimg), cond_e: (1,N,N,2) with channel 0 = distance RBF.
        Returns dict with mu (N,G) in log2(1+x) space, mu_count (N,G), w (N,C), w_logits, log_lib (N)."""
        feat = cond_x.squeeze(0).float()                                        # (N, D)
        N = feat.shape[0]
        if cond_e is not None:
            Wd = cond_e.squeeze(0)[..., 0].float()                              # (N, N)
            Wd = Wd + torch.eye(N, device=feat.device, dtype=feat.dtype)
            if mask is not None:
                m = mask.squeeze(0).float()
                Wd = Wd * m.unsqueeze(0)
            ctx = (Wd @ feat) / Wd.sum(dim=1, keepdim=True).clamp_min(1e-6)
        else:
            ctx = feat
        h = self.prior_mlp(torch.cat([feat, ctx], dim=-1))
        w_logits = self.comp_head(h)                                            # (N, C)
        w = torch.softmax(w_logits, dim=-1)
        log_lib = self.lib_head(h).squeeze(-1)                                  # (N,)

        T = self.prototypes()                                                   # (C, G)
        mix = w @ T                                                             # (N, G) fraction
        lam = torch.exp(log_lib.clamp(max=15.0)).unsqueeze(-1) * mix * torch.exp(self.mod_a) \
              + F.softplus(self.mod_b)
        mu = torch.log2(1.0 + lam)
        return {'mu': mu, 'mu_count': lam, 'w': w, 'w_logits': w_logits, 'log_lib': log_lib}

    # ------------------------------------------------------------------
    def forward(self, x, e, t, cond_x, cond_e, mask):
        """x: (1,N,G) noisy expression x_t. Returns score (N,G), zs (list), aux dict."""
        x_t = x.squeeze(0).float()                                              # (N, G)
        N, G = x_t.shape
        sigma = self.sigma(t).view(-1, 1)                                       # (1,1) broadcast over spots

        prior = self.compute_prior(cond_x, cond_e, mask)
        if self.use_prior:
            mu = prior['mu'].detach() if self.detach_prior else prior['mu']
            w_cond = prior['w']
        else:
            mu = torch.zeros_like(x_t)
            w_cond = None

        sd = self.sigma_data
        c_in = 1.0 / torch.sqrt(sigma ** 2 + sd ** 2)
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out = sigma * sd / torch.sqrt(sigma ** 2 + sd ** 2)
        r_in = c_in * (x_t - mu)                                                # normalised residual

        # ---- spatial graph encoder (unchanged FLAG backbone) ----
        t_emb = self.time_embed(t)                                              # (1, H)
        graph_in = (mu + r_in).unsqueeze(0)                                     # (1, N, G)
        z, _ = self.graph_backbone(graph_in, e, t_emb, cond_x, cond_e, mask)
        cond_graph = self.node2gene(z.squeeze(0))                               # (N, H)

        # ---- gene tokens ----
        tok = self.gene_joint_embed(r_in)                                       # (N, G, H)
        if self.use_prior:
            tok = tok + self.prior_embed(mu.unsqueeze(-1))
        if self.use_proto_gene_embed:
            lt = self.log_T0.t()                                                # (G, C)
            if lt.shape[1] > 1:   # per-gene z-score across cell types = cell-type specificity
                lt = (lt - lt.mean(dim=1, keepdim=True)) / (lt.std(dim=1, keepdim=True) + 1e-6)
            tok = tok + self.proto_gene_embed(lt).unsqueeze(0)

        c = t_emb.expand(N, -1) + cond_graph
        if w_cond is not None:
            c = c + self.comp_embed(w_cond)
        if self.use_img_cond:
            c = c + self.label_embed(cond_x.squeeze(0).float())

        zs = None
        h = tok
        for i, block in enumerate(self.blocks):
            h = block(h, c)
            if (i + 1) == self.encoder_layer:
                zs = [proj(h.reshape(-1, self.hidden_dim)).reshape(N, self.node_dim, -1)
                      for proj in self.projectors]
        Fx = self.final_layer(h, c).squeeze(1)                                  # (N, G)

        D = mu + c_skip * (x_t - mu) + c_out * Fx                               # denoised estimate x0
        score = (D - x_t) / sigma ** 2
        aux = dict(prior)
        aux['D'] = D
        aux['theta'] = torch.exp(self.log_theta)    # computed in forward so DDP sees the parameter as used
        aux['sigma'] = sigma
        return score, zs, aux
