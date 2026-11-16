"""
Offline preparation of the cell-type prior for FLAG-Proto.

Outputs
  prototypes.npz : T (C x G_all, each row sums to 1), genes (G_all,), cell_types (C,)
  <celltype_dir>/<slide>.csv : spot x cell-type proportions (rows sum to 1), index = barcode

Sub-commands
  1) scref  : build prototypes from an annotated scRNA-seq reference (raw counts)
      python tools/prepare_celltype_prior.py scref \
          --sc_h5ad ref.h5ad --celltype_key celltype_major --out prototypes.npz

  2) deconv : estimate spot proportions with fixed prototypes (Poisson/KL NNLS on marker genes).
              Target genes are EXCLUDED from the markers (DUET-style) to avoid label leakage.
      python tools/prepare_celltype_prior.py deconv \
          --prototypes prototypes.npz --data_dir DATA --gene_list_filename hmhvg_geneformer_gene_list.txt \
          --slides train_list.txt valid_list.txt test_list.txt --out_dir DATA/celltype

     (Alternatively run cell2location / RCTD yourself and write the CSVs in the same format; the
      columns must be the same cell types, in the same order, as prototypes.npz.)

  3) nmf    : no scRNA reference available -> learn C "pseudo cell types" by KL-NMF on the
              TRAINING spots only, then deconvolve all slides with the learned prototypes.
      python tools/prepare_celltype_prior.py nmf --n_types 12 --data_dir DATA \
          --gene_list_filename ... --train_slides train_list.txt \
          --slides train_list.txt valid_list.txt test_list.txt \
          --out prototypes.npz --out_dir DATA/celltype
"""
import argparse
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp


# ----------------------------------------------------------------------------- utils
def _read_list(paths):
    out = []
    for p in paths:
        x = np.genfromtxt(p, dtype=str)
        out.extend([str(x)] if x.ndim == 0 else [str(s) for s in x])
    return list(dict.fromkeys(out))


def _upper(names):
    return np.array([str(g).upper() for g in names])


def _load_st(data_dir, slide):
    import anndata
    ad = anndata.read_h5ad(os.path.join(data_dir, 'st', f'{slide}.h5ad'))
    X = ad.X.tocsr() if sp.issparse(ad.X) else sp.csr_matrix(np.asarray(ad.X))
    genes = _upper(ad.var_names)
    # collapse duplicated gene symbols
    if len(set(genes)) != len(genes):
        _, first = np.unique(genes, return_index=True)
        first = np.sort(first)
        X, genes = X[:, first], genes[first]
    return X, genes, np.array([str(b) for b in ad.obs_names])


def _kl_nnls(Y, T, n_iter=300, eps=1e-10):
    """Y (S x G) counts, T (C x G) nonneg. Solve Y ~ A T, A>=0 (Poisson MLE). Returns A (S x C)."""
    S, C = Y.shape[0], T.shape[0]
    A = np.full((S, C), max(Y.sum(1).mean(), 1.0) / C / max(T.sum(1).mean(), eps))
    Tsum = T.sum(1)[None, :] + eps
    for _ in range(n_iter):
        R = Y / (A @ T + eps)
        A *= (R @ T.T) / Tsum
    return A


def _kl_nmf(Y, C, n_iter=500, seed=0, eps=1e-10):
    rng = np.random.default_rng(seed)
    S, G = Y.shape
    A = rng.random((S, C)) + 0.1
    T = rng.random((C, G)) + 0.1
    for it in range(n_iter):
        R = Y / (A @ T + eps)
        T *= (A.T @ R) / (A.sum(0)[:, None] + eps)
        R = Y / (A @ T + eps)
        A *= (R @ T.T) / (T.sum(1)[None, :] + eps)
        if it % 100 == 0:
            ll = (Y * np.log(A @ T + eps) - A @ T).sum() / S
            print(f"  nmf iter {it}: poisson ll/spot = {ll:.2f}")
    return A, T


def _select_markers(T, genes, exclude, n_per_type=30, min_frac=1e-5):
    """Top genes by log fold-change of each type vs. the mean of other types."""
    keep = ~np.isin(genes, exclude) & (T.max(0) > min_frac)
    logT = np.log(T + 1e-9)
    sel = set()
    for c in range(T.shape[0]):
        other = np.log(np.delete(T, c, axis=0).mean(0) + 1e-9)
        lfc = np.where(keep, logT[c] - other, -np.inf)
        sel.update(np.argsort(-lfc)[:n_per_type].tolist())
    return np.array(sorted(i for i in sel if keep[i]))


# ----------------------------------------------------------------------------- scref
def cmd_scref(a):
    import anndata
    ad = anndata.read_h5ad(a.sc_h5ad)
    if a.layer:
        X = ad.layers[a.layer]
    elif ad.raw is not None and a.use_raw:
        X = ad.raw.X
        ad = ad.raw.to_adata()
    else:
        X = ad.X
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(np.asarray(X))
    if X.data.size and np.abs(X.data - np.round(X.data)).max() > 1e-3:
        print("[warn] reference X does not look like raw counts; use --layer counts or --use_raw if available")
    labels = ad.obs[a.celltype_key].astype(str).values
    types, inv = np.unique(labels, return_inverse=True)
    counts = np.bincount(inv)
    ok = counts >= a.min_cells
    print("cell types:", dict(zip(types, counts)))
    if (~ok).any():
        print("dropping types with <", a.min_cells, "cells:", list(types[~ok]))
    # library-normalise each cell, then average per type (robust to deep cells)
    lib = np.asarray(X.sum(1)).ravel().clip(min=1.0)
    Xn = sp.diags(1.0 / lib) @ X
    M = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(types), len(inv)))
    T = np.asarray((M @ Xn).todense()) / counts[:, None]
    T, types = T[ok], types[ok]
    T = T / T.sum(1, keepdims=True)
    genes = _upper(ad.var_names)
    np.savez(a.out, T=T.astype(np.float32), genes=genes, cell_types=types)
    print(f"saved {a.out}: T {T.shape}")


# ----------------------------------------------------------------------------- deconv
def _deconv_slides(T, genes_T, types, a):
    targets = _upper(np.genfromtxt(os.path.join(a.data_dir, 'processed_data', a.gene_list_filename), dtype=str))
    os.makedirs(a.out_dir, exist_ok=True)
    slides = _read_list(a.slides)
    exclude = targets if not a.allow_target_genes else np.array([])
    for slide in slides:
        X, genes_st, barcodes = _load_st(a.data_dir, slide)
        common = np.intersect1d(genes_T, genes_st)
        iT = {g: i for i, g in enumerate(genes_T)}
        iS = {g: i for i, g in enumerate(genes_st)}
        Tc = T[:, [iT[g] for g in common]]
        mk = _select_markers(Tc, common, exclude, n_per_type=a.n_markers)
        mk_genes = common[mk]
        Y = np.asarray(X[:, [iS[g] for g in mk_genes]].todense(), dtype=np.float64)
        Tm = Tc[:, mk]
        Tm = Tm / Tm.sum(1, keepdims=True)
        A = _kl_nnls(Y, Tm, n_iter=a.n_iter)
        W = A / np.clip(A.sum(1, keepdims=True), 1e-10, None)
        W[Y.sum(1) == 0] = np.nan          # no marker counts -> unknown, masked in training
        pd.DataFrame(W, index=barcodes, columns=types).to_csv(os.path.join(a.out_dir, f'{slide}.csv'))
        print(f"{slide}: {len(barcodes)} spots, {len(mk_genes)} marker genes "
              f"(overlap with targets: {int(np.isin(mk_genes, targets).sum())}), "
              f"mean composition: {np.round(np.nanmean(W, 0), 3)}")


def cmd_deconv(a):
    d = np.load(a.prototypes, allow_pickle=True)
    _deconv_slides(np.asarray(d['T'], dtype=np.float64), _upper(d['genes']), [str(c) for c in d['cell_types']], a)


# ----------------------------------------------------------------------------- nmf
def cmd_nmf(a):
    targets = _upper(np.genfromtxt(os.path.join(a.data_dir, 'processed_data', a.gene_list_filename), dtype=str))
    train = _read_list([a.train_slides])
    mats, gsets = [], None
    for s in train:
        X, g, _ = _load_st(a.data_dir, s)
        mats.append((X, g))
        gsets = set(g) if gsets is None else gsets & set(g)
    genes = np.array(sorted(gsets))
    # gene set: targets + top-variance genes (by mean normalised expression)
    Ys = []
    for X, g in mats:
        idx = {gg: i for i, gg in enumerate(g)}
        Ys.append(np.asarray(X[:, [idx[x] for x in genes]].todense(), dtype=np.float64))
    Y = np.concatenate(Ys, 0)
    Y = Y[Y.sum(1) > 0]
    frac = Y / Y.sum(1, keepdims=True)
    disp = frac.var(0) / (frac.mean(0) + 1e-9)
    top = set(np.argsort(-disp)[:a.n_hvg].tolist()) | set(np.where(np.isin(genes, targets))[0].tolist())
    top = np.array(sorted(top))
    if a.max_spots and Y.shape[0] > a.max_spots:
        Y = Y[np.random.default_rng(0).choice(Y.shape[0], a.max_spots, replace=False)]
    print(f"NMF on {Y.shape[0]} training spots x {len(top)} genes, C={a.n_types}")
    _, T = _kl_nmf(Y[:, top], a.n_types, n_iter=a.n_iter_nmf)
    T = T / T.sum(1, keepdims=True)
    types = np.array([f'type{c}' for c in range(a.n_types)])
    np.savez(a.out, T=T.astype(np.float32), genes=genes[top], cell_types=types)
    print(f"saved {a.out}")
    # deconvolution with learned prototypes (target genes may be used here: T comes from train spots only)
    a.allow_target_genes = True
    _deconv_slides(T.astype(np.float64), genes[top], list(types), a)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('scref')
    s.add_argument('--sc_h5ad', required=True)
    s.add_argument('--celltype_key', required=True)
    s.add_argument('--layer', default=None, help='layer with raw counts (e.g. counts)')
    s.add_argument('--use_raw', action='store_true')
    s.add_argument('--min_cells', type=int, default=50)
    s.add_argument('--out', required=True)

    def _deconv_args(s):
        s.add_argument('--data_dir', required=True)
        s.add_argument('--gene_list_filename', required=True)
        s.add_argument('--slides', nargs='+', required=True, help='one or more slide-list txt files')
        s.add_argument('--out_dir', required=True)
        s.add_argument('--n_markers', type=int, default=30, help='markers per cell type')
        s.add_argument('--n_iter', type=int, default=300)
        s.add_argument('--allow_target_genes', action='store_true')

    s = sub.add_parser('deconv'); s.add_argument('--prototypes', required=True); _deconv_args(s)

    s = sub.add_parser('nmf')
    s.add_argument('--n_types', type=int, default=12)
    s.add_argument('--train_slides', required=True)
    s.add_argument('--n_hvg', type=int, default=2000)
    s.add_argument('--n_iter_nmf', type=int, default=500)
    s.add_argument('--max_spots', type=int, default=30000)
    s.add_argument('--out', required=True)
    _deconv_args(s)

    a = p.parse_args()
    {'scref': cmd_scref, 'deconv': cmd_deconv, 'nmf': cmd_nmf}[a.cmd](a)
