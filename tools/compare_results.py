"""
Compare test predictions of several runs on the same test slides / genes.

  python tools/compare_results.py --data_dir $DATA --ref FLAG \
      --runs FLAG=$RES/repa_graph_fixed/.../preds_best \
             Proto=$RES/proto_graph_fixed/.../proto_full/preds_best \
             Proto_noprior=$RES/proto_graph_fixed/.../no_prior/preds_best \
             Prior_only=$RES/proto_graph_fixed/.../proto_full/prior_best

Metrics (all on log2(1+x), pooled over test spots, same as utils/metric.py):
  PCC-all / PCC-50 / MSE / MAE
  GeneStruct : Pearson between predicted and true gene-gene correlation matrices (upper triangle)
  SpatStruct : Pearson between predicted and true per-gene Moran's I (kNN graph), averaged over slides
  (GeneStruct / SpatStruct are our implementation of the gene-/spatial-structure idea behind the paper's
   GSC / SSC; the paper's exact formulas are not in this repo, so use them to compare runs with each other,
   not to compare against numbers printed in the paper.)
  p_vs_ref   : paired Wilcoxon signed-rank test on per-gene PCC vs. the --ref run
"""
import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from scipy.spatial import cKDTree


def _gt(data_dir, slide, genes):
    import anndata
    ad = anndata.read_h5ad(os.path.join(data_dir, 'st', f'{slide}.h5ad'))
    X = ad[:, list(genes)].X
    X = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
    df = pd.DataFrame(np.log2(X + 1.0), index=ad.obs_names.astype(str), columns=genes)
    xy = pd.DataFrame(np.asarray(ad.obsm['spatial'])[:, :2], index=df.index, columns=['x', 'y'])
    return df, xy


def _pcc_per_gene(p, g):
    pc, gc = p - p.mean(0), g - g.mean(0)
    den = np.sqrt((pc ** 2).sum(0) * (gc ** 2).sum(0))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, (pc * gc).sum(0) / den, np.nan)


def _morans_i(V, xy, k=6):
    n = len(xy)
    if n <= k + 1:
        return np.full(V.shape[1], np.nan)
    _, nb = cKDTree(xy).query(xy, k=k + 1)
    nb = nb[:, 1:]
    Z = V - V.mean(0)
    num = (Z[:, None, :] * Z[nb]).sum(1).sum(0) / (n * k)
    den = (Z ** 2).sum(0) / n
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / den, np.nan)


def _corr(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 2 else np.nan


def evaluate(pred_dir, data_dir, slides, genes, gt_cache):
    P, G, mor = [], [], []
    for s in slides:
        f = os.path.join(pred_dir, f'{s}.csv')
        if not os.path.exists(f):
            raise FileNotFoundError(f)
        pr = pd.read_csv(f, index_col=0)
        pr.index = pr.index.astype(str)
        if s not in gt_cache:
            gt_cache[s] = _gt(data_dir, s, genes)
        gt, xy = gt_cache[s]
        bc = pr.index.intersection(gt.index)
        p, g = pr.loc[bc, genes].values.astype(np.float64), gt.loc[bc, genes].values
        P.append(p); G.append(g)
        mor.append(_corr(_morans_i(p, xy.loc[bc].values), _morans_i(g, xy.loc[bc].values)))
    P, G = np.concatenate(P), np.concatenate(G)
    pcc = _pcc_per_gene(P, G)
    v = np.sort(pcc[~np.isnan(pcc)])[::-1]
    iu = np.triu_indices(len(genes), 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        cp, cg = np.corrcoef(P.T)[iu], np.corrcoef(G.T)[iu]
    return {
        'n_spots': len(P),
        'PCC-all': v.mean(), 'PCC-50': v[:50].mean(), 'PCC-10': v[:10].mean(),
        'MSE': ((P - G) ** 2).mean(), 'MAE': np.abs(P - G).mean(),
        'GeneStruct': _corr(cp, cg), 'SpatStruct': np.nanmean(mor),
    }, pcc


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--gene_list_filename', default='hmhvg_geneformer_gene_list.txt')
    ap.add_argument('--test_list', default=None, help='default: <data_dir>/splits/test.txt')
    ap.add_argument('--runs', nargs='+', required=True, help='name=pred_dir (dir of <slide>.csv)')
    ap.add_argument('--ref', default=None, help='run name used as reference for the paired test')
    ap.add_argument('--out', default='compare_results.csv')
    a = ap.parse_args()

    genes = [str(g) for g in np.genfromtxt(os.path.join(a.data_dir, 'processed_data', a.gene_list_filename), dtype=str)]
    tl = a.test_list or os.path.join(a.data_dir, 'splits', 'test.txt')
    slides = [str(s) for s in np.atleast_1d(np.genfromtxt(tl, dtype=str))]

    cache, rows, per_gene = {}, {}, {}
    for r in a.runs:
        name, path = r.split('=', 1)
        rows[name], per_gene[name] = evaluate(path, a.data_dir, slides, genes, cache)
    ref = a.ref or list(rows)[0]
    for name in rows:
        if name == ref:
            rows[name]['p_vs_ref'] = np.nan
            continue
        x, y = per_gene[name], per_gene[ref]
        m = ~(np.isnan(x) | np.isnan(y))
        rows[name]['dPCC_vs_ref'] = np.nanmean(x[m] - y[m])
        rows[name]['genes_better'] = f"{int((x[m] > y[m]).sum())}/{int(m.sum())}"
        rows[name]['p_vs_ref'] = wilcoxon(x[m], y[m]).pvalue if m.sum() > 0 else np.nan

    df = pd.DataFrame(rows).T
    pd.set_option('display.width', 200, 'display.float_format', lambda v: f'{v:.4f}')
    print(f"test slides: {slides}   reference run: {ref}\n")
    print(df.to_string())
    df.to_csv(a.out)
    pd.DataFrame(per_gene, index=genes).to_csv(os.path.splitext(a.out)[0] + '_per_gene_pcc.csv')
    print(f"\nsaved {a.out}")
