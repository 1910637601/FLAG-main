"""
End-to-end data preparation for FLAG / FLAG-Proto from HEST-1k (huggingface: MahmoodLab/hest).

Produces exactly the layout the training code expects:
  $OUT/st/<id>.h5ad                         raw counts, obsm['spatial'], only spots that have an H&E patch
  $OUT/processed_data/hmhvg_geneformer_gene_list.txt
  $OUT/splits/{train,valid,test}.txt
  $OUT/features/<id>_uni.pt                 (N, 1, 1024) UNI features, same spot order as st/<id>.h5ad
  $OUT/gfm/<id>.csv                         Geneformer spot embedding (index = barcode)
  $OUT/gfm/<id>_gene_embeddings.npy         (N, G, D) Geneformer gene-token embeddings of the target genes
  $OUT/gfm/<id>_gene_mask.npy               (N, G) 1 if the gene was in the spot's Geneformer input

Prerequisites (on the server):
  pip install huggingface_hub h5py anndata scanpy "timm>=0.9.8" transformers
  huggingface-cli login     # your account must have been granted access to MahmoodLab/hest and MahmoodLab/UNI

Steps:
  python tools/prepare_hest.py list       --query "Squamous Cell Carcinoma"
  python tools/prepare_hest.py download   --query "Squamous Cell Carcinoma" --raw_dir $RAW
  python tools/prepare_hest.py organize   --raw_dir $RAW --out $OUT
  python tools/prepare_hest.py uni        --raw_dir $RAW --out $OUT --device cuda:0
  python tools/prepare_hest.py geneformer --out $OUT --device cuda:0        (can shard: --shard 0/4 ... 3/4)
  python tools/prepare_hest.py config     --out $OUT --result_dir $RES
"""
import argparse
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp

HEST_REPO = 'MahmoodLab/hest'
GF_REPO = 'ctheodoris/Geneformer'


# ----------------------------------------------------------------------------- metadata
def _meta(args):
    from huggingface_hub import HfApi, hf_hub_download
    if args.meta_csv:
        name = args.meta_csv
    else:
        files = HfApi().list_repo_files(HEST_REPO, repo_type='dataset')
        cands = sorted(f for f in files if f.startswith('HEST_v') and f.endswith('.csv'))
        if not cands:
            raise RuntimeError("no HEST_v*.csv metadata found in the repo; pass --meta_csv")
        name = cands[-1]
    path = hf_hub_download(HEST_REPO, name, repo_type='dataset')
    return pd.read_csv(path)


def _select(df, args):
    if args.ids:
        return df[df['id'].isin(args.ids)]
    q = args.query.lower()
    str_cols = [c for c in df.columns if df[c].dtype == object]
    hit = df[str_cols].apply(lambda col: col.astype(str).str.lower().str.contains(q, regex=False)).any(axis=1)
    sel = df[hit]
    if args.species:
        sel = sel[sel['species'].astype(str).str.contains(args.species, case=False)] if 'species' in sel else sel
    return sel


def cmd_list(args):
    df = _meta(args)
    sel = _select(df, args)
    cols = [c for c in ['id', 'dataset_title', 'organ', 'oncotree_code', 'st_technology', 'species',
                        'spots_under_tissue', 'pixel_size_um_estimated'] if c in sel.columns]
    pd.set_option('display.width', 250, 'display.max_colwidth', 70, 'display.max_rows', 500)
    print(sel[cols].to_string(index=False))
    print(f"\n{len(sel)} samples matched. Metadata columns: {list(df.columns)}")


def cmd_download(args):
    from huggingface_hub import snapshot_download
    sel = _select(_meta(args), args)
    ids = sel['id'].tolist()
    if not ids:
        raise SystemExit("nothing matched; run `list` first")
    print(f"downloading {len(ids)} samples: {ids}")
    pats = []
    for i in ids:
        pats += [f"st/{i}.h5ad", f"patches/{i}.h5", f"metadata/{i}.json"]
        if args.with_wsi:
            pats += [f"wsis/{i}.tif"]
    snapshot_download(HEST_REPO, repo_type='dataset', local_dir=args.raw_dir, allow_patterns=pats)
    os.makedirs(args.raw_dir, exist_ok=True)
    np.savetxt(os.path.join(args.raw_dir, 'ids.txt'), ids, fmt='%s')
    print("done ->", args.raw_dir)


# ----------------------------------------------------------------------------- organize
def _patch_barcodes(h5_path):
    import h5py
    with h5py.File(h5_path, 'r') as f:
        bc = f['barcode'][:]
    return np.array([b.decode() if isinstance(b, bytes) else str(b) for b in np.asarray(bc).reshape(-1)])


def _gf_dicts(cache=None):
    """Geneformer gene dictionaries (gc104M vocabulary, used by the V2 models)."""
    from huggingface_hub import hf_hub_download
    out = {}
    for key, fn in [('token', 'token_dictionary_gc104M.pkl'), ('median', 'gene_median_dictionary_gc104M.pkl'),
                    ('name2id', 'gene_name_id_dict_gc104M.pkl')]:
        p = hf_hub_download(GF_REPO, f'geneformer/{fn}', cache_dir=cache)
        with open(p, 'rb') as f:
            out[key] = pickle.load(f)
    return out


def cmd_organize(args):
    import anndata
    ids = [str(x) for x in np.atleast_1d(np.genfromtxt(os.path.join(args.raw_dir, 'ids.txt'), dtype=str))]
    for d in ['st', 'processed_data', 'splits']:
        os.makedirs(os.path.join(args.out, d), exist_ok=True)

    libs = {}
    for i in ids:
        ad = anndata.read_h5ad(os.path.join(args.raw_dir, 'st', f'{i}.h5ad'))
        ad.obs_names = ad.obs_names.astype(str)
        ad = ad[:, ~ad.var_names.duplicated()].copy()
        keep = set(_patch_barcodes(os.path.join(args.raw_dir, 'patches', f'{i}.h5')))
        m = ad.obs_names.isin(keep)
        if 'in_tissue' in ad.obs:
            m &= ad.obs['in_tissue'].astype(int).values == 1
        ad = ad[m].copy()
        X = ad.X.tocsr() if sp.issparse(ad.X) else sp.csr_matrix(np.asarray(ad.X))
        if X.data.size and np.abs(X.data - np.round(X.data)).max() > 1e-3:
            print(f"[warn] {i}: X does not look like raw counts")
        new = anndata.AnnData(X=X.astype(np.float32), obs=pd.DataFrame(index=ad.obs_names),
                              var=pd.DataFrame(index=ad.var_names.astype(str)))
        new.obsm['spatial'] = np.asarray(ad.obsm['spatial'], dtype=np.float64)
        new.write_h5ad(os.path.join(args.out, 'st', f'{i}.h5ad'))
        libs[i] = np.asarray(X.sum(1)).ravel()
        print(f"{i}: {new.n_obs} spots, {new.n_vars} genes, median counts/spot {np.median(libs[i]):.0f}")

    # ---- slide split (7:2:1 by default, at least one slide in valid/test) ----
    rng = np.random.default_rng(args.seed)
    perm = list(rng.permutation(ids))
    n = len(perm)
    n_test = max(1, int(round(n * args.test_frac)))
    n_val = max(1, int(round(n * args.val_frac)))
    test, valid, train = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]
    for name, lst in [('train', train), ('valid', valid), ('test', test)]:
        np.savetxt(os.path.join(args.out, 'splits', f'{name}.txt'), lst, fmt='%s')
        print(f"{name}: {lst}")

    # ---- gene list: highly-expressed + highly-variable genes, computed on TRAIN slides only,
    #      restricted to genes present in every slide and in the Geneformer vocabulary ----
    gf = _gf_dicts()
    vocab = {g.upper() for g, e in gf['name2id'].items() if e in gf['token']}
    common = None
    mats = []
    for i in ids:
        ad = anndata.read_h5ad(os.path.join(args.out, 'st', f'{i}.h5ad'))
        g = set(ad.var_names)
        common = g if common is None else common & g
        if i in train:
            mats.append(ad)
    genes = np.array(sorted(gg for gg in common if gg.upper() in vocab and not gg.upper().startswith(('MT-', 'RPL', 'RPS'))))
    Y = np.concatenate([np.asarray(ad[:, genes].X.todense()) for ad in mats], 0)
    lib = Y.sum(1, keepdims=True) + 1e-8
    L = np.log1p(Y / lib * 1e4)
    mean = L.mean(0)
    top_mean = np.argsort(-mean)[:args.n_top_mean]
    var = L[:, top_mean].var(0)
    sel = top_mean[np.argsort(-var)[:args.n_genes]]
    gene_list = genes[np.sort(sel)]
    np.savetxt(os.path.join(args.out, 'processed_data', 'hmhvg_geneformer_gene_list.txt'), gene_list, fmt='%s')
    print(f"selected {len(gene_list)} target genes -> processed_data/hmhvg_geneformer_gene_list.txt")


# ----------------------------------------------------------------------------- UNI features
def cmd_uni(args):
    import h5py
    import timm
    import torch
    import anndata
    ids = [os.path.basename(p)[:-5] for p in sorted(glob.glob(os.path.join(args.out, 'st', '*.h5ad')))]
    os.makedirs(os.path.join(args.out, 'features'), exist_ok=True)
    model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
    model = model.to(args.device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    for i in ids:
        outp = os.path.join(args.out, 'features', f'{i}_uni.pt')
        if os.path.exists(outp) and not args.overwrite:
            print("skip", i); continue
        bcs = anndata.read_h5ad(os.path.join(args.out, 'st', f'{i}.h5ad'), backed='r').obs_names.astype(str)
        pb = _patch_barcodes(os.path.join(args.raw_dir, 'patches', f'{i}.h5'))
        idx = {b: k for k, b in enumerate(pb)}
        order = [idx[b] for b in bcs]
        with h5py.File(os.path.join(args.raw_dir, 'patches', f'{i}.h5'), 'r') as f:
            imgs = f['img']
            feats = []
            with torch.no_grad():
                for s in range(0, len(order), args.batch):
                    batch = np.stack([imgs[k] for k in order[s:s + args.batch]])
                    x = torch.from_numpy(batch[..., :3]).permute(0, 3, 1, 2).float() / 255.0
                    x = ((x - mean) / std).to(args.device)
                    if x.shape[-1] != 224:
                        x = torch.nn.functional.interpolate(x, size=224, mode='bilinear', align_corners=False)
                    feats.append(model(x).float().cpu())
        feats = torch.cat(feats, 0).unsqueeze(1)       # (N, 1, 1024); dataset reads [:, 0, :]
        torch.save(feats, outp)
        print(f"{i}: UNI features {tuple(feats.shape)}")


# ----------------------------------------------------------------------------- Geneformer
def cmd_geneformer(args):
    import torch
    import anndata
    from huggingface_hub import snapshot_download
    from transformers import BertForMaskedLM

    gf = _gf_dicts()
    tok, med, n2id = gf['token'], gf['median'], gf['name2id']
    n2id_u = {k.upper(): v for k, v in n2id.items()}
    cls_id, eos_id, pad_id = tok.get('<cls>'), tok.get('<eos>'), tok.get('<pad>', 0)

    mdir = snapshot_download(GF_REPO, allow_patterns=[f'{args.gf_model}/*'])
    model = BertForMaskedLM.from_pretrained(os.path.join(mdir, args.gf_model), output_hidden_states=True)
    model = model.to(args.device).eval()
    if args.fp16:
        model = model.half()
    max_len = int(model.config.max_position_embeddings)
    D = int(model.config.hidden_size)
    print(f"Geneformer {args.gf_model}: hidden={D}, max_len={max_len}, emb layer={args.layer}")

    targets = np.genfromtxt(os.path.join(args.out, 'processed_data', 'hmhvg_geneformer_gene_list.txt'), dtype=str)
    G = len(targets)
    os.makedirs(os.path.join(args.out, 'gfm'), exist_ok=True)
    ids = [os.path.basename(p)[:-5] for p in sorted(glob.glob(os.path.join(args.out, 'st', '*.h5ad')))]
    k, n = (int(v) for v in args.shard.split('/'))
    ids = ids[k::n]

    for sid in ids:
        if os.path.exists(os.path.join(args.out, 'gfm', f'{sid}_gene_mask.npy')) and not args.overwrite:
            print("skip", sid); continue
        ad = anndata.read_h5ad(os.path.join(args.out, 'st', f'{sid}.h5ad'))
        X = ad.X.tocsr()
        genes_u = [g.upper() for g in ad.var_names]
        ens = [n2id_u.get(g) for g in genes_u]
        ok = np.array([e is not None and e in tok and e in med for e in ens])
        col = np.where(ok)[0]
        tok_ids = np.array([tok[ens[c]] for c in col])
        medians = np.array([med[ens[c]] for c in col], dtype=np.float64)
        tgt_tok = {tok[n2id_u[t.upper()]]: j for j, t in enumerate(targets) if n2id_u.get(t.upper()) in tok}
        n_counts = np.asarray(X.sum(1)).ravel()
        Xc = X[:, col]

        Nsp = ad.n_obs
        gene_emb = np.zeros((Nsp, G, D), dtype=np.float16)
        gene_mask = np.zeros((Nsp, G), dtype=np.float32)
        cell_emb = np.zeros((Nsp, D), dtype=np.float32)

        # rank-value encoding (Geneformer): counts / total * 1e4 / gene median, sorted desc
        seqs = []
        n_special = (cls_id is not None) + (eos_id is not None)
        for s in range(Nsp):
            row = Xc[s]
            nz, vals = row.indices, row.data
            norm = vals / max(n_counts[s], 1.0) * 1e4 / medians[nz]
            order = nz[np.argsort(-norm, kind='stable')][:max_len - n_special]
            seq = tok_ids[order].tolist()
            if cls_id is not None:
                seq = [cls_id] + seq
            if eos_id is not None:
                seq = seq + [eos_id]
            seqs.append(seq)

        lengths = np.array([len(q) for q in seqs])
        order_sp = np.argsort(-lengths)
        with torch.no_grad():
            for b in range(0, Nsp, args.batch):
                idx = order_sp[b:b + args.batch]
                L = int(lengths[idx].max())
                inp = torch.full((len(idx), L), pad_id, dtype=torch.long)
                att = torch.zeros((len(idx), L), dtype=torch.long)
                for r, s in enumerate(idx):
                    inp[r, :lengths[s]] = torch.tensor(seqs[s])
                    att[r, :lengths[s]] = 1
                out = model(input_ids=inp.to(args.device), attention_mask=att.to(args.device))
                H = out.hidden_states[args.layer].float().cpu().numpy()      # (B, L, D)
                for r, s in enumerate(idx):
                    seq = seqs[s]
                    start = 1 if cls_id is not None else 0
                    stop = lengths[s] - (1 if eos_id is not None else 0)
                    cell_emb[s] = H[r, start:stop].mean(0) if stop > start else 0.0
                    for p in range(start, stop):
                        j = tgt_tok.get(seq[p])
                        if j is not None:
                            gene_emb[s, j] = H[r, p]
                            gene_mask[s, j] = 1.0

        bc = ad.obs_names.astype(str)
        pd.DataFrame(cell_emb, index=bc, columns=[str(c) for c in range(D)]).to_csv(os.path.join(args.out, 'gfm', f'{sid}.csv'))
        np.save(os.path.join(args.out, 'gfm', f'{sid}_gene_embeddings.npy'), gene_emb)
        np.save(os.path.join(args.out, 'gfm', f'{sid}_gene_mask.npy'), gene_mask)
        print(f"{sid}: {Nsp} spots, target-gene coverage {gene_mask.mean():.3f}")


# ----------------------------------------------------------------------------- config
def cmd_config(args):
    import yaml
    import anndata
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    c = yaml.safe_load(open(os.path.join(here, 'configs', 'graph_proto_diffusion.yaml'), encoding='utf-8'))
    out = os.path.abspath(args.out)
    dc = c['data_config']
    dc.update(data_dir=out, gene_list_filename='hmhvg_geneformer_gene_list.txt',
              llm_embedding_dir=os.path.join(out, 'gfm'), feature_dir=os.path.join(out, 'features'),
              train_slides_list=os.path.join(out, 'splits', 'train.txt'),
              valid_slides_list=os.path.join(out, 'splits', 'valid.txt'),
              test_slides_list=os.path.join(out, 'splits', 'test.txt'),
              celltype_dir=os.path.join(out, 'celltype'))
    any_npy = sorted(glob.glob(os.path.join(out, 'gfm', '*_gene_embeddings.npy')))
    if any_npy:
        c['model_config']['z_dim'] = int(np.load(any_npy[0], mmap_mode='r').shape[-1])
    c['model_config']['prototype_path'] = os.path.join(out, 'prototypes.npz')
    train = np.atleast_1d(np.genfromtxt(dc['train_slides_list'], dtype=str))
    libs = np.concatenate([np.asarray(anndata.read_h5ad(os.path.join(out, 'st', f'{s}.h5ad')).X.sum(1)).ravel()
                           for s in train])
    c['model_config']['log_lib_init'] = float(np.log(np.median(libs[libs > 0])))
    c['train_config']['result_dir'] = os.path.abspath(args.result_dir)
    c['exp_name'] = args.exp_name
    dst = os.path.join(here, 'configs', f'proto_{args.name}.yaml')
    yaml.safe_dump(c, open(dst, 'w', encoding='utf-8'), sort_keys=False, allow_unicode=True)
    print(f"wrote {dst}  (z_dim={c['model_config']['z_dim']}, log_lib_init={c['model_config']['log_lib_init']:.2f})")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)

    def sel_args(s):
        s.add_argument('--query', default='Squamous Cell Carcinoma', help='case-insensitive match on any text column')
        s.add_argument('--ids', nargs='*', default=None, help='explicit HEST sample ids (overrides --query)')
        s.add_argument('--species', default='Homo sapiens')
        s.add_argument('--meta_csv', default=None)

    s = sub.add_parser('list'); sel_args(s)
    s = sub.add_parser('download'); sel_args(s)
    s.add_argument('--raw_dir', required=True)
    s.add_argument('--with_wsi', action='store_true', help='also download the (large) WSIs; not needed for training')

    s = sub.add_parser('organize')
    s.add_argument('--raw_dir', required=True); s.add_argument('--out', required=True)
    s.add_argument('--seed', type=int, default=42)
    s.add_argument('--val_frac', type=float, default=0.2); s.add_argument('--test_frac', type=float, default=0.1)
    s.add_argument('--n_genes', type=int, default=200); s.add_argument('--n_top_mean', type=int, default=1000)

    s = sub.add_parser('uni')
    s.add_argument('--raw_dir', required=True); s.add_argument('--out', required=True)
    s.add_argument('--device', default='cuda:0'); s.add_argument('--batch', type=int, default=128)
    s.add_argument('--overwrite', action='store_true')

    s = sub.add_parser('geneformer')
    s.add_argument('--out', required=True); s.add_argument('--device', default='cuda:0')
    s.add_argument('--gf_model', default='Geneformer-V2-316M', help='subfolder of ctheodoris/Geneformer')
    s.add_argument('--layer', type=int, default=-2, help='hidden-state layer (Geneformer default: penultimate)')
    s.add_argument('--batch', type=int, default=4); s.add_argument('--fp16', action='store_true', default=True)
    s.add_argument('--shard', default='0/1', help='k/n: process every n-th slide starting at k')
    s.add_argument('--overwrite', action='store_true')

    s = sub.add_parser('config')
    s.add_argument('--out', required=True); s.add_argument('--result_dir', required=True)
    s.add_argument('--name', default='cscc'); s.add_argument('--exp_name', default='proto_full')

    a = p.parse_args()
    {'list': cmd_list, 'download': cmd_download, 'organize': cmd_organize, 'uni': cmd_uni,
     'geneformer': cmd_geneformer, 'config': cmd_config}[a.cmd](a)
