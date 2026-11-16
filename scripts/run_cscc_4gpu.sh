#!/usr/bin/env bash
# FLAG-Proto, full model only, one dataset (default: cSCC from HEST-1k), 4x V100.
#
#   bash scripts/run_cscc_4gpu.sh list      # 1) look at which HEST samples match, set IDS below if needed
#   bash scripts/run_cscc_4gpu.sh prep      # 2) download + preprocess (UNI, Geneformer, cell-type prior, config)
#   bash scripts/run_cscc_4gpu.sh train     # 3) 4-GPU DDP training of the full model
#   bash scripts/run_cscc_4gpu.sh test      # 4) test best.pt on the test slides
#
# One-time setup:
#   pip install huggingface_hub h5py anndata scanpy "timm>=0.9.8" transformers einops torch_geometric tensorboard pyyaml
#   huggingface-cli login     # account needs approved access to MahmoodLab/hest and MahmoodLab/UNI
set -e
cd "$(dirname "$0")/.."

WORK=${WORK:-$HOME/flag_work}            # all data / results go here
RAW=$WORK/hest_raw
DATA=$WORK/cscc
RES=$WORK/results
QUERY=${QUERY:-"Squamous Cell Carcinoma"}
IDS=${IDS:-""}                           # e.g. IDS="NCBI759 NCBI760 ..." to pin exact samples
N_TYPES=${N_TYPES:-12}
CFG=configs/proto_cscc.yaml
SEL=$([ -n "$IDS" ] && echo "--ids $IDS" || echo "--query \"$QUERY\"")

case "$1" in
list)
  eval python tools/prepare_hest.py list $SEL
  ;;
prep)
  eval python tools/prepare_hest.py download $SEL --raw_dir $RAW
  python tools/prepare_hest.py organize --raw_dir $RAW --out $DATA
  python tools/prepare_hest.py uni --raw_dir $RAW --out $DATA --device cuda:0
  for k in 0 1 2 3; do       # Geneformer on 4 GPUs in parallel (slides sharded)
    python tools/prepare_hest.py geneformer --out $DATA --device cuda:$k --shard $k/4 &
  done
  wait
  # cell-type prior: NMF pseudo cell types learned on TRAIN spots only (no scRNA download needed)
  python tools/prepare_celltype_prior.py nmf --n_types $N_TYPES --data_dir $DATA \
      --gene_list_filename hmhvg_geneformer_gene_list.txt --train_slides $DATA/splits/train.txt \
      --slides $DATA/splits/train.txt $DATA/splits/valid.txt $DATA/splits/test.txt \
      --out $DATA/prototypes.npz --out_dir $DATA/celltype
  python tools/prepare_hest.py config --out $DATA --result_dir $RES --name cscc --exp_name proto_full
  ;;
train)
  mkdir -p logs
  torchrun --standalone --nproc_per_node 4 main.py --config $CFG 2>&1 | tee logs/proto_cscc_train.log
  ;;
test)
  python main.py --config $CFG --device cuda:0 --pipeline test_graph_latent_diffusion --ckpt best.pt \
      2>&1 | tee logs/proto_cscc_test.log
  ;;
*)
  echo "usage: $0 {list|prep|train|test}"; exit 1 ;;
esac
