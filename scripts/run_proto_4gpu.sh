#!/usr/bin/env bash
# FLAG-Proto on 4x V100: the pipeline uses batch_size=1 (one tile per step), so instead of DDP we
# run 4 independent experiments in parallel, one per GPU (main result + key ablations).
#
# Step 0 (once): build the cell-type prior. Choose ONE of:
#   (a) with a scRNA reference (HER2ST/STNet: GSE176078, cSCC: GSE144240), raw counts + cell-type column
#   python tools/prepare_celltype_prior.py scref --sc_h5ad ref.h5ad --celltype_key celltype_major --out $DATA/prototypes.npz
#   python tools/prepare_celltype_prior.py deconv --prototypes $DATA/prototypes.npz --data_dir $DATA \
#          --gene_list_filename hmhvg_geneformer_gene_list.txt --slides $TRAIN $VALID $TEST --out_dir $DATA/celltype
#   (b) no reference: NMF pseudo cell types learned on training spots only
#   python tools/prepare_celltype_prior.py nmf --n_types 12 --data_dir $DATA --gene_list_filename hmhvg_geneformer_gene_list.txt \
#          --train_slides $TRAIN --slides $TRAIN $VALID $TEST --out $DATA/prototypes.npz --out_dir $DATA/celltype
set -e
CFG=${CFG:-configs/graph_proto_diffusion.yaml}
BASE_CFG=${BASE_CFG:-configs/graph_latent_diffusion.yaml}
mkdir -p logs

# GPU0: full FLAG-Proto
python main.py --config $CFG --device cuda:0 --exp_name proto_full                        > logs/proto_full.log 2>&1 &
# GPU1: original FLAG baseline (same data / split / seed)
python main.py --config $BASE_CFG --device cuda:1                                          > logs/flag_base.log 2>&1 &
# GPU2: ablation - same network + EDM preconditioning, but no cell-type prior (isolates the prior's gain)
python main.py --config $CFG --device cuda:2 --exp_name no_prior --set model_config.use_prior=false \
       model_config.use_proto_gene_embed=false                                             > logs/no_prior.log 2>&1 &
# GPU3: ablation - prior without deconvolution supervision (composition learned only through the NB likelihood)
python main.py --config $CFG --device cuda:3 --exp_name no_comp --set train_config.proto_loss.lambda_comp=0 \
                                                                                           > logs/no_comp.log 2>&1 &
wait

# testing (best.pt chosen on validation PCC)
for EXP in proto_full no_prior no_comp; do
  python main.py --config $CFG --device cuda:0 --exp_name $EXP --pipeline test_graph_latent_diffusion --ckpt best.pt
done
python main.py --config $BASE_CFG --device cuda:1 --pipeline test_graph_latent_diffusion --ckpt <set_epoch_xxxx.pt>
