#!/usr/bin/env bash
# Train the released SAGE-guider with the original two-GPU hyperparameters.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

torchrun --standalone \
  --nproc_per_node "${NPROC_PER_NODE:-2}" \
  --master_port "${MASTER_PORT:-12345}" \
  "${CODE_ROOT}/sage_guider/main_pretrain.py" \
  --num_workers 8 \
  --batch_size 64 \
  --epochs 10 \
  --lr_decay_epochs 100 \
  --is_augmentation \
  --warmup_iterations 2000 \
  --use_vision_cls_token \
  --proj_dim 768 \
  --num_hidden_layers 2 \
  --aug_degrees 20 \
  --aug_scale 0.8 1.0 \
  --rad_dino_output_layer -1 \
  --use_counterfactual \
  --pos_sample_weight 4.0 \
  "$@"
