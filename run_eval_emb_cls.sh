#!/bin/bash
set -e

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1

ARCH=encoder # ["decoder", "encoder"]
# BASE_MODEL="../llms/Qwen/Qwen3-0.6B"
# BASE_MODEL="../llms/Qwen/Qwen3-Embedding-0.6B"
BASE_MODEL="../llms/intfloat/multilingual-e5-large"
# CKPT_DIR="./outputs/emb_esci_cls/checkpoint-8000"
CKPT_DIR="./outputs/emb_esci_cls-multilingual-e5-large-lora-qv/checkpoint-2944"
EVAL_FILE="../datasets/esci-data/esci_multiclass_test.parquet"
# TARGET_MODULES="gate_proj up_proj down_proj" # full="q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"
TARGET_MODULES="query value" # for E5 model full="query key value dense"

SEED=42
EVAL_BSZ=256
NUM_LABELS=4

python eval_emb_cls.py \
  --arch $ARCH \
  --base_model $BASE_MODEL \
  --checkpoint_dir $CKPT_DIR \
  --eval_file $EVAL_FILE \
  --eval_ratio 0.05 \
  --seed $SEED \
  --per_device_eval_batch_size $EVAL_BSZ \
  --target_modules $TARGET_MODULES \
  --num_labels $NUM_LABELS \
  --normalize_emb \
  --no_flash_attn \
  --bf16
