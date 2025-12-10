#!/bin/bash
set -e

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1

# BASE_MODEL="../llms/Qwen/Qwen3-0.6B"
BASE_MODEL="../llms/Qwen/Qwen3-Embedding-0.6B"
CKPT_DIR="./outputs/emb_esci_cls/checkpoint-8000"
EVAL_FILE="../datasets/esci-data/esci_multiclass_test.parquet"


python eval_emb_cls.py \
  --base_model $BASE_MODEL \
  --checkpoint_dir $CKPT_DIR \
  --eval_file $EVAL_FILE \
  --eval_ratio 0.05 \
  --seed 42 \
  --per_device_eval_batch_size 256 \
  --no_flash_attn \
  --bf16
