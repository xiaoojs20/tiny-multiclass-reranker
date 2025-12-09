#!/bin/bash
set -euo pipefail

########################
# 基本路径与模型设置
########################

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1

BASE_MODEL="../llms/Qwen/Qwen3-0.6B"
TRAIN_FILE="../datasets/esci-data/esci_multiclass_train.parquet"
EVAL_FILE="../datasets/esci-data/esci_multiclass_test.parquet"
OUTPUT_DIR="./outputs/qwen3_esci_reranker_lora"


MAX_LEN=512
BATCH_SIZE=8
GRAD_ACCUM=4
EPOCHS=1
LR=2e-4
WARMUP=0.03
LOGGING_STEPS=50
SAVE_STEPS=2000
SAVE_TOTAL_LIMIT=2

# LoRA 超参
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

wandb_project="qwen3-multiclass-reranker"
wandb_run_name="qwen3-esci-lora-v1"

# --eval_file "$EVAL_FILE" \

python train.py \
  --base_model "$BASE_MODEL" \
  --train_file "$TRAIN_FILE" \
  --eval_ratio 0.05 \
  --output_dir "$OUTPUT_DIR" \
  --max_length $MAX_LEN \
  --per_device_train_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps $GRAD_ACCUM \
  --num_train_epochs $EPOCHS \
  --learning_rate $LR \
  --warmup_ratio $WARMUP \
  --logging_steps $LOGGING_STEPS \
  --save_steps $SAVE_STEPS \
  --save_total_limit $SAVE_TOTAL_LIMIT \
  --lora_r $LORA_R \
  --lora_alpha $LORA_ALPHA \
  --lora_dropout $LORA_DROPOUT \
  --bf16 \
  --report_to wandb \
  --wandb_project $wandb_project \
  --wandb_run_name $wandb_run_name
