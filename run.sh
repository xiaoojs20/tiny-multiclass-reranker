#!/bin/bash
set -euo pipefail

########################
# 基本路径与模型设置
########################

# 基座模型（和 Qwen3-Reranker 一样，从 Base 起步）
BASE_MODEL="../llms/Qwen/Qwen3-0.6B-Base"

# 你的 ESCI parquet 路径
TRAIN_FILE="../datasets/esci-data/esci_multiclass_train.parquet"
EVAL_FILE="../datasets/esci-data/esci_multiclass_test.parquet"

# 微调输出（LoRA adapter）目录
OUTPUT_DIR="./outputs/qwen3_esci_reranker_lora"

########################
# 训练超参数（按需改）
########################

MAX_LEN=2048
BATCH_SIZE=4
GRAD_ACCUM=8
EPOCHS=2
LR=2e-4
WARMUP=0.03
LOGGING_STEPS=50
SAVE_STEPS=2000
SAVE_TOTAL_LIMIT=2

# LoRA 超参
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# 哪张卡训练
export CUDA_VISIBLE_DEVICES=0

########################
# 训练
########################

python train.py \
  --base_model "$BASE_MODEL" \
  --train_file "$TRAIN_FILE" \
  --eval_file "$EVAL_FILE" \
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
  --bf16

########################
# 评估
########################

python eval.py \
  --base_model "$BASE_MODEL" \
  --lora_model "$OUTPUT_DIR" \
  --eval_file "$EVAL_FILE" \
  --max_length $MAX_LEN \
  --batch_size 16
