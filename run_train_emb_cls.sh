#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

BASE_MODEL="../llms/Qwen/Qwen3-Embedding-0.6B"
TRAIN_FILE="../datasets/esci-data/esci_multiclass_train.parquet"
OUTPUT_DIR="./outputs/emb_esci_cls"
MAX_LEN=512
LR=1e-4
WARMUP=0.03
SCHEDULER="cosine"
SAVE_TOTAL_LIMIT=-1

# LoRA 超参
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

EPOCHS=1.0
LOGGING_STEPS=100
SAVE_STEPS=2000
EVAL_STEPS=2000
EVAL_RATIO=0.05

wandb_project="esci-emb-cls"
wandb_run_name="qwen3-emb-0.6b-esci-lora-mlp"

TRAIN_BSZ=16
EVAL_BSZ=32
GRAD_ACCUM=2

python train_emb_cls.py \
  --base_model "$BASE_MODEL" \
  --train_file "$TRAIN_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --max_length $MAX_LEN \
  --per_device_train_batch_size $TRAIN_BSZ \
  --per_device_eval_batch_size $EVAL_BSZ \
  --gradient_accumulation_steps $GRAD_ACCUM \
  --num_train_epochs $EPOCHS \
  --learning_rate $LR \
  --warmup_ratio $WARMUP \
  --lr_scheduler_type $SCHEDULER \
  --lora_r $LORA_R \
  --lora_alpha $LORA_ALPHA \
  --lora_dropout $LORA_DROPOUT \
  --target_modules gate_proj up_proj down_proj \
  --logging_steps $LOGGING_STEPS \
  --save_steps $SAVE_STEPS \
  --save_total_limit $SAVE_TOTAL_LIMIT \
  --bf16 \
  --eval_steps $EVAL_STEPS \
  --eval_ratio $EVAL_RATIO \
  --report_to wandb \
  --wandb_project $wandb_project \
  --wandb_run_name $wandb_run_name

#   --gradient_checkpointing \
# nohup bash run_train.sh > logs/train_emb_cls_lora_mlp.log 2>&1 &