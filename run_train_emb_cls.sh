#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

ARCH=encoder # ["decoder", "encoder"]
# BASE_MODEL="../llms/Qwen/Qwen3-Embedding-0.6B" # [Qwen/Qwen3-Embedding-0.6B, intfloat/multilingual-e5-large]
BASE_MODEL="../llms/intfloat/multilingual-e5-large"
TRAIN_FILE="../datasets/esci-data/esci_multiclass_train.parquet"
MODEL_BASE_NAME=$(basename "$BASE_MODEL") 
OUTPUT_DIR="./outputs/emb_esci_cls-${MODEL_BASE_NAME}-lora-qv"
MAX_LEN=512
LR=1e-4
WARMUP=0.03
SCHEDULER="cosine"
SAVE_TOTAL_LIMIT=-1


# LoRA 超参
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
# TARGET_MODULES="gate_proj up_proj down_proj" # full="q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"
TARGET_MODULES="query value" # for E5 model full="query key value dense"

NUM_LABELS=4

EPOCHS=1.0
LOGGING_STEPS=100
SAVE_STEPS=4000
EVAL_STEPS=2000
DATA_RATIO=0.30
EVAL_RATIO=0.05

wandb_project="esci-emb-cls"
wandb_run_name="${MODEL_BASE_NAME}-lora-qv"

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
  --data_ratio $DATA_RATIO \
  --eval_ratio $EVAL_RATIO \
  --lora_r $LORA_R \
  --lora_alpha $LORA_ALPHA \
  --lora_dropout $LORA_DROPOUT \
  --target_modules $TARGET_MODULES \
  --num_labels $NUM_LABELS \
  --normalize_emb \
  --logging_steps $LOGGING_STEPS \
  --save_steps $SAVE_STEPS \
  --save_total_limit $SAVE_TOTAL_LIMIT \
  --bf16 \
  --eval_steps $EVAL_STEPS \
  --no_flash_attn \
  --report_to wandb \
  --wandb_project $wandb_project \
  --wandb_run_name $wandb_run_name

#   --gradient_checkpointing \
# nohup bash run_train.sh > logs/train_emb_cls_lora_mlp.log 2>&1 &

# nohup bash run_train_emb_cls.sh > logs/train_emb_cls_lora_mlp.log 2>&1 &