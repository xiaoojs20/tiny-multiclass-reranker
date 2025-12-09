#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

BASE_MODEL="../llms/Qwen/Qwen3-Embedding-0.6B"
TRAIN_FILE="../datasets/esci-data/esci_multiclass_train.parquet"
OUTPUT_DIR="./outputs/emb_esci_cls"

wandb_project="esci-emb-cls"
wandb_run_name="qwen3-emb-0.6b-esci-lora-qv-cls"

python train_emb_cls.py \
  --base_model "$BASE_MODEL" \
  --train_file "$TRAIN_FILE" \
  --output_dir "$OUTPUT_DIR" \
  --max_length 512 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 1.0 \
  --learning_rate 1e-4 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 50 \
  --save_steps 2000 \
  --save_total_limit 2 \
  --bf16 \
  --eval_steps 200 \
  --eval_ratio 0.05 \
  --report_to wandb \
  --wandb_project $wandb_project \
  --wandb_run_name $wandb_run_name

#   --gradient_checkpointing \
# nohup bash run_train.sh > logs/train_lora.log 2>&1 &