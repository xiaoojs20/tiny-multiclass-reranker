#!/bin/bash
set -e

MODEL_DIR="./outputs/emb_esci_cls"
EVAL_FILE="../datasets/esci-data/esci_multiclass_test.parquet"

python eval_cls.py \
  --model_dir "$MODEL_DIR" \
  --eval_file "$EVAL_FILE" \
  --max_length 512 \
  --per_device_eval_batch_size 64 \
  --bf16 \
  --output_metrics_file "$MODEL_DIR/eval_metrics_test.json"
