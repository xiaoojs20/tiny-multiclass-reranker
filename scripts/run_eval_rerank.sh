export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1

BASE_MODEL="../llms/Qwen/Qwen3-0.6B"
EVAL_FILE="../datasets/esci-data/esci_multiclass_test.parquet"
# OUTPUT_DIR="./outputs/qwen3_esci_reranker_lora"
OUTPUT_DIR=""
MAX_LEN=512
BATCH_SIZE=16
SAVE_PATH="./outputs/esci_eval_scores.npz"

# torchrun --nnodes=1 --nproc_per_node=2 \

python eval_rerank.py \
    --base_model "$BASE_MODEL" \
    --lora_model "$OUTPUT_DIR" \
    --eval_file "$EVAL_FILE" \
    --max_length $MAX_LEN \
    --batch_size $BATCH_SIZE \
    --save_scores_path $SAVE_PATH \
    --bf16

