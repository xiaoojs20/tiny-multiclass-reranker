# tiny-reranker: Qwen3-0.6B ESCI Multi-class Reranker

This repo fine-tunes **Qwen3-0.6B** with **LoRA** on the **Amazon ESCI** dataset to build a tiny but practical **4-way relevance reranker** for e-commerce search.

- Discrete labels: **E / S / C / I**
  - **E** – exact
  - **S** – substitute
  - **C** – complement
  - **I** – irrelevant
- Plus a continuous relevance score in **[0, 1]** derived from the 4-way probabilities.

The training style follows **Qwen3-Reranker**: chat-style prompts + last-token prediction.

---

## 🔍 What This Project Does

- Fine-tunes **Qwen3-0.6B** as an ESCI-aware reranker.
- Uses **LoRA** for parameter-efficient fine-tuning.
- Uses **chat-format prompts** and predicts exactly one of:
  - `"exact"`, `"substitute"`, `"complement"`, `"irrelevant"`.
- Computes:
  - 4-way probabilities `P(E), P(S), P(C), P(I)`
  - A scalar relevance score:
    $
    \text{score} = \frac{3P(E) + 2P(S) + 1P(C) + 0P(I)}{3}
    $
  - This score can be used as ranking signal / for AUC metrics.

---

## 📂 Project Structure

```text
tiny-reranker/
├── train.py          # Fine-tune Qwen3-0.6B with LoRA on ESCI
├── eval.py           # Evaluate 4-way classification + relevance score
├── esci_dataset.py   # ESCI dataset loading & prompt formatting
├── scripts/
│   ├── run_train.sh  # Example training script
│   └── run_eval.sh   # Example eval script
├── README.md
└── README_zh.md
```

Key pieces:
- esci_dataset.py
    - load_esci_parquet(path): load preprocessed parquet file.
    - ESCIMultiClassRerankDataset: builds chat-style inputs and labels.
    - LABEL_TEXT = {"E": "exact", "S": "substitute", "C": "complement", "I": "irrelevant"}.
    - SYSTEM_PROMPT, INSTRUCT, format_instruction(...): Qwen3-Reranker style formatting.
- train.py
    - Loads base model Qwen3-0.6B.
	- Applies LoRA on attention projection layers (e.g. q_proj, v_proj).
	- Reads a single --train_file (ESCI parquet) and splits into train/eval by --eval_ratio.
	- Uses transformers.Trainer for training.
	- Optional logging to Weights & Biases (wandb).
- eval.py
	- Loads base model + LoRA adapter.
	- Builds prompts from query and item_text.
	- Takes last-token logits, extracts logits for the 4 label tokens and computes:
	- 4-way softmax over [E, S, C, I]
	- Predicted label (argmax)
	- Relevance score in [0,1]

---

## 🧱 Data Preparation

We assume you have preprocessed Amazon ESCI into parquet files:
- Columns (minimal):
    - query
	- item_text  (e.g. title + brand + attributes / short description)
	- esci_label ∈ {E, S, C, I}

Example:

```python
from pathlib import Path
import pandas as pd

OUT_DIR = Path("../datasets/esci-data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df_train = ...  # ESCI training split
df_test = ...   # ESCI test split

df_train.to_parquet(OUT_DIR / "esci_multiclass_train.parquet", index=False)
df_test.to_parquet(OUT_DIR / "esci_multiclass_test.parquet", index=False)
```

Typical paths used in this repo:
- ../datasets/esci-data/esci_multiclass_train.parquet
- ../datasets/esci-data/esci_multiclass_test.parquet

--- 

## ⚙️ Environment Setup

```bash
conda create -n tiny-reranker python=3.10 -y
conda activate tiny-reranker

Install dependencies (adjust CUDA version as needed):

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft accelerate datasets pandas pyarrow scikit-learn tqdm
pip install wandb  # optional

# If flash-attn is available:

pip install flash-attn --no-build-isolation

# If not available, run scripts with --no_flash_attn.
```

---

## 🚀 Training

Single-GPU Example

```bash
BASE_MODEL="../llms/Qwen/Qwen3-0.6B"
TRAIN_FILE="../datasets/esci-data/esci_multiclass_train.parquet"
OUTPUT_DIR="./outputs/qwen3_esci_reranker_lora"

MAX_LEN=512
BATCH_SIZE=4
GRAD_ACCUM=8
EPOCHS=1
LR=2e-4
WARMUP=0.03
LOGGING_STEPS=50
SAVE_STEPS=2000
SAVE_TOTAL_LIMIT=2
EVAL_RATIO=0.05   # use 5% of data as eval split

python train.py \
  --base_model "$BASE_MODEL" \
  --train_file "$TRAIN_FILE" \
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
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --eval_ratio $EVAL_RATIO \
  --bf16
```

Notes:
- --eval_ratio: fraction of the training file used as validation.
- --save_total_limit: keep at most N checkpoints (older ones are deleted).
- --bf16: use bfloat16 if supported.

Multi-GPU (DDP via torchrun)

```bash
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

torchrun --nnodes=1 --nproc_per_node=2 \
  train.py \
  --base_model "$BASE_MODEL" \
  --train_file "$TRAIN_FILE" \
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
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --eval_ratio $EVAL_RATIO \
  --bf16
```

---

## 📈 Evaluation

Basic Usage

```bash
BASE_MODEL="../llms/Qwen/Qwen3-0.6B"
EVAL_FILE="../datasets/esci-data/esci_multiclass_test.parquet"
LORA_DIR="./outputs/qwen3_esci_reranker_lora"

MAX_LEN=512
BATCH_SIZE=16

python eval.py \
  --base_model "$BASE_MODEL" \
  --lora_model "$LORA_DIR" \
  --eval_file "$EVAL_FILE" \
  --max_length $MAX_LEN \
  --batch_size $BATCH_SIZE \
  --bf16
```

The script will print:
- Overall 4-class accuracy (E/S/C/I)
- Average relevance score for each true label
- Optionally save per-sample outputs (labels, scores, probabilities) via --save_scores_path.

---


## 📡 Weights & Biases (Optional)

wandb login
export WANDB_PROJECT=esci-qwen3-reranker

Then run:

```bash
python train.py \
  ... \
  --report_to wandb \
  --wandb_run_name qwen3-esci-lora-v1
```

Trainer will automatically push metrics to W&B.

---