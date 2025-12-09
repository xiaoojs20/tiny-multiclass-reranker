# tiny-reranker：基于 Qwen3-0.6B 的 ESCI 四档相关性重排模型

本仓库在 **Qwen3-0.6B** 上使用 **LoRA**，基于 **Amazon ESCI** 数据集进行微调，构建一个面向电商搜索的 **四分类相关性重排模型**。

- 离散相关性标签：**E / S / C / I**
  - **E**：exact，完全匹配  
  - **S**：substitute，替代品  
  - **C**：complement，互补品  
  - **I**：irrelevant，不相关
- 同时输出一个 **[0, 1] 的连续相关性分数**，方便用于排序、AUC 等评估。

训练风格参考 **Qwen3-Reranker**：使用 chat 格式 prompt，并在最后一个 token 做分类预测。

---

## 🔍 这个项目在做什么？

- 把 **Qwen3-0.6B** 微调成 **理解 ESCI 四档的 reranker**；
- 使用 **LoRA** 进行参数高效微调，显存需求较小；
- 模型在最后一个 token 处预测四个 label 中的一个：
  - `"exact"`, `"substitute"`, `"complement"`, `"irrelevant"`；
- 推理阶段计算：
  - 四分类概率：`P(E), P(S), P(C), P(I)`；
  - 一个连续相关性分数：
    $
    \text{score} = \frac{3P(E) + 2P(S) + 1P(C) + 0P(I)}{3}
    $
  - 这个分数可以直接作为排序信号。

---

## 📂 项目结构

```text
tiny-reranker/
├── train.py          # LoRA 微调脚本（基于 ESCI 四分类）
├── eval.py           # 评估脚本（E/S/C/I + 连续相关性分数）
├── esci_dataset.py   # ESCI 数据集加载 & prompt 构造
├── scripts/
│   ├── run_train.sh  # 训练脚本示例
│   └── run_eval.sh   # 评估脚本示例
├── README.md
└── README_zh.md
```

关键文件：
- esci_dataset.py
    - load_esci_parquet(path)：读取预处理好的 parquet 文件；
	- ESCIMultiClassRerankDataset：构造 chat 风格输入与标签；
	- LABEL_TEXT = {"E": "exact", "S": "substitute", "C": "complement", "I": "irrelevant"}；
	- SYSTEM_PROMPT、INSTRUCT、format_instruction(...)：对齐 Qwen3-Reranker 的提示词格式。
- train.py
	- 加载 Qwen3-0.6B；
	- 在注意力投影层（如 q_proj, v_proj）上注入 LoRA；
	- 通过 --train_file 读取 ESCI parquet，并用 --eval_ratio 在内部划分 train / eval；
	- 使用 Hugging Face Trainer 进行训练，可选接入 wandb 日志。
- eval.py
	- 加载 base model + LoRA adapter；
	- 用 query + item_text 构造 prompt；
	- 在最后一个 token 的 logits 上：
	- 提取四个 label 对应 token 的 logits；
	- 对这 4 个 logits 做 softmax，得到 ESCI 四档概率；
	- 输出预测 label + 连续相关性分数。

---

## 🧱 数据准备

数据来源：Amazon ESCI 数据集（英文）。

预处理后，建议 parquet 文件包含以下字段：
- query：搜索 query；
- item_text：商品侧文本（如标题 + 品牌 + 属性 等）；
- esci_label：E / S / C / I 四档之一。

示例代码：

```python
from pathlib import Path
import pandas as pd

OUT_DIR = Path("../datasets/esci-data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df_train = ...  # ESCI 训练集
df_test = ...   # ESCI 测试集

df_train.to_parquet(OUT_DIR / "esci_multiclass_train.parquet", index=False)
df_test.to_parquet(OUT_DIR / "esci_multiclass_test.parquet", index=False)
```

本项目中默认使用：
- ../datasets/esci-data/esci_multiclass_train.parquet
- ../datasets/esci-data/esci_multiclass_test.parquet

---

## ⚙️ 环境配置

conda create -n tiny-reranker python=3.10 -y
conda activate tiny-reranker

安装依赖（按需修改 CUDA 版本）：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft accelerate datasets pandas pyarrow scikit-learn tqdm
pip install wandb  # 可选：如需接入 Weights & Biases

# 如果环境支持 flash-attn：

pip install flash-attn --no-build-isolation

# 如不支持，在运行脚本时加 --no_flash_attn 即可。
```

--- 

## 🚀 训练（Training）

单卡训练示例

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
EVAL_RATIO=0.05   # 例如：5% 样本作为验证集

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

说明：
- --eval_ratio：在 train_file 里按比例切出验证集；
- --save_total_limit：最多保留多少个 checkpoint，旧的自动删除；
- --bf16：使用 bfloat16，如 GPU 不支持可去掉此参数。

多卡训练（torchrun + DDP）

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

⸻

📈 评估（Evaluation）

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

eval.py 会：
	•	构造 prompt，调用模型前向；
	•	提取最后一个 token 对应的 vocab logits；
	•	取出 "exact" / "substitute" / "complement" / "irrelevant" 四个 token 的 logits，做 softmax 得到 ESCI 四档概率；
	•	输出：
	•	四分类准确率；
	•	每个真实档位下的平均相关性分数；
	•	若开启相应代码，还可输出分类报告 / 混淆矩阵；
	•	通过 --save_scores_path 保存每条样本的 label / score / prob。

---

## 📡 Weights & Biases

```bash
wandb login
export WANDB_PROJECT=esci-qwen3-reranker
```

训练时：

```bash
python train.py \
  ... \
  --report_to wandb \
  --wandb_run_name qwen3-esci-lora-v1
```

Trainer 会自动把 loss / eval 指标同步到 W&B。