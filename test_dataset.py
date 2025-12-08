"""
python test_dataset.py \
  --base_model ../llms/Qwen/Qwen3-0.6B \
  --data_file ../datasets/esci-data/esci_multiclass_train.parquet \
  --max_length 512 \
  --index 0 \
  --num_samples 2
"""
import argparse
from typing import List

import torch
from transformers import AutoTokenizer

from esci_dataset import (
    load_esci_parquet,
    ESCIMultiClassRerankDataset,
    LABEL_TEXT,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect ESCI dataset sample: prompt+response and input masks."
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model name or path (e.g., Qwen/Qwen3-0.6B-Base).",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to esci_multiclass_train.parquet or test parquet.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Max sequence length used in dataset.",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Which sample index to inspect.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="How many consecutive samples to print starting from index.",
    )
    return parser.parse_args()


def summarize_tensor(t: torch.Tensor, name: str, max_show: int = 32):
    """打印 tensor 的长度和前后若干个值，避免一大串看不清。"""
    t_list = t.tolist()
    print(f"{name}: len={len(t_list)}")
    if len(t_list) <= 2 * max_show:
        print(f"{name} values: {t_list}")
    else:
        head = t_list[:max_show]
        tail = t_list[-max_show:]
        print(f"{name} head[{max_show}]: {head}")
        print(f"{name} tail[{max_show}]: {tail}")


def inspect_sample(
    idx: int,
    df,
    dataset: ESCIMultiClassRerankDataset,
    tokenizer: AutoTokenizer,
):
    print("=" * 80)
    print(f"Sample index: {idx}")
    row = df.iloc[idx]

    # 1. 打印原始信息
    print("\n[RAW ROW]")
    print(f"query      : {row['query']}")
    print(f"esci_label : {row['esci_label']} (mapped to text: {LABEL_TEXT.get(row['esci_label'], 'UNKNOWN')})")
    print(f"item_text  : {row['item_text'][:300]}{'...' if len(row['item_text']) > 300 else ''}")

    # 2. 取 dataset 中构造好的 input_ids / attention_mask / labels
    sample = dataset[idx]
    input_ids: torch.Tensor = sample["input_ids"]
    attention_mask: torch.Tensor = sample["attention_mask"]
    labels: torch.Tensor = sample["labels"]

    print("\n[SHAPES & BASIC INFO]")
    print(f"input_ids shape      : {tuple(input_ids.shape)}")
    print(f"attention_mask shape : {tuple(attention_mask.shape)}")
    print(f"labels shape         : {tuple(labels.shape)}")
    print(f"non-pad tokens (attention_mask sum): {int(attention_mask.sum().item())}")
    print(f"label tokens count (labels != -100): {int((labels != -100).sum().item())}")

    # 3. 解码非 pad 部分的 input_ids -> 完整 prompt+response 文本
    print("\n[DECODED TEXT]")
    # 找到第一个非 pad 的位置（attention_mask==1 的第一个 index）
    nonpad_indices = (attention_mask != 0).nonzero(as_tuple=False)
    if len(nonpad_indices) > 0:
        start_idx = nonpad_indices[0].item()
    else:
        start_idx = 0
    valid_input_ids = input_ids[start_idx:]
    decoded = tokenizer.decode(
        valid_input_ids,
        skip_special_tokens=False,  # 保留 <|im_start|> 等，方便调试
    )
    print(decoded)

    # 4. 打印 label 区间对应的 token（真正算 loss 的部分）
    print("\n[LABEL POSITIONS (where labels != -100)]")
    label_positions: torch.Tensor = (labels != -100).nonzero(as_tuple=False).view(-1)
    if len(label_positions) == 0:
        print("No label positions found (labels are all -100).")
    else:
        print(f"Label positions indices: {label_positions.tolist()}")
        label_token_ids: List[int] = input_ids[label_positions].tolist()
        label_tokens = [tokenizer.decode([tid], skip_special_tokens=False) for tid in label_token_ids]
        print(f"Label token ids   : {label_token_ids}")
        print(f"Label token texts : {label_tokens}")
        print("-> These tokens correspond to the textual label, e.g. 'exact' / 'substitute' / ...")

    # 5. 打印部分 input_ids / attention_mask / labels 的数值，方便核对
    print("\n[NUMERIC SUMMARY]")
    summarize_tensor(input_ids, "input_ids")
    summarize_tensor(attention_mask, "attention_mask")
    summarize_tensor(labels, "labels")


def main():
    args = parse_args()

    print(f"Loading tokenizer from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading data from {args.data_file}")
    df = load_esci_parquet(args.data_file)
    print(f"Data size: {len(df)}")

    dataset = ESCIMultiClassRerankDataset(
        df=df,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    start = args.index
    end = min(start + args.num_samples, len(dataset))
    for idx in range(start, end):
        inspect_sample(idx, df, dataset, tokenizer)


if __name__ == "__main__":
    main()
