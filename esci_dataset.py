# esci_dataset.py

from typing import List, Dict
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

# https://github.com/amazon-science/esci-data

# 可选的商品文本列（按需要自己调）
TEXT_COLS: List[str] = [
    "product_title",
    # "product_description",
    # "product_bullet_point",
    # "product_brand",
    # "product_color",
]

# ESCI 四档 label -> 文本答案
LABEL_TEXT: Dict[str, str] = {
    "E": "exact",
    "S": "substitute",
    "C": "complement",
    "I": "irrelevant",
}

# system / instruct，用于构造和 Qwen3-reranker 风格一致的 prompt
SYSTEM_PROMPT: str = (
    "You are an expert e-commerce search relevance judge. "
    "Given a user search query and a candidate product (Document), "
    "classify the relevance into one of the following categories:\n"
    "- exact: the product exactly matches the query intent;\n"
    "- substitute: the product can be used instead of what is requested;\n"
    "- complement: the product is a reasonable accessory or complement;\n"
    "- irrelevant: the product is not relevant to the query."
)

INSTRUCT: str = (
    "Given a shopping query and a candidate product, "
    "classify their relevance into one of: exact, substitute, complement, or irrelevant."
)


def build_item_text(row: pd.Series) -> str:
    """把商品的 title/description/bullet 等信息串成一段文本。"""
    parts: List[str] = []
    for col in TEXT_COLS:
        if col in row:
            val = row[col]
            if isinstance(val, str) and val.strip():
                pretty = col.replace("_", " ")
                parts.append(f"{pretty}: {val}")
    if not parts and "product_title" in row:
        val = row["product_title"]
        if isinstance(val, str) and val.strip():
            parts.append(val)
    return "\n".join(parts)



def load_esci_parquet(path: str) -> pd.DataFrame:
    """
    读取 parquet，并确保有三列：
    - query
    - esci_label（E/S/C/I）
    - item_text（若不存在则自动构造）
    """
    df = pd.read_parquet(path)
    if "query" not in df.columns:
        raise ValueError("Expected column 'query' in parquet file.")

    if "esci_label" not in df.columns:
        raise ValueError("Expected column 'esci_label' in parquet file.")
    df = df.copy()
    df["esci_label"] = (
        df["esci_label"]
        .astype(str)
        .str.upper()
        .str[0]
    )

    if "item_text" not in df.columns:
        df["item_text"] = df.apply(build_item_text, axis=1)

    return df


def format_instruction(query: str, doc: str, instruction: str = INSTRUCT) -> str:
    """构造 user 段落文本：<Instruct> / <Query> / <Document>。"""
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


class ESCIMultiClassRerankDataset(Dataset):
    """
    用于多分类 ESCI 训练的 Dataset：
    - input_ids: prefix + body + suffix + label + eos
    - labels: 只有 label 对应 token 位置是真实 id，其余为 -100（不算 loss）
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2048,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # 和 Qwen3-reranker 一致的对话包装
        self.prefix = (
            "<|im_start|>system\n"
            + SYSTEM_PROMPT +
            "<|im_end|>\n<|im_start|>user\n"
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        self.prefix_tokens = self.tokenizer.encode(
            self.prefix, add_special_tokens=False
        )
        self.suffix_tokens = self.tokenizer.encode(
            self.suffix, add_special_tokens=False
        )

        # label 文本 -> token 序列（可能是多 token，一并算 loss）
        self.label_token_ids: Dict[str, List[int]] = {
            k: self.tokenizer.encode(v, add_special_tokens=False)
            for k, v in LABEL_TEXT.items()
        }
        self.max_label_len = max(len(v) for v in self.label_token_ids.values())

        self.eos_ids = self.tokenizer.encode(
            self.tokenizer.eos_token, add_special_tokens=False
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        query = row["query"]
        doc = row["item_text"]
        label_key = row["esci_label"]

        if label_key not in LABEL_TEXT:
            raise ValueError(f"Unexpected esci_label: {label_key}")

        body_text = format_instruction(query=query, doc=doc)
        # 预留 prefix/suffix/label/eos 的长度，保证截断后还能塞得下
        body_ids = self.tokenizer.encode(
            body_text,
            add_special_tokens=False,
            truncation=True,
            max_length=(
                self.max_length
                - len(self.prefix_tokens)
                - len(self.suffix_tokens)
                - self.max_label_len
                - len(self.eos_ids)
                - 4
            ),
        )

        label_ids = self.label_token_ids[label_key]

        input_ids = (
            self.prefix_tokens
            + body_ids
            + self.suffix_tokens
            + label_ids
            + self.eos_ids
        )
        attention_mask = [1] * len(input_ids)
        labels = [-100] * len(input_ids)

        # 只在 label_ids 对应区间放真实 label
        label_start = len(input_ids) - len(label_ids) - len(self.eos_ids)
        for i in range(len(label_ids)):
            labels[label_start + i] = label_ids[i]

        # 左 padding（和 Qwen3-reranker 一致）
        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length:]
            attention_mask = attention_mask[-self.max_length:]
            labels = labels[-self.max_length:]
        else:
            pad_len = self.max_length - len(input_ids)
            pad_id = self.tokenizer.pad_token_id
            input_ids = [pad_id] * pad_len + input_ids
            attention_mask = [0] * pad_len + attention_mask
            labels = [-100] * pad_len + labels

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
