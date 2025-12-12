# emb_classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# MODEL_NAME = "../llms/Qwen/Qwen3-Embedding-0.6B"
# MODEL_NAME = "../llms/Qwen/Qwen3-0.6B"

"""
用到的几个模型
Qwen3-Embedding-0.6B: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
multilingual-e5-large: https://huggingface.co/intfloat/multilingual-e5-large
"""


def last_token_pool(last_hidden_states, attention_mask):
    """ e.g. Qwen3-Emedding
    last token 池化：取 [batch, seq_len, hidden] 中 attention mask 最后一个为 True 的 token
    Args:
        last_hidden_states: [batch, seq_len, hidden]
        attention_mask: [batch, seq_len]
    Returns:
        embeddings: [batch, hidden]
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths
        ]
    
def average_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    """ e.g. multilingual-e5
    平均池化：对 padding mask 之外的 token 取平均
    Args:
        last_hidden_states: [batch, seq_len, hidden]
        attention_mask: [batch, seq_len]
    Returns:
        embeddings: [batch, hidden]
    """
    masked_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return masked_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]



class QwenEmbeddingClassifier(nn.Module):
    def __init__(self, encoder: AutoModel, num_labels: int = 4, dropout: float = 0, normalize_emb: bool = True):
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        self.normalize_emb = normalize_emb
        self.dropout = nn.Dropout(dropout)
        # 关键：拿 encoder 的 dtype
        encoder_dtype = next(encoder.parameters()).dtype
        self.classifier = nn.Linear(hidden_size, num_labels, dtype=encoder_dtype)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = last_token_pool(outputs.last_hidden_state, attention_mask)

        if self.normalize_emb:
            pooled_emb = F.normalize(pooled, p=2, dim=1)

        pooled_emb = self.dropout(pooled_emb)
        logits = self.classifier(pooled_emb)  # (batch, num_labels)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}


class E5EmbeddingClassifier(nn.Module):
    """
    用于分类任务的 multilingual-e5 模型封装：
    encoder -> 平均池化 -> 归一化 -> 分类头
    """
    def __init__(self, encoder: nn.Module, num_labels: int = 4, normalize_emb: bool = True):
        super().__init__()
        self.encoder = encoder
        self.num_labels = num_labels
        self.normalize_emb = normalize_emb

        hidden_size = getattr(encoder.config, "hidden_size", 1024)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Step 1: 编码
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [B, L, H]
        # Step 2: 平均池化得到句向量
        pooled_emb = average_pool(last_hidden, attention_mask)  # [B, H]
        # Step 3: 归一化
        if self.normalize_emb:
            pooled_emb = F.normalize(pooled_emb, p=2, dim=1)
        # Step 4: 分类
        logits = self.classifier(pooled_emb)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
            "embeddings": pooled_emb,
        }

