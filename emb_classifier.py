# emb_classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# MODEL_NAME = "../llms/Qwen/Qwen3-Embedding-0.6B"
# MODEL_NAME = "../llms/Qwen/Qwen3-0.6B"

def last_token_pool(last_hidden_states, attention_mask):
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

class QwenEmbeddingClassifier(nn.Module):
    def __init__(self, encoder: AutoModel, num_labels: int = 4, dropout: float = 0):
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # # === 1. 冻结 encoder 所有参数 ===
        # for p in self.encoder.parameters():
        #     p.requires_grad = False

        # # === 2. 只训练 q_proj 与 v_proj ===
        # for name, p in self.encoder.named_parameters():
        #     if "q_proj" in name or "v_proj" in name:
        #         p.requires_grad = True

        # # === 3. 分类头参与训练 ===
        # for p in self.classifier.parameters():
        #     p.requires_grad = True

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = last_token_pool(outputs.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # (batch, num_labels)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits}
