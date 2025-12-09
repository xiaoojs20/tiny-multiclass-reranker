import torch
import torch.nn.functional as F
from transformers import AutoConfig, Qwen3Model, AutoTokenizer
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from types import MethodType
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

MODEL_NAME = "../llms/Qwen/Qwen3-0.6B"


def bi_qwen3_attention_forward(
    self: Qwen3Attention,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor = None,
    position_ids: torch.LongTensor = None,
    position_embeddings: torch.Tensor | None = None,
    past_key_value: tuple[torch.Tensor] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs, # 捕获其他可能的参数，例如 is_causal
) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
    """
    一个用于实现双向注意力（Bi-directional Attention）的 Qwen3Attention.forward 替代品。
    核心目的：禁用内部的因果掩码逻辑。
    """
    
    # --- 核心修改部分 ---
    # 1. 确保不会使用 KV 缓存，从而避免触发生成模式下的因果逻辑
    past_key_value = None 
    use_cache = False 

    # 2. 将 attention_mask 强制设置为只包含 padding mask
    # 这一步依赖于您已经在 Model 级别对 _update_causal_mask 的修改，
    # 确保传入这里的 attention_mask 已经是 4D 且只包含 padding 信息（上三角非 -inf）。
    # 如果您没有修改 model._update_causal_mask，则这里传入的 mask 仍然会是下三角。
    # 假设：此处传入的 attention_mask (4D) 是非因果的。

    # 3. 确保底层调用的注意力函数（如 torch.nn.functional.scaled_dot_product_attention）
    # 不会被 is_causal 标志影响。
    # Qwen3 的 forward 内部可能会调用一个接受 is_causal 的函数。
    # 我们将参数 is_causal 设置为 False 传入原始方法。
    
    
    # --- 调用原始逻辑 ---
    # 为了避免复杂的重写，我们直接调用原始类的方法，并强制传入关键参数。
    # 注意：这里需要保存原始方法（见下一步）。
    
    # 强制 is_causal=False 传入下一层（SDPA/eager attention）
    # 对于 eager 模式，通常是 self._attn(query, key, value, attention_mask=attention_mask, is_causal=False)
    
    # 调用原始方法时，传入修改后的参数
    # 使用 self.original_forward 是最简洁的方式。
    
    # 由于原始 Qwen3Attention.forward 并不直接接受 is_causal 作为参数，
    # 且模型内部会根据 model.config.is_causal 或其他标志来控制，
    # 最彻底的办法是定位并替换 **实际应用因果掩码的代码行**。
    
    # 如果要避免重写整个 forward，但您使用的是 **eager** 模式，
    # 核心在调用 F.scaled_dot_product_attention 时强制 `is_causal=False`。
    
    # ⚠️ 考虑到 Qwen3 模型的复杂性，如果您使用的是 `eager` 模式：
    # 直接修改 Qwen3Attention._init_attn_params 是一个更干净的选择，
    # 确保内部的 `_use_flash_attention_2` 和 `_use_sdpa` 标志被正确控制。
    
    # **最终方案：** 我们将依赖您已有的 `is_causal = False` 和 `_update_causal_mask_bi` 的修改，
    # 并强制把这个新方法应用到每一层，确保所有参数都是非因果的。

    kwargs["is_causal"] = False

    # 尝试调用原始方法
    attn_output, attn_weights, past_key_value = self.original_forward(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        past_key_value=past_key_value, 
        output_attentions=output_attentions,
        use_cache=use_cache,
        **kwargs,
    )

    return attn_output, attn_weights, past_key_value
    return self.original_forward(
        hidden_states=hidden_states,
        attention_mask=attention_mask, # 传入只包含 padding 的 mask
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        past_key_value=past_key_value, # 强制为 None
        output_attentions=output_attentions,
        use_cache=use_cache, # 强制为 False
    )


def build_bi_qwen3_encoder(model_name: str = MODEL_NAME):
    config = AutoConfig.from_pretrained(model_name)
    # 1. 关掉因果相关配置
    config.is_causal = False
    config.use_cache = False
    if hasattr(config, "sliding_window"):
        config.sliding_window = None
    if hasattr(config, "max_window_layers"):
        config.max_window_layers = config.num_hidden_layers

    # 2. 希望拿到 attentions，用 eager 实现
    if hasattr(config, "attn_implementation"):
        config.attn_implementation = "eager"

    model = Qwen3Model.from_pretrained(
        model_name, 
        config=config,
        attn_implementation="eager",
        )
    
    # 3. 关键：把 _update_causal_mask 改成“只做 padding mask，不做因果 mask”
    def _update_causal_mask_bi(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.LongTensor,
        past_key_values_length: int,
    ):
        """
        attention_mask: [B, L]，1=保留，0=padding
        返回：4D mask [B, 1, L_q, L_k]，只屏蔽 padding，不做下三角因果
        """
        if attention_mask is None:
            return None

        bsz, q_len = input_tensor.shape[:2]
        kv_len = q_len + past_key_values_length

        # 这里用 is_causal=False，只保留 padding 信息
        converter = AttentionMaskConverter(
            is_causal=False,
            sliding_window=self.config.sliding_window,
        )
        full_mask = converter.to_4d(
            attention_mask,
            batch_size=bsz,
            query_length=q_len,
            key_value_length=kv_len,
            dtype=input_tensor.dtype,
        )
        return full_mask

    # 绑定到实例上（monkey-patch）
    model._update_causal_mask = MethodType(_update_causal_mask_bi, model)

    # 4. 把每一层的 self_attn.is_causal 改掉
    for layer in model.layers:
        if hasattr(layer, "self_attn"):
            layer.self_attn.is_causal = False
            # 步骤 a: 保存原始方法
            layer.self_attn.original_forward = layer.self_attn.forward
            # 步骤 b: 绑定新的方法
            layer.self_attn.forward = MethodType(bi_qwen3_attention_forward, layer.self_attn)
            # print("Set layer.self_attn.is_causal = False")

            if hasattr(layer.self_attn, "_init_attn_params"):
                layer.self_attn._init_attn_params(is_causal=False)
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer



@torch.no_grad()
def encode_sentences(model, tokenizer, sentences, device="cuda"):
    model.to(device)
    model.eval()

    inputs = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    outputs = model(**inputs)
    hidden = outputs.last_hidden_state   # [B, L, H]
    mask = inputs["attention_mask"].unsqueeze(-1)  # [B, L, 1]

    # mean pooling over非 padding token
    summed = (hidden * mask).sum(dim=1)           # [B, H]
    counts = mask.sum(dim=1).clamp(min=1)         # 防止除 0
    embeddings = summed / counts                  # [B, H]

    # 做一下 L2 normalize，更像 SBERT / gte 的风格
    embeddings = F.normalize(embeddings, p=2, dim=-1)
    return embeddings.cpu()

def get_attention_matrix(model, tokenizer, text, layer=0, head=0, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model(**inputs, output_attentions=True)

    # out.attentions 是一个 tuple，长度 = num_layers
    # 每个元素形状 [batch, num_heads, L, L]
    att = out.attentions[layer][0, head]  # [L, L]

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return att.cpu(), tokens

def plot_attention_heatmap(att, tokens, save_path="attn_layer0_head0.png"):
    import matplotlib.pyplot as plt

    L = att.shape[0]
    fig, ax = plt.subplots(figsize=(max(4, L * 0.6), max(4, L * 0.6)))

    im = ax.imshow(att)

    ax.set_xticks(range(L))
    ax.set_yticks(range(L))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)

    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    ax.set_title("Attention heatmap")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"saved attention heatmap to {save_path}")


def main():
    model, tokenizer = build_bi_qwen3_encoder()

    print("config.is_causal:", model.config.is_causal)
    # 看一层 attention 的 is_causal
    first_layer = model.layers[0]
    print("layer0.self_attn.is_causal:", first_layer.self_attn.is_causal)

    sents = [
        "我喜欢吃北京烤鸭。",
        "我很爱北京的美食，尤其是烤鸭。",
        "今天的天气很好，适合出门散步。",
    ]

    emb = encode_sentences(model, tokenizer, sents, device=model.device)
    print("embedding shape:", emb.shape)

    # 计算余弦相似度
    sim = emb @ emb.T   # [B, B]
    print("cosine similarity matrix:")
    print(sim)


    model.config.output_attentions = True
    with torch.no_grad():
        out = model(**tokenizer("今天 天气 很好", return_tensors="pt").to(model.device))
    att = out.attentions[0][0, 0]  # [L, L]
    print("first layer head0 attention:\n", att)


    text = "today is a good day to go for a walk. I love sunny days!"
    att, tokens = get_attention_matrix(model, tokenizer, text, layer=0, head=0)
    print("attention shape:", att.shape)
    print("tokens:", tokens)

    # 看看“是否关注未来 token”
    future_max = float(att.triu(1).max())
    past_max = float(att.tril(-1).max())
    print("max attention to future tokens:", future_max)
    print("max attention to past tokens  :", past_max)

    plot_attention_heatmap(att, tokens, save_path="bi_qwen3_layer0_head0.png")


if __name__ == "__main__":
    main()

