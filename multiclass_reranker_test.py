# Requires transformers>=4.51.0
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

from esci_dataset import (
    load_esci_parquet,
    LABEL_TEXT,
    INSTRUCT,
)


def build_prefix_suffix() -> (str, str):
    from esci_dataset import SYSTEM_PROMPT
    prefix = (
        "<|im_start|>system\n"
        + SYSTEM_PROMPT +
        "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return prefix, suffix

def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
    return output

def process_inputs(pairs):
    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    return inputs

@torch.no_grad()
def compute_logits(inputs, **kwargs):
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()

    
    logits = model(**inputs).logits[:, -1, :] # [B, vocab_size] 原始 logits
    log_probs = F.log_softmax(logits, dim=-1)   # [B, vocab_size]

    true_logp_full = log_probs[:, token_true_id]    # [B]
    false_logp_full = log_probs[:, token_false_id]  # [B]

    true_p_full = true_logp_full.exp()   # P(true | vocab)
    false_p_full = false_logp_full.exp() # P(false | vocab)

    binary_logits = torch.stack([false_logp_full, true_logp_full], dim=-1)  # [B, 2]
    binary_log_probs = F.log_softmax(binary_logits, dim=-1)
    p_false_bin = binary_log_probs[:, 0].exp()  # P(false | {false,true})
    p_true_bin = binary_log_probs[:, 1].exp()   # P(true  | {false,true})

    # return scores, true_vector, false_vector
    return {
        "p_true_full": true_p_full,           # tensor，P(true | vocab)
        "p_false_full": false_p_full,         # tensor，P(false | vocab)
        "p_true_binary": p_true_bin,          # tensor，P(true | {false,true})
        "p_false_binary": p_false_bin,
        "true_logits": logits[:, token_true_id],
        "false_logits": logits[:, token_false_id],
    }

tokenizer = AutoTokenizer.from_pretrained("./Qwen/Qwen3-Reranker-0.6B", padding_side='left')
model = AutoModelForCausalLM.from_pretrained("./Qwen/Qwen3-Reranker-0.6B").eval()
# We recommend enabling flash_attention_2 for better acceleration and memory saving.
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B", torch_dtype=torch.float16, attn_implementation="flash_attention_2").cuda().eval()
token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")
max_length = 1024

prefix, suffix = build_prefix_suffix()
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

task = "Given an e-commerce search query and a candidate product title, \
        请你判断该商品标题是否符合用户的搜索意图，需要完美符合才能回答 1，否则回答 0，比如搜索无糖就不能出现有糖，你只能回答1和0"

queries = [
    "无糖可乐",
    "无糖可乐",
]

documents = [
    "百事可乐标准版",
    "可口可乐，无糖",
]

pairs = [format_instruction(task, query, doc) for query, doc in zip(queries, documents)]

# Tokenize the input texts
inputs = process_inputs(pairs)
result = compute_logits(inputs)

print("scores: ", result['p_true_binary'].tolist())
print("true prob: ", result['p_true_full'].tolist())
print("false prob: ", result['p_false_full'].tolist())
