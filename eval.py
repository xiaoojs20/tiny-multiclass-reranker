# eval.py

import argparse
from typing import List, Dict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from esci_dataset import (
    load_esci_parquet,
    LABEL_TEXT,
    INSTRUCT,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen3 ESCI multi-class reranker (LoRA).")

    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--lora_model", type=str, required=True)
    parser.add_argument("--eval_file", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no_flash_attn", action="store_true")

    return parser.parse_args()


def build_prefix_suffix() -> (str, str):
    from esci_dataset import SYSTEM_PROMPT
    prefix = (
        "<|im_start|>system\n"
        + SYSTEM_PROMPT +
        "<|im_end|>\n<|im_start|>user\n"
    )
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return prefix, suffix


def build_inference_inputs(
    tokenizer: AutoTokenizer,
    queries: List[str],
    docs: List[str],
    max_length: int,
) -> Dict[str, torch.Tensor]:
    from esci_dataset import format_instruction

    prefix, suffix = build_prefix_suffix()
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)

    bodies: List[List[int]] = []
    for q, d in zip(queries, docs):
        text = format_instruction(q, d, INSTRUCT)
        body_ids = tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length - len(prefix_ids) - len(suffix_ids) - 16,
        )
        bodies.append(body_ids)

    input_ids = []
    for body in bodies:
        ids = prefix_ids + body + suffix_ids
        input_ids.append(ids)

    # padding 到 batch 内最长
    model_inputs = tokenizer.pad(
        {"input_ids": input_ids},
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return model_inputs


def main():
    args = parse_args()
    print(f"Loading tokenizer from: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model from: {args.base_model}")
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": "auto",
    }
    if not args.no_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        **model_kwargs,
    )

    print(f"Loading LoRA adapter from: {args.lora_model}")
    model = PeftModel.from_pretrained(base_model, args.lora_model)
    model.eval()

    print(f"Loading eval data from: {args.eval_file}")
    df_eval = load_esci_parquet(args.eval_file)
    print(f"Eval size: {len(df_eval)}")

    # label 文本 -> token id（这里假设是单 token，大多数情况下是的）
    label_tokens: Dict[str, int] = {
        k: tokenizer.encode(v, add_special_tokens=False)[0]
        for k, v in LABEL_TEXT.items()
    }
    id2label = {i: k for i, k in enumerate(["E", "S", "C", "I"])}

    batch_size = args.batch_size
    all_preds: List[str] = []
    all_trues: List[str] = []

    for start in range(0, len(df_eval), batch_size):
        end = min(start + batch_size, len(df_eval))
        batch = df_eval.iloc[start:end]

        queries = batch["query"].tolist()
        docs = batch["item_text"].tolist()
        trues = batch["esci_label"].tolist()

        inputs = build_inference_inputs(
            tokenizer,
            queries,
            docs,
            max_length=args.max_length,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]  # (bs, vocab)

        # 取 4 个 label 的 logits
        class_logits = torch.stack(
            [logits[:, label_tokens[k]] for k in ["E", "S", "C", "I"]], dim=-1
        )  # (bs, 4)
        probs = torch.softmax(class_logits, dim=-1)  # (bs, 4)
        pred_idx = probs.argmax(dim=-1).cpu().numpy()

        preds = [id2label[i] for i in pred_idx]

        all_preds.extend(preds)
        all_trues.extend(trues)

    all_preds_arr = np.array(all_preds)
    all_trues_arr = np.array(all_trues)

    acc = (all_preds_arr == all_trues_arr).mean()
    print("====== Evaluation Result ======")
    print(f"Accuracy (4-class E/S/C/I): {acc:.4f}")
    print("===============================")


if __name__ == "__main__":
    main()
