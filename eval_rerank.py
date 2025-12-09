# eval.py

import argparse
from typing import List, Dict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score


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
    parser.add_argument("--save_scores_path", type=str, default=None)

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
        # "device_map": "auto",
    }
    if not args.no_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        **model_kwargs,
    )

    if args.lora_model is None or args.lora_model == "":
        model = base_model
    else:
        print(f"Loading LoRA adapter from: {args.lora_model}")
        model = PeftModel.from_pretrained(base_model, args.lora_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"Loading eval data from: {args.eval_file}")
    df_eval = load_esci_parquet(args.eval_file)
    print(f"Eval size: {len(df_eval)}")

    # label 文本 -> token id（这里假设是单 token，大多数情况下是的）
    label_tokens: Dict[str, int] = {
        k: tokenizer.encode(v, add_special_tokens=False)[0]
        for k, v in LABEL_TEXT.items()
    } 
    # {'E': 46385, 'S': 1966, 'C': 874, 'I': 404} # ["exact", "sub", "com", "ir"]

    id2label = {i: k for i, k in enumerate(["E", "S", "C", "I"])}

    batch_size = args.batch_size
    all_preds: List[str] = []              # 预测相关性标签 E/S/C/I
    all_trues: List[str] = []              # 真实相关性标签 E/S/C/I
    all_scores: List[float] = []           # 相关性分数 in [0, 1]
    all_probs: List[List[float]] = []      # 4 个 label 的概率 in [0, 1]
    all_probs_vocab: List[List[float]] = []  # 4 个 label 的概率 in [0, 1]，基于完整 vocab 计算得到
    
    rel_weights = torch.tensor([3.0, 2.0, 1.0, 0.0])

    # for start in range(0, len(df_eval), batch_size):
    for start in tqdm(
        range(0, len(df_eval), batch_size),
        desc="Evaluating",
        ncols=80
    ):
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
        rel_weights_batch = rel_weights.to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]  # (bs, vocab)

        # 取 4 个 label 的 logits
        class_logits = torch.stack(
            [logits[:, label_tokens[k]] for k in ["E", "S", "C", "I"]], dim=-1
        )  # (bs, 4)
        
        vocab_probs = torch.softmax(logits, dim=-1)  # (bs, vocab) 完整 vocab 下的概率分布
        target_ids = torch.tensor(
            [label_tokens[k] for k in ["E", "S", "C", "I"]],
            device=logits.device,
        )
        probs_vocab = vocab_probs[:, target_ids]  # (bs, 4)
        probs = torch.softmax(class_logits, dim=-1)  # (bs, 4) 「只考虑 ESCI 四类」前提下的类别概率
        pred_idx = probs.argmax(dim=-1).cpu().numpy()

        preds = [id2label[i] for i in pred_idx] # 四分类的预测标签

        # print(f"class_logits: {class_logits}")
        # print(f"probs_vocab: {probs_vocab}")
        # print(f"probs: {probs}")
        # print(f"preds: {preds}")
        # print(f"trues: {trues}")

        # if start > 150:
        #     exit()
        
        # score = (3*P(E) + 2*P(S) + 1*P(C) + 0*P(I)) / 3
        rel_scores = (probs * rel_weights_batch).sum(dim=-1) / rel_weights_batch.max()
        rel_scores = rel_scores.cpu().tolist()

        all_preds.extend(preds)
        all_trues.extend(trues)
        all_scores.extend(rel_scores)
        all_probs_vocab.extend(probs_vocab.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())



    all_preds_arr = np.array(all_preds)
    all_trues_arr = np.array(all_trues)
    all_scores_arr = np.array(all_scores)
    all_probs = np.array(all_probs)
    

    acc = (all_preds_arr == all_trues_arr).mean()
    print("====== Evaluation Result ======")
    print(f"Accuracy (4-class E/S/C/I): {acc:.4f}")
    print("===============================")
    
    # ⭐ 简单打印一下不同真实档位下，平均相关性分数
    for lbl in ["E", "S", "C", "I"]:
        mask = all_trues_arr == lbl
        if mask.sum() == 0:
            continue
        avg_score = all_scores_arr[mask].mean()
        print(f"Avg relevance score for {lbl}: {avg_score:.4f}")
    print("===============================")

    # ===== 更多分类指标：precision / recall / F1 / support =====
    label_order = ["E", "S", "C", "I"]

    print("Classification report (per class):")
    cls_report = classification_report(
        all_trues_arr,
        all_preds_arr,
        labels=label_order,
        digits=4,
        zero_division=0,   # 某类完全没预测到时不报错，F1=0
    )
    print(cls_report)

    macro_f1 = f1_score(
        all_trues_arr,
        all_preds_arr,
        labels=label_order,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        all_trues_arr,
        all_preds_arr,
        labels=label_order,
        average="weighted",
        zero_division=0,
    )
    print(f"Macro-F1:    {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}")
    print("===============================")

    # ===== 混淆矩阵：行是真实，列是预测 =====
    cm = confusion_matrix(
        all_trues_arr,
        all_preds_arr,
        labels=label_order,
    )
    print("Confusion Matrix (rows=true, cols=pred):")
    header = "      " + " ".join(f"{lbl:>6}" for lbl in label_order)
    print(header)
    for i, row in enumerate(cm):
        row_str = " ".join(f"{n:6d}" for n in row)
        print(f"{label_order[i]:>3} | {row_str}")
    print("===============================")


    if args.save_scores_path is not None:
        np.savez(
            args.save_scores_path,
            true_labels=all_trues_arr,
            pred_labels=all_preds_arr,
            rel_scores=all_scores_arr,
            class_probs=np.array(all_probs),
            class_probs_vocab=np.array(all_probs_vocab),
        )
        print(f"Saved per-sample scores & probs to {args.save_scores_path}")


if __name__ == "__main__":
    main()