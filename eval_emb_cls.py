import argparse
from pathlib import Path
import os
import random 
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, TaskType
from safetensors.torch import load_file as safe_load
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm


# 复用你项目里的这两个
from esci_dataset import load_esci_parquet, ESCIEmbClassifierDataset
from emb_classifier import QwenEmbeddingClassifier

def parse_args():
    p = argparse.ArgumentParser("Evaluate QwenEmbeddingClassifier (LoRA on q/v)")
    p.add_argument("--base_model", type=str, required=True,
                   help="Qwen3-0.6B")
    p.add_argument("--checkpoint_dir", type=str, required=True,
                   help="包含 model.safetensors 的目录，如 ./emb_esci_cls/checkpoint-6000")
    p.add_argument("--eval_file", type=str, required=True,
                   help="评估 parquet 文件（与训练同格式）")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--per_device_eval_batch_size", type=int, default=256)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--no_flash_attn", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # LoRA 超参必须与训练一致
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    p.add_argument("--eval_ratio", type=float, default=1.0,
                   help="评估集采样比例(0,1]，例如 0.1 表示抽样10%做评估")
    p.add_argument("--seed", type=int, default=42, help="随机种子（影响采样、dataloader打乱等）")
    
    return p.parse_args()

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n_batches = 0.0, 0
    n_samples = 0

    # 若 DataLoader 可取 len()，直接用
    total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
    pbar = tqdm(dataloader, total=total_batches, desc="Evaluating", leave=False)


    # for batch in dataloader:
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = out["logits"]
        loss = out["loss"]

        total_loss += loss.item()
        n_batches += 1

        preds = torch.argmax(logits, dim=-1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

        avg_loss = total_loss / n_batches
        pbar.set_postfix({
            "avg_loss": f"{avg_loss:.4f}",
            "seen": n_samples
        })

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_micro = f1_score(all_labels, all_preds, average="micro")
    report = classification_report(all_labels, all_preds, digits=4)

    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "report": report,
    }

def main():
    args = parse_args()
    set_seed(args.seed)
    
    ckpt_dir = Path(args.checkpoint_dir)
    model_path = ckpt_dir / "model.safetensors"
    assert model_path.exists(), f"未找到 {model_path}"

    # 1) tokenizer
    print(f"Loading tokenizer from: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2) encoder + LoRA（与训练一致）
    print(f"Loading base model from: {args.base_model}")
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model_kwargs = {"dtype": dtype}
    if not args.no_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    encoder = AutoModel.from_pretrained(args.base_model, **model_kwargs)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.FEATURE_EXTRACTION,
        bias="none",
    )
    encoder = get_peft_model(encoder, lora_cfg)

    # 3) 包装分类器并加载 state_dict（safetensors）
    print("Wrapping into QwenEmbeddingClassifier...")
    model = QwenEmbeddingClassifier(encoder=encoder, num_labels=4)  # E/S/C/I
    # safetensors -> strict=True 可保证结构一致
    state = safe_load(str(model_path), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print("[Warn] missing keys:", missing)
    if unexpected:
        print("[Warn] unexpected keys:", unexpected)

    device = torch.device(args.device)
    model.to(device)

    # 4) 构造评估集
    print(f"Loading eval data from: {args.eval_file}")
    df_eval = load_esci_parquet(args.eval_file)

    eval_ratio = args.eval_ratio  # NEW
    if not (0.0 < eval_ratio <= 1.0):
        raise ValueError(f"--eval_ratio 必须在 (0,1]，当前为 {eval_ratio}")
    if eval_ratio < 1.0:
        # 按比例抽样，固定随机种子
        orig_len = len(df_eval)
        df_eval = df_eval.sample(frac=eval_ratio, random_state=args.seed).reset_index(drop=True)
        print(f"[Eval Sampling] Using {len(df_eval)} / {orig_len} rows (ratio={eval_ratio})")
    else:
        print(f"[Eval Sampling] Using full eval set: {len(df_eval)} rows")


    eval_ds = ESCIEmbClassifierDataset(df_eval, tokenizer=tokenizer, max_length=args.max_length)
    eval_loader = DataLoader(eval_ds, batch_size=args.per_device_eval_batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # 5) Eval
    print("Starting evaluation...")
    metrics = evaluate(model, eval_loader, device)
    print("\n==== Evaluation ====")
    print(f"loss:      {metrics['loss']:.6f}")
    print(f"accuracy:  {metrics['accuracy']:.4f}")
    print(f"f1_macro:  {metrics['f1_macro']:.4f}")
    print(f"f1_micro:  {metrics['f1_micro']:.4f}")
    print("\nClassification report:\n", metrics["report"])

if __name__ == "__main__":
    main()
