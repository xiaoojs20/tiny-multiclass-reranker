import argparse
from pathlib import Path

import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

from esci_dataset import (
    load_esci_parquet,
    ESCIEmbClassifierDataset,
)
from emb_classifier import QwenEmbeddingClassifier


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-Embedding-0.6B ESCI classifier."
    )
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory of fine-tuned classifier (where you saved model/tokenizer).")
    parser.add_argument("--eval_file", type=str, required=True,
                        help="Parquet file used for evaluation.")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--output_metrics_file", type=str, default=None,
                        help="If set, save metrics as a JSON file.")
    return parser.parse_args()


def compute_macro_f1(preds: np.ndarray, labels: np.ndarray, num_labels: int) -> float:
    f1_list = []
    for i in range(num_labels):
        tp = np.sum((preds == i) & (labels == i))
        fp = np.sum((preds == i) & (labels != i))
        fn = np.sum((preds != i) & (labels == i))

        if tp == 0 and fp == 0 and fn == 0:
            # 该类在预测和标签中都不存在，跳过
            continue

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1_list.append(f1)

    if not f1_list:
        return 0.0
    return float(np.mean(f1_list))


def build_compute_metrics(num_labels: int = 4):
    def _compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        accuracy = (preds == labels).mean().item()
        macro_f1 = compute_macro_f1(preds, labels, num_labels)

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        }

    return _compute_metrics


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)

    # ===== Load tokenizer & model =====
    print(f"Loading tokenizer from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    print(f"Loading classifier model from: {model_dir} (dtype={dtype})")
    # QwenEmbeddingClassifier 应该继承自 PreTrainedModel，支持 from_pretrained
    model = QwenEmbeddingClassifier.from_pretrained(
        model_dir,
        torch_dtype=dtype,
    )

    # ===== Load eval data =====
    print(f"Loading eval data from: {args.eval_file}")
    df_eval = load_esci_parquet(args.eval_file)
    print(f"Eval size: {len(df_eval)}")

    eval_dataset = ESCIEmbClassifierDataset(
        df_eval,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    # ===== Build Trainer for evaluation only =====
    training_args = TrainingArguments(
        output_dir=str(model_dir / "eval_logs"),
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        bf16=args.bf16,
        do_train=False,
        do_eval=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics(num_labels=4),
    )

    metrics = trainer.evaluate()
    print("=== Evaluation metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    if args.output_metrics_file is not None:
        import json

        out_path = Path(args.output_metrics_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Metrics saved to: {out_path}")


if __name__ == "__main__":
    main()
