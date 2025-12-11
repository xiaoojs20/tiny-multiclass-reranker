import argparse
from pathlib import Path
import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType


from esci_dataset import (
    load_esci_parquet,
    ESCIMultiClassRerankDataset,
)
from utils import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen3-0.6B on ESCI multi-class reranking (LoRA).")

    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--train_file", type=str, required=True)
    # parser.add_argument("--eval_file", type=str, required=True)
    parser.add_argument("--eval_ratio", type=float, default=0.05,
                        help="Ratio of training data to use as eval set (e.g. 0.05 for 5%)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no_flash_attn", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--eval_steps", type=int, default=100,  
                    help="Run evaluation every N training steps when eval_dataset is provided.")


    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",                    
        default=["q_proj", "v_proj"], 
        help="List of target modules for LoRA (e.g. --target_modules q_proj v_proj gate_proj up_proj down_proj)",
    )

    parser.add_argument("--seed", type=int, default=42)

    # Logging & reporting
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        choices=["none", "wandb"],
        help='Where to report training logs. Use "wandb" to enable Weights & Biases logging.',
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Wandb project name. If None, will use WANDB_PROJECT env var if set.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Wandb run name (also used as TrainingArguments.run_name).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model from: {args.base_model}")
    dtype = torch.bfloat16 if args.bf16 else torch.float16

    model_kwargs = {
        "dtype": dtype,
        # "device_map": "auto",
    }
    if not args.no_flash_attn:
        # 如果版本不支持，可删除这一行或改成 attn_implementation="eager"
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        **model_kwargs,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    print("Applying LoRA...")
    print(f"target_modules: {args.target_modules}")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading train data from: {args.train_file}")
    df_train = load_esci_parquet(args.train_file)
    print(f"Train size: {len(df_train)}")

    # print(f"Loading eval data from: {args.eval_file}")
    # df_eval = load_esci_parquet(args.eval_file)
    # print(f"Eval size: {len(df_eval)}")

    # 从train data里划分一部分作为eval set
    df_all = df_train

    eval_ratio = args.eval_ratio
    if not (0.0 <= eval_ratio < 1.0):
        raise ValueError(f"eval_ratio must be in [0,1), got {eval_ratio}")

    if eval_ratio > 0.0:
        # 简单随机划分
        df_all_shuffled = df_all.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        eval_size = int(len(df_all_shuffled) * eval_ratio)
        if eval_size == 0:
            # 数据太少时保护一下
            print(f"[Warn] eval_ratio={eval_ratio} but dataset is small, eval_size=0. "
                  f"Disable eval and use all data for training.")
            df_train = df_all_shuffled
            df_eval = None
        else:
            df_eval = df_all_shuffled.iloc[:eval_size].reset_index(drop=True)
            df_train = df_all_shuffled.iloc[eval_size:].reset_index(drop=True)
    else:
        df_train = df_all
        df_eval = None

    print(f"Train size: {len(df_train)}")
    if df_eval is not None:
        print(f"Eval size:  {len(df_eval)}")
    else:
        print("Eval disabled (no eval split).")


    train_dataset = ESCIMultiClassRerankDataset(
        df_train,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    # eval_dataset = ESCIMultiClassRerankDataset(
    #     df_eval,
    #     tokenizer=tokenizer,
    #     max_length=args.max_length,
    # )

    eval_dataset = None
    if df_eval is not None:
        eval_dataset = ESCIMultiClassRerankDataset(
            df_eval,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )


    # ========= TrainingArguments =========
    report_to = args.report_to
    if args.report_to == "wandb":
        if args.wandb_project is not None:
            os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_run_name is not None:
            os.environ["WANDB_RUN_NAME"] = args.wandb_run_name 
    logging_dir = str(output_dir / "logs")

    save_total_limit = None if args.save_total_limit < 0 else args.save_total_limit

    if eval_dataset is not None:
        evaluation_strategy = "steps"
        eval_steps = args.eval_steps
    else:
        evaluation_strategy = "no"
        eval_steps = None

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=save_total_limit,
        bf16=args.bf16,
        report_to=report_to,
        logging_dir=logging_dir,
        run_name=args.wandb_run_name,
        eval_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    # 保存 LoRA adapter
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"LoRA adapter & tokenizer saved to: {output_dir}")


if __name__ == "__main__":
    main()

