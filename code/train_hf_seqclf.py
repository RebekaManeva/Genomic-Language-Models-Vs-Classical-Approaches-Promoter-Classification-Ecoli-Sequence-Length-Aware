import os
import json
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune a Hugging Face sequence classifier on promoter CSV data.")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", default=None)
    p.add_argument("--test-csv", default=None)
    p.add_argument("--model-name", required=True)
    p.add_argument("--tokenizer-name", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--sequence-col", default="sequence")
    p.add_argument("--label-col", default="label")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--train-batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--save-total-limit", type=int, default=2)
    p.add_argument("--eval-strategy", default="epoch", choices=["no", "steps", "epoch"])
    p.add_argument("--save-strategy", default="epoch", choices=["no", "steps", "epoch"])
    p.add_argument("--logging-steps", type=int, default=20)
    p.add_argument("--metric-for-best-model", default="eval_f1")
    p.add_argument("--greater-is-better", action="store_true")
    p.add_argument("--no-greater-is-better", dest="greater_is_better", action="store_false")
    p.set_defaults(greater_is_better=True)
    p.add_argument(
        "--split-from-single-csv",
        action="store_true",
        help="If set, ignore --val-csv/--test-csv and create 70/15/15 stratified splits from --train-csv."
    )
    return p.parse_args()


def load_and_split(args):
    if args.split_from_single_csv:
        df = pd.read_csv(args.train_csv)
        df = df[[args.sequence_col, args.label_col]].dropna().copy()
        df[args.sequence_col] = df[args.sequence_col].astype(str).str.upper()
        df[args.label_col] = df[args.label_col].astype(int)

        X = df[args.sequence_col]
        y = df[args.label_col]

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=args.seed
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=args.seed
        )

        train_df = pd.DataFrame({args.sequence_col: X_train, args.label_col: y_train})
        val_df = pd.DataFrame({args.sequence_col: X_val, args.label_col: y_val})
        test_df = pd.DataFrame({args.sequence_col: X_test, args.label_col: y_test})
    else:
        if not args.val_csv or not args.test_csv:
            raise ValueError("Provide --val-csv and --test-csv, or use --split-from-single-csv.")

        train_df = pd.read_csv(args.train_csv)
        val_df = pd.read_csv(args.val_csv)
        test_df = pd.read_csv(args.test_csv)

        for frame in (train_df, val_df, test_df):
            frame.dropna(subset=[args.sequence_col, args.label_col], inplace=True)
            frame[args.sequence_col] = frame[args.sequence_col].astype(str).str.upper()
            frame[args.label_col] = frame[args.label_col].astype(int)

    return train_df, val_df, test_df


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    train_df, val_df, test_df = load_and_split(args)

    tokenizer_name = args.tokenizer_name or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=args.trust_remote_code,
    )

    config = AutoConfig.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    config.num_labels = 2

    if not hasattr(config, "is_decoder"):
        config.is_decoder = False

    if not hasattr(config, "add_cross_attention"):
        config.add_cross_attention = False

    if not hasattr(config, "use_return_dict"):
        config.use_return_dict = True

    if not hasattr(config, "output_attentions"):
        config.output_attentions = False

    if not hasattr(config, "output_hidden_states"):
        config.output_hidden_states = False

    if not hasattr(config, "chunk_size_feed_forward"):
        config.chunk_size_feed_forward = 0

    if not hasattr(config, "position_embedding_type"):
        config.position_embedding_type = "absolute"

    if getattr(config, "pad_token_id", None) is None:
        if getattr(tokenizer, "pad_token_id", None) is not None:
            config.pad_token_id = tokenizer.pad_token_id
        elif getattr(tokenizer, "eos_token_id", None) is not None:
            config.pad_token_id = tokenizer.eos_token_id
        elif getattr(tokenizer, "cls_token_id", None) is not None:
            config.pad_token_id = tokenizer.cls_token_id
        else:
            config.pad_token_id = 0

    print("tokenizer.pad_token_id =", getattr(tokenizer, "pad_token_id", None))
    print("config.pad_token_id =", getattr(config, "pad_token_id", None))

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        config=config,
        trust_remote_code=args.trust_remote_code,
        ignore_mismatched_sizes=True,
    )

    if not hasattr(model, "all_tied_weights_keys"):
        tied = getattr(model, "_tied_weights_keys", None)
        if tied is None:
            model.all_tied_weights_keys = {}
        elif isinstance(tied, dict):
            model.all_tied_weights_keys = tied
        else:
            model.all_tied_weights_keys = {k: k for k in tied}

        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()

    def tokenize_batch(batch):
        return tokenizer(
            batch[args.sequence_col],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    train_ds = Dataset.from_pandas(
        train_df[[args.sequence_col, args.label_col]].rename(columns={args.label_col: "labels"}),
        preserve_index=False
    )
    val_ds = Dataset.from_pandas(
        val_df[[args.sequence_col, args.label_col]].rename(columns={args.label_col: "labels"}),
        preserve_index=False
    )
    test_ds = Dataset.from_pandas(
        test_df[[args.sequence_col, args.label_col]].rename(columns={args.label_col: "labels"}),
        preserve_index=False
    )

    train_ds = train_ds.map(tokenize_batch, batched=True)
    val_ds = val_ds.map(tokenize_batch, batched=True)
    test_ds = test_ds.map(tokenize_batch, batched=True)

    keep_cols = ["input_ids", "labels"]

    if "attention_mask" in train_ds.column_names:
        keep_cols.append("attention_mask")

    if "token_type_ids" in train_ds.column_names:
        keep_cols.append("token_type_ids")

    train_ds.set_format(type="torch", columns=keep_cols)
    val_ds.set_format(type="torch", columns=keep_cols)
    test_ds.set_format(type="torch", columns=keep_cols)

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if (args.fp16 or args.bf16) else None
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        preds = np.argmax(logits, axis=-1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        acc = accuracy_score(labels, preds)

        try:
            roc_auc = roc_auc_score(labels, probs)
        except ValueError:
            roc_auc = float("nan")

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
        }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy=args.eval_strategy,
        save_strategy="no",
        load_best_model_at_end=False,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        # save_safetensors=False,
        report_to="none",
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    val_metrics = trainer.evaluate(eval_dataset=val_ds)
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")

    metrics = {
        "model_name": args.model_name,
        "tokenizer_name": tokenizer_name,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "max_length": args.max_length,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()