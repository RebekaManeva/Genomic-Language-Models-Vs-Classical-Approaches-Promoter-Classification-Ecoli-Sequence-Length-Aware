"""
optimize_hf_seqclf.py
Bayesian hyperparameter optimization (Optuna) for HyenaDNA and Nucleotide Transformer
on promoter sequence classification.
Uses the same tokenization/training logic as train_hf_seqclf.py.

Usage:
    python optimize_hf_seqclf.py \
        --train-csv /path/to/data.csv \
        --split-from-single-csv \
        --model-name LongSafari/hyenadna-tiny-16k-seqlen-d128-hf \
        --output-dir /path/to/outputs/hyena_opt \
        --n-trials 30 --search-epochs 5 --full-epochs 50 \
        --trust-remote-code
"""

import os
import json
import argparse
import logging

import numpy as np
import torch
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score
)
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer, TrainingArguments, set_seed,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEED = 42

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", default=None)
    p.add_argument("--test-csv", default=None)
    p.add_argument("--split-from-single-csv", action="store_true")
    p.add_argument("--model-name", required=True)
    p.add_argument("--tokenizer-name", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--sequence-col", default="sequence")
    p.add_argument("--label-col", default="label")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--n-trials", type=int, default=30)
    p.add_argument("--search-epochs", type=int, default=5,
                   help="Short epochs per Optuna trial")
    p.add_argument("--full-epochs", type=int, default=50,
                   help="Epochs for final retrain")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--study-name", default="hf_seqclf_opt")
    p.add_argument("--storage", default=None)
    return p.parse_args()

def load_splits(args):
    if args.split_from_single_csv:
        df = pd.read_csv(args.train_csv)
        df = df[[args.sequence_col, args.label_col]].dropna().copy()
        df[args.sequence_col] = df[args.sequence_col].astype(str).str.upper()
        df[args.label_col]    = df[args.label_col].astype(int)
        X, y = df[args.sequence_col], df[args.label_col]
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30,
                                                     stratify=y, random_state=SEED)
        X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50,
                                                     stratify=y_tmp, random_state=SEED)
        train_df = pd.DataFrame({args.sequence_col: X_tr,  args.label_col: y_tr})
        val_df   = pd.DataFrame({args.sequence_col: X_val, args.label_col: y_val})
        test_df  = pd.DataFrame({args.sequence_col: X_te,  args.label_col: y_te})
    else:
        train_df = pd.read_csv(args.train_csv)
        val_df   = pd.read_csv(args.val_csv)
        test_df  = pd.read_csv(args.test_csv)
        for df in (train_df, val_df, test_df):
            df.dropna(subset=[args.sequence_col, args.label_col], inplace=True)
            df[args.sequence_col] = df[args.sequence_col].astype(str).str.upper()
            df[args.label_col]    = df[args.label_col].astype(int)
    return train_df, val_df, test_df

def df_to_hf_dataset(df, sequence_col, label_col):
    return Dataset.from_pandas(
        df[[sequence_col, label_col]].rename(columns={label_col: "labels"}),
        preserve_index=False,
    )

def tokenize_dataset(ds, tokenizer, sequence_col, max_length):
    def _tok(batch):
        return tokenizer(batch[sequence_col], truncation=True,
                         max_length=max_length, padding=False)
    ds = ds.map(_tok, batched=True)
    keep = ["input_ids", "labels"]
    if "attention_mask" in ds.column_names:  keep.append("attention_mask")
    if "token_type_ids" in ds.column_names:  keep.append("token_type_ids")
    ds.set_format(type="torch", columns=keep)
    return ds

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = np.argmax(logits, axis=-1)
    try:
        roc = roc_auc_score(labels, probs)
    except ValueError:
        roc = float("nan")
    return {
        "accuracy":  accuracy_score(labels, preds),
        "f1":        f1_score(labels, preds, average="macro", zero_division=0),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall":    recall_score(labels, preds, average="macro", zero_division=0),
        "roc_auc":   roc,
    }

def build_model_and_tokenizer(args):
    tok_name  = args.tokenizer_name or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_name,
                                              trust_remote_code=args.trust_remote_code)
    config = AutoConfig.from_pretrained(args.model_name,
                                        trust_remote_code=args.trust_remote_code)
    config.num_labels = 2
    if not hasattr(config, "is_decoder"):
        config.is_decoder = False
    if not hasattr(config, "add_cross_attention"):
        config.add_cross_attention = False
    # pad_token fallback
    if getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = (
            getattr(tokenizer, "pad_token_id", None)
            or getattr(tokenizer, "eos_token_id", None)
            or 0
        )
    return tokenizer, config

def run_one(
    args, tokenizer, config,
    train_ds_hf, val_ds_hf,
    lr, warmup_ratio, weight_decay,
    per_device_bs, grad_accum, scheduler,
    num_epochs, run_dir,
):
    os.makedirs(run_dir, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, config=config,
        trust_remote_code=args.trust_remote_code,
        ignore_mismatched_sizes=True,
    )
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
        except ValueError:
            logger.warning("Model does not support gradient checkpointing, skipping.")

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if (args.fp16 or args.bf16) else None,
    )

    ta = TrainingArguments(
        output_dir=run_dir,
        learning_rate=lr,
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=16,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=scheduler,
        gradient_accumulation_steps=grad_accum,
        fp16=args.fp16,
        bf16=args.bf16,
        eval_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        logging_steps=50,
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=model, args=ta,
        train_dataset=train_ds_hf,
        eval_dataset=val_ds_hf,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer.evaluate()

def make_objective(args, tokenizer, config, train_ds_hf, val_ds_hf):
    def objective(trial: optuna.Trial) -> float:
        lr            = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
        warmup_ratio  = trial.suggest_float("warmup_ratio", 0.0, 0.2)
        weight_decay  = trial.suggest_float("weight_decay", 0.0, 0.1)
        per_device_bs = trial.suggest_categorical("per_device_bs", [4, 8, 16])
        grad_accum    = trial.suggest_categorical("grad_accum", [1, 2, 4])
        scheduler     = trial.suggest_categorical("scheduler",
                                                   ["linear", "cosine", "cosine_with_restarts"])

        run_dir = os.path.join(args.output_dir, "trials", f"trial_{trial.number}")
        try:
            metrics = run_one(
                args, tokenizer, config,
                train_ds_hf, val_ds_hf,
                lr=lr, warmup_ratio=warmup_ratio,
                weight_decay=weight_decay,
                per_device_bs=per_device_bs,
                grad_accum=grad_accum,
                scheduler=scheduler,
                num_epochs=args.search_epochs,
                run_dir=run_dir,
            )
            f1 = float(metrics.get("eval_f1", 0.0))
        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            f1 = 0.0
            metrics = {}
        trial.set_user_attr("raw_metrics",
                            {k: float(v) for k, v in metrics.items()
                             if isinstance(v, (int, float))})
        return f1
    return objective

def main():
    args = parse_args()
    set_seed(SEED)
    os.makedirs(args.output_dir, exist_ok=True)

    train_df, val_df, test_df = load_splits(args)
    tokenizer, config         = build_model_and_tokenizer(args)

    logger.info("Tokenizing datasets …")
    train_ds_hf = tokenize_dataset(
        df_to_hf_dataset(train_df, args.sequence_col, args.label_col),
        tokenizer, args.sequence_col, args.max_length
    )
    val_ds_hf   = tokenize_dataset(
        df_to_hf_dataset(val_df, args.sequence_col, args.label_col),
        tokenizer, args.sequence_col, args.max_length
    )
    test_ds_hf  = tokenize_dataset(
        df_to_hf_dataset(test_df, args.sequence_col, args.label_col),
        tokenizer, args.sequence_col, args.max_length
    )

    storage = args.storage or f"sqlite:///{os.path.join(args.output_dir, 'study.db')}"
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=TPESampler(seed=SEED),
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(
        make_objective(args, tokenizer, config, train_ds_hf, val_ds_hf),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    best = study.best_params
    logger.info(f"Best params: {best}")
    with open(os.path.join(args.output_dir, "best_params.json"), "w") as f:
        json.dump(best, f, indent=2)

    # Final retrain
    logger.info(f"Final retrain for {args.full_epochs} epochs …")
    final_dir = os.path.join(args.output_dir, "best_model")
    final_metrics = run_one(
        args, tokenizer, config,
        train_ds_hf, val_ds_hf,
        lr=best["lr"], warmup_ratio=best["warmup_ratio"],
        weight_decay=best["weight_decay"],
        per_device_bs=best["per_device_bs"],
        grad_accum=best["grad_accum"],
        scheduler=best["scheduler"],
        num_epochs=args.full_epochs,
        run_dir=final_dir,
    )

    result = {
        "model_name":    args.model_name,
        "best_params":   best,
        "val_metrics":   {k: float(v) for k, v in final_metrics.items() if isinstance(v, (int, float))},
        "n_trials":      args.n_trials,
        "search_epochs": args.search_epochs,
        "full_epochs":   args.full_epochs,
        "train_size":    len(train_df),
        "val_size":      len(val_df),
        "test_size":     len(test_df),
    }
    with open(os.path.join(args.output_dir, "final_results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
