"""
optimize_hf_seqclf_nlp.py
==========================
Hyperparameter optimisation for HyenaDNA and Nucleotide Transformer v2
on DNA promoter sequence classification.

Algorithms (--algorithm):
    rs    Random Search
    ts    Tree Parzen Estimator / TPE    (Optuna)
    bayes Bayesian CMA-ES / TPE          (Optuna)
    ga    Genetic Algorithm
    hc    Hill Climbing
    sa    Simulated Annealing
    sopt  Sequential GP-BO               (scikit-optimize)

Workflow:
    1. Run N short trials (--search-epochs) to find the best config.
    2. Retrain from scratch with the best config for --full-epochs.
    3. Save best_params.json and final_results.json to --output-dir.

Usage:
    python optimize_hf_seqclf_nlp.py \\
        --train-csv /path/to/data.csv \\
        --split-from-single-csv \\
        --model-name LongSafari/hyenadna-tiny-16k-seqlen-d128-hf \\
        --output-dir /path/to/outputs/hyena_opt \\
        --algorithm ts \\
        --n-trials 30 --search-epochs 5 --full-epochs 50 \\
        --trust-remote-code
"""

import os
import json
import copy
import logging
import argparse
import traceback

import numpy as np
import torch
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
)
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer, TrainingArguments, set_seed,
)

from optimize_base import dispatch, ALGO_CHOICES, json_safe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Search space
# ─────────────────────────────────────────────────────────────────────────────

SEARCH_SPACE = {
    "lr":            [1e-5, 3e-5, 5e-5, 1e-4],
    "warmup_ratio":  [0.0, 0.05, 0.1, 0.2],
    "weight_decay":  [0.0, 0.01, 0.05, 0.1],
    "per_device_bs": [4, 8, 16],
    "grad_accum":    [1, 2, 4],
    "scheduler":     ["linear", "cosine", "cosine_with_restarts"],
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="HF model optimiser for promoter classification.")
    p.add_argument("--train-csv",             required=True)
    p.add_argument("--val-csv",               default=None)
    p.add_argument("--test-csv",              default=None)
    p.add_argument("--split-from-single-csv", action="store_true")
    p.add_argument("--model-name",            required=True)
    p.add_argument("--tokenizer-name",        default=None)
    p.add_argument("--output-dir",            required=True)
    p.add_argument("--sequence-col",          default="sequence")
    p.add_argument("--label-col",             default="label")
    p.add_argument("--max-length",            type=int, default=512)
    p.add_argument("--algorithm",             required=True, choices=ALGO_CHOICES,
                   help=f"Optimisation algorithm. Choices: {ALGO_CHOICES}")
    p.add_argument("--n-trials",              type=int, default=30)
    p.add_argument("--search-epochs",         type=int, default=5,
                   help="Short epochs per trial during search phase.")
    p.add_argument("--full-epochs",           type=int, default=50,
                   help="Epochs for final retrain with best config.")
    p.add_argument("--fp16",                  action="store_true")
    p.add_argument("--bf16",                  action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--trust-remote-code",     action="store_true")
    p.add_argument("--study-name",            default="hf_seqclf_opt",
                   help="Optuna study name (only used for ts/bayes).")
    p.add_argument("--storage",               default=None,
                   help="Optuna storage URI, e.g. sqlite:///study.db "
                        "(only used for ts/bayes).")
    p.add_argument("--seed",                  type=int, default=SEED)
    p.add_argument("--continue-on-trial-failure", action="store_true",
                   help="Keep searching after a failed trial. By default failures abort the job.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_splits(args):
    if args.split_from_single_csv:
        df = pd.read_csv(args.train_csv)
        df = df[[args.sequence_col, args.label_col]].dropna().copy()
        df[args.sequence_col] = df[args.sequence_col].astype(str).str.upper()
        df[args.label_col]    = df[args.label_col].astype(int)
        X, y = df[args.sequence_col], df[args.label_col]
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=args.seed)
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=args.seed)
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


# ─────────────────────────────────────────────────────────────────────────────
# HF dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

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
    if "attention_mask"  in ds.column_names: keep.append("attention_mask")
    if "token_type_ids"  in ds.column_names: keep.append("token_type_ids")
    ds.set_format(type="torch", columns=keep)
    return ds


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    logits = np.asarray(logits)
    labels = np.asarray(labels)

    if logits.ndim == 1:
        pos_scores = logits
        preds = (pos_scores >= 0).astype(int)
    else:
        preds = np.argmax(logits, axis=-1)
        if logits.shape[-1] >= 2:
            pos_scores = logits[:, 1] - logits[:, 0]
        else:
            pos_scores = logits[:, 0]

    pos_scores = np.nan_to_num(pos_scores, nan=0.0, posinf=1e6, neginf=-1e6)
    pred_counts = np.bincount(preds, minlength=2)
    label_counts = np.bincount(labels, minlength=2)
    collapsed = int(pred_counts[0] == 0 or pred_counts[1] == 0)
    try:
        roc = roc_auc_score(labels, pos_scores)
    except ValueError:
        roc = float("nan")
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    selection_score = 0.0 if collapsed else f1
    return {
        "accuracy":  accuracy_score(labels, preds),
        "f1":        f1,
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall":    recall_score(labels, preds, average="macro", zero_division=0),
        "roc_auc":   roc,
        "selection_score": selection_score,
        "collapsed_predictions": collapsed,
        "pred_0":    int(pred_counts[0]),
        "pred_1":    int(pred_counts[1]),
        "label_0":   int(label_counts[0]),
        "label_1":   int(label_counts[1]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_model_and_tokenizer(args):
    tok_name  = args.tokenizer_name or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tok_name, trust_remote_code=args.trust_remote_code)
    config = AutoConfig.from_pretrained(
        args.model_name, trust_remote_code=args.trust_remote_code)
    config.num_labels = 2
    if not hasattr(config, "is_decoder"):
        config.is_decoder = False
    if not hasattr(config, "add_cross_attention"):
        config.add_cross_attention = False
    if getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = (
            getattr(tokenizer, "pad_token_id", None)
            or getattr(tokenizer, "eos_token_id", None)
            or 0
        )
    return tokenizer, config


# ─────────────────────────────────────────────────────────────────────────────
# Single training run
# ─────────────────────────────────────────────────────────────────────────────

def run_one(args, tokenizer, config, train_ds_hf, val_ds_hf, cfg: dict,
            num_epochs: int, run_dir: str, select_best_model: bool = False,
            test_ds_hf=None) -> dict:
    """Train for num_epochs with the given cfg dict; return eval metrics."""
    os.makedirs(run_dir, exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, config=config,
        trust_remote_code=args.trust_remote_code,
        ignore_mismatched_sizes=True,
    )
    if args.gradient_checkpointing:
        try:
            model.gradient_checkpointing_enable()
        except (ValueError, AttributeError):
            logger.warning("Gradient checkpointing not supported by this model, skipping.")

    if not hasattr(model, "all_tied_weights_keys"):
        tied = getattr(model, "_tied_weights_keys", None)
        if tied is None:
            model.all_tied_weights_keys = {}
        elif isinstance(tied, dict):
            model.all_tied_weights_keys = tied
        else:
            model.all_tied_weights_keys = {k: k for k in tied}

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if (args.fp16 or args.bf16) else None,
    )

    ta = TrainingArguments(
        output_dir=run_dir,
        learning_rate=cfg["lr"],
        per_device_train_batch_size=cfg["per_device_bs"],
        per_device_eval_batch_size=16,
        num_train_epochs=num_epochs,
        weight_decay=cfg["weight_decay"],
        warmup_ratio=cfg["warmup_ratio"],
        lr_scheduler_type=cfg["scheduler"],
        gradient_accumulation_steps=cfg["grad_accum"],
        fp16=args.fp16,
        bf16=args.bf16,
        eval_strategy="epoch",
        save_strategy="epoch" if select_best_model else "no",
        load_best_model_at_end=select_best_model,
        metric_for_best_model="eval_selection_score" if select_best_model else None,
        greater_is_better=True if select_best_model else None,
        save_total_limit=2 if select_best_model else None,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model, args=ta,
        train_dataset=train_ds_hf,
        eval_dataset=val_ds_hf,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    if test_ds_hf is not None:
        metrics["test_metrics"] = trainer.evaluate(
            eval_dataset=test_ds_hf,
            metric_key_prefix="test",
        )
    if select_best_model:
        metrics["best_model_checkpoint"] = trainer.state.best_model_checkpoint
        metrics["best_metric"] = trainer.state.best_metric
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Objective closure for optimize_base.dispatch
# ─────────────────────────────────────────────────────────────────────────────

def make_objective(args, tokenizer, config, train_ds_hf, val_ds_hf,
                   trial_counter: dict):
    """
    Returns a callable cfg -> float that optimize_base algorithms can call.
    trial_counter is a mutable dict {"n": 0} so the closure can track the index.
    """
    def objective(cfg: dict) -> float:
        idx     = trial_counter["n"]
        run_dir = os.path.join(args.output_dir, "trials", f"trial_{idx:04d}")
        try:
            metrics = run_one(args, tokenizer, config,
                              train_ds_hf, val_ds_hf,
                              cfg=cfg,
                              num_epochs=args.search_epochs,
                              run_dir=run_dir)
            f1 = float(metrics.get("eval_selection_score", 0.0))
            eval_loss = float(metrics.get("eval_loss", 0.0))
            if not math.isfinite(eval_loss):
                f1 = 0.0
        except Exception as e:
            logger.error("Trial %s failed for cfg=%s\n%s", idx, cfg, traceback.format_exc())
            if not args.continue_on_trial_failure:
                raise
            f1 = 0.0
        trial_counter["n"] += 1
        return f1

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    train_df, val_df, test_df = load_splits(args)
    tokenizer, config         = build_model_and_tokenizer(args)

    logger.info("Tokenizing datasets …")
    train_ds_hf = tokenize_dataset(
        df_to_hf_dataset(train_df, args.sequence_col, args.label_col),
        tokenizer, args.sequence_col, args.max_length)
    val_ds_hf   = tokenize_dataset(
        df_to_hf_dataset(val_df, args.sequence_col, args.label_col),
        tokenizer, args.sequence_col, args.max_length)
    test_ds_hf  = tokenize_dataset(
        df_to_hf_dataset(test_df, args.sequence_col, args.label_col),
        tokenizer, args.sequence_col, args.max_length)

    # ── Search ───────────────────────────────────────────────────────────────
    algo_name = ALGO_CHOICES[ALGO_CHOICES.index(args.algorithm)]
    logger.info(f"\n{'='*60}")
    logger.info(f"  Model     : {args.model_name}")
    logger.info(f"  Algorithm : {algo_name}")
    logger.info(f"  Trials    : {args.n_trials} × {args.search_epochs} epoch(s)")
    logger.info(f"{'='*60}\n")

    trial_counter = {"n": 0}
    objective_fn  = make_objective(
        args, tokenizer, config, train_ds_hf, val_ds_hf, trial_counter)

    # Optuna-specific kwargs (silently ignored by non-Optuna algorithms)
    extra_kw = {
        "study_name": args.study_name,
        "storage":    args.storage or f"sqlite:///{os.path.join(args.output_dir, 'study.db')}",
    }

    best_cfg, best_score, all_results = dispatch(
        algorithm=args.algorithm,
        objective_fn=objective_fn,
        space=SEARCH_SPACE,
        n_trials=args.n_trials,
        seed=args.seed,
        **extra_kw,
    )

    logger.info(f"\nBest val F1 after search: {best_score:.4f}")
    logger.info(f"Best config: {json.dumps(best_cfg, indent=2)}")

    # Save all trial results
    all_results_path = os.path.join(args.output_dir, "all_results.jsonl")
    with open(all_results_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(json_safe(r), allow_nan=False) + "\n")

    with open(os.path.join(args.output_dir, "best_params.json"), "w") as f:
        json.dump(json_safe(best_cfg), f, indent=2, allow_nan=False)

    # ── Final retrain ─────────────────────────────────────────────────────────
    logger.info(f"\nRetraining best config for {args.full_epochs} epochs …")
    final_dir = os.path.join(args.output_dir, "best_model")
    final_metrics = run_one(
        args, tokenizer, config,
        train_ds_hf, val_ds_hf,
        cfg=best_cfg,
        num_epochs=args.full_epochs,
        run_dir=final_dir,
        select_best_model=True,
        test_ds_hf=test_ds_hf,
    )
    test_metrics = final_metrics.pop("test_metrics", None)

    result = {
        "model_name":    args.model_name,
        "algorithm":     args.algorithm,
        "best_params":   json_safe(best_cfg),
        "search_val_f1": float(best_score),
        "val_metrics":   {k: float(v) for k, v in final_metrics.items()
                          if isinstance(v, (int, float))},
        "test_metrics":  {k: float(v) for k, v in test_metrics.items()
                          if isinstance(v, (int, float))} if test_metrics else None,
        "n_trials":      args.n_trials,
        "search_epochs": args.search_epochs,
        "full_epochs":   args.full_epochs,
        "best_model_checkpoint": final_metrics.get("best_model_checkpoint"),
        "best_metric": final_metrics.get("best_metric"),
        "train_size":    len(train_df),
        "val_size":      len(val_df),
        "test_size":     len(test_df),
    }
    out_path = os.path.join(args.output_dir, "final_results.json")
    with open(out_path, "w") as f:
        json.dump(json_safe(result), f, indent=2, allow_nan=False)

    print(json.dumps(json_safe(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
