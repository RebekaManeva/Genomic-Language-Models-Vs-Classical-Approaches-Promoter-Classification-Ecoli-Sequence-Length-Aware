"""
optimize_dnabert2_nlp.py
========================
Hyperparameter optimisation for DNABERT-2 on DNA promoter classification.

Algorithms (--algorithm):
    rs    Random Search
    ts    Tree Parzen Estimator / TPE    (Optuna)
    bayes Bayesian CMA-ES / TPE          (Optuna)
    ga    Genetic Algorithm
    hc    Hill Climbing
    sa    Simulated Annealing
    sopt  Sequential GP-BO               (scikit-optimize)

Expects a pre-split data directory:
    <data-path>/
        train.csv
        dev.csv
        test.csv

Workflow:
    1. Run N short trials (--search-epochs) to find the best config.
    2. Retrain from scratch with best config for --full-epochs.
    3. Save best_params.json and final_results.json to --output-dir.

Usage:
    python optimize_dnabert2_nlp.py \\
        --data-path /path/to/splits \\
        --output-dir /path/to/outputs/dnabert2_opt \\
        --algorithm ts \\
        --n-trials 30 --search-epochs 5 --full-epochs 50
"""

import os
import sys
import csv
import json
import logging
import argparse
import traceback
from typing import Dict, Sequence, Tuple, Union

import numpy as np
import torch
import transformers
import sklearn.metrics
import optuna
from dataclasses import dataclass
from torch.utils.data import Dataset

from optimize_base import dispatch, ALGO_CHOICES, json_safe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PYTHONPATH is managed by the shell launcher (smoke_test.sh / run_optimization.sh)
# to avoid tokenizers version conflicts between model families. Do not hardcode it here.

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

MODEL_ID = "zhihan1996/DNABERT-2-117M"

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
    p = argparse.ArgumentParser(description="DNABERT-2 optimiser for promoter classification.")
    p.add_argument("--data-path",    required=True,
                   help="Dir with train.csv / dev.csv / test.csv")
    p.add_argument("--model-name",   default=MODEL_ID)
    p.add_argument("--output-dir",   required=True)
    p.add_argument("--algorithm",    required=True, choices=ALGO_CHOICES,
                   help=f"Optimisation algorithm. Choices: {ALGO_CHOICES}")
    p.add_argument("--n-trials",     type=int, default=30)
    p.add_argument("--search-epochs", type=int, default=5,
                   help="Epochs per trial during search phase.")
    p.add_argument("--full-epochs",  type=int, default=50,
                   help="Epochs for final retrain with best config.")
    p.add_argument("--max-length",   type=int, default=512)
    p.add_argument("--kmer",         type=int, default=-1)
    p.add_argument("--fp16",         action="store_true")
    p.add_argument("--study-name",   default="dnabert2_opt",
                   help="Optuna study name (only used for ts/bayes).")
    p.add_argument("--storage",      default=None,
                   help="Optuna storage URI (only used for ts/bayes).")
    p.add_argument("--seed",         type=int, default=SEED)
    p.add_argument("--continue-on-trial-failure", action="store_true",
                   help="Keep searching after a failed trial. By default failures abort the job.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset  (mirrors train.py's SupervisedDataset)
# ─────────────────────────────────────────────────────────────────────────────

def generate_kmer_str(sequence: str, k: int) -> str:
    return " ".join([sequence[i:i + k] for i in range(len(sequence) - k + 1)])


class SupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, kmer: int = -1):
        super().__init__()
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        texts  = [d[0].strip().upper() for d in data]
        labels = [int(d[1]) for d in data]
        if kmer != -1:
            texts = [generate_kmer_str(t, kmer) for t in texts]
        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        self.input_ids      = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels         = labels
        self.num_labels     = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [inst[k] for inst in instances] for k in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.tensor(labels, dtype=torch.long)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_logits_for_metrics(logits: Union[torch.Tensor, Tuple], _):
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    return torch.softmax(logits, dim=-1)


def compute_metrics(eval_pred):
    probs, labels = eval_pred
    if isinstance(probs, tuple):
        probs = probs[0]
    probs  = np.asarray(probs)
    labels = np.asarray(labels)
    mask   = labels != -100
    labels, probs = labels[mask], probs[mask]
    preds     = np.argmax(probs, axis=-1)
    pos_probs = probs[:, 1]
    try:
        auc = sklearn.metrics.roc_auc_score(labels, pos_probs)
    except ValueError:
        auc = float("nan")
    return {
        "accuracy":             sklearn.metrics.accuracy_score(labels, preds),
        "f1":                   sklearn.metrics.f1_score(labels, preds, average="macro", zero_division=0),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(labels, preds),
        "precision":            sklearn.metrics.precision_score(labels, preds, average="macro", zero_division=0),
        "recall":               sklearn.metrics.recall_score(labels, preds, average="macro", zero_division=0),
        "roc_auc":              auc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_model(model_name: str, num_labels: int):
    config = transformers.AutoConfig.from_pretrained(
        model_name, trust_remote_code=True)
    config.num_labels = num_labels
    if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
        config.pad_token_id = 0
    return transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        low_cpu_mem_usage=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Single training run
# ─────────────────────────────────────────────────────────────────────────────

def run_one(args, tokenizer, train_ds, val_ds,
            cfg: dict, num_epochs: int, run_dir: str, select_best_model: bool = False) -> dict:
    """Train for num_epochs with the given cfg dict; return eval metrics."""
    os.makedirs(run_dir, exist_ok=True)
    collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    model    = build_model(args.model_name, train_ds.num_labels)

    total_steps  = (len(train_ds) // (cfg["per_device_bs"] * cfg["grad_accum"])) * num_epochs
    warmup_steps = int(total_steps * cfg["warmup_ratio"])

    training_args = transformers.TrainingArguments(
        output_dir=run_dir,
        learning_rate=cfg["lr"],
        per_device_train_batch_size=cfg["per_device_bs"],
        per_device_eval_batch_size=16,
        num_train_epochs=num_epochs,
        weight_decay=cfg["weight_decay"],
        warmup_steps=warmup_steps,
        lr_scheduler_type=cfg["scheduler"],
        gradient_accumulation_steps=cfg["grad_accum"],
        fp16=args.fp16,
        evaluation_strategy="epoch",
        save_strategy="epoch" if select_best_model else "no",
        load_best_model_at_end=select_best_model,
        metric_for_best_model="eval_f1" if select_best_model else None,
        greater_is_better=True if select_best_model else None,
        save_total_limit=2 if select_best_model else None,
        logging_steps=50,
        report_to="none",
        seed=args.seed,
        dataloader_pin_memory=False,
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    if select_best_model:
        metrics["best_model_checkpoint"] = trainer.state.best_model_checkpoint
        metrics["best_metric"] = trainer.state.best_metric
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Objective closure for optimize_base.dispatch
# ─────────────────────────────────────────────────────────────────────────────

def make_objective(args, tokenizer, train_ds, val_ds, trial_counter: dict):
    def objective(cfg: dict) -> float:
        idx     = trial_counter["n"]
        run_dir = os.path.join(args.output_dir, "trials", f"trial_{idx:04d}")
        try:
            metrics = run_one(args, tokenizer, train_ds, val_ds,
                              cfg=cfg,
                              num_epochs=args.search_epochs,
                              run_dir=run_dir)
            f1 = float(metrics.get("eval_f1", 0.0))
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
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name,
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    # ── Datasets ──────────────────────────────────────────────────────────────
    logger.info("Loading datasets …")
    train_ds = SupervisedDataset(
        os.path.join(args.data_path, "train.csv"), tokenizer, kmer=args.kmer)
    val_ds   = SupervisedDataset(
        os.path.join(args.data_path, "dev.csv"),   tokenizer, kmer=args.kmer)
    test_ds  = SupervisedDataset(
        os.path.join(args.data_path, "test.csv"),  tokenizer, kmer=args.kmer)

    # ── Search ────────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"  Model     : {args.model_name}")
    logger.info(f"  Algorithm : {args.algorithm}")
    logger.info(f"  Trials    : {args.n_trials} × {args.search_epochs} epoch(s)")
    logger.info(f"{'='*60}\n")

    trial_counter = {"n": 0}
    objective_fn  = make_objective(args, tokenizer, train_ds, val_ds, trial_counter)

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
        args, tokenizer, train_ds, val_ds,
        cfg=best_cfg,
        num_epochs=args.full_epochs,
        run_dir=final_dir,
        select_best_model=True,
    )

    result = {
        "model_name":    args.model_name,
        "algorithm":     args.algorithm,
        "best_params":   json_safe(best_cfg),
        "search_val_f1": float(best_score),
        "val_metrics":   {k: float(v) for k, v in final_metrics.items()
                          if isinstance(v, (int, float))},
        "n_trials":      args.n_trials,
        "search_epochs": args.search_epochs,
        "full_epochs":   args.full_epochs,
        "best_model_checkpoint": final_metrics.get("best_model_checkpoint"),
        "best_metric": final_metrics.get("best_metric"),
        "train_size":    len(train_ds),
        "val_size":      len(val_ds),
        "test_size":     len(test_ds),
    }
    out_path = os.path.join(args.output_dir, "final_results.json")
    with open(out_path, "w") as f:
        json.dump(json_safe(result), f, indent=2, allow_nan=False)

    print(json.dumps(json_safe(result), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
