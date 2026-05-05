"""
optimize_dnabert2.py
Bayesian hyperparameter optimization (Optuna) for DNABERT-2 on promoter classification.
Searches over: lr, warmup, weight_decay, batch_size, scheduler_type, gradient_accumulation.
Runs N_TRIALS trials of SHORT_EPOCHS epochs each, then retrains the best config for FULL_EPOCHS.
"""

import os
import sys
import csv
import json
import logging
import argparse
import copy
from typing import Dict, List, Optional, Tuple, Union, Any, Sequence

import numpy as np
import torch
import transformers
import sklearn
import sklearn.metrics
import optuna
from optuna.samplers import TPESampler
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from peft import LoraConfig, get_peft_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, "/home/hpc/users/ml_models/rebeka.maneva/py_pkgs_dnabert2")


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True,
                   help="Dir with train.csv / dev.csv / test.csv")
    p.add_argument("--model-name", default="zhihan1996/DNABERT-2-117M")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-trials", type=int, default=30,
                   help="Number of Optuna trials")
    p.add_argument("--search-epochs", type=int, default=5,
                   help="Epochs per Optuna trial (short)")
    p.add_argument("--full-epochs", type=int, default=50,
                   help="Epochs for final retrain with best params")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--kmer", type=int, default=-1)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--study-name", default="dnabert2_opt")
    p.add_argument("--storage", default=None,
                   help="Optuna storage URI, e.g. sqlite:///study.db")
    return p.parse_args()


def generate_kmer_str(sequence: str, k: int) -> str:
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

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
            texts, return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        self.input_ids     = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels        = labels
        self.num_labels    = len(set(labels))

    def __len__(self): return len(self.input_ids)
    def __getitem__(self, i): return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset:
    tokenizer: transformers.PreTrainedTokenizer
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [inst[k] for inst in instances] for k in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.tensor(labels, dtype=torch.long)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def preprocess_logits_for_metrics(logits: Union[torch.Tensor, Tuple], _):
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.ndim == 3:
        logits = logits.reshape(-1, logits.shape[-1])
    return torch.argmax(logits, dim=-1)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    preds  = logits.reshape(-1) if logits.ndim == 1 else np.argmax(logits, axis=-1)
    mask   = labels != -100
    preds, labels = preds[mask], labels[mask]
    return {
        "accuracy":              sklearn.metrics.accuracy_score(labels, preds),
        "f1":                    sklearn.metrics.f1_score(labels, preds, average="macro", zero_division=0),
        "matthews_correlation":  sklearn.metrics.matthews_corrcoef(labels, preds),
        "precision":             sklearn.metrics.precision_score(labels, preds, average="macro", zero_division=0),
        "recall":                sklearn.metrics.recall_score(labels, preds, average="macro", zero_division=0),
    }


def build_model(model_name: str, num_labels: int, cache_dir=None):
    config = transformers.AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    config.num_labels = num_labels
    if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
        config.pad_token_id = 0

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True,
        cache_dir=cache_dir,
        ignore_mismatched_sizes=True,
    )
    return model


def run_training(
    args,
    tokenizer,
    train_ds, val_ds,
    lr: float,
    warmup_ratio: float,
    weight_decay: float,
    per_device_bs: int,
    grad_accum: int,
    scheduler_type: str,
    num_epochs: int,
    run_output_dir: str,
) -> Dict:
    collator   = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    model      = build_model(args.model_name, train_ds.num_labels)

    total_steps = (len(train_ds) // (per_device_bs * grad_accum)) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)

    training_args = transformers.TrainingArguments(
        output_dir=run_output_dir,
        learning_rate=lr,
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=16,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type=scheduler_type,
        gradient_accumulation_steps=grad_accum,
        fp16=args.fp16,
        evaluation_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        logging_steps=50,
        report_to="none",
        seed=SEED,
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
    return metrics


def make_objective(args, tokenizer, train_ds, val_ds):
    def objective(trial: optuna.Trial) -> float:
        lr            = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
        warmup_ratio  = trial.suggest_float("warmup_ratio", 0.0, 0.2)
        weight_decay  = trial.suggest_float("weight_decay", 0.0, 0.1)
        per_device_bs = trial.suggest_categorical("per_device_bs", [4, 8, 16])
        grad_accum    = trial.suggest_categorical("grad_accum", [1, 2, 4])
        scheduler     = trial.suggest_categorical("scheduler", ["linear", "cosine", "cosine_with_restarts"])

        run_dir = os.path.join(args.output_dir, "trials", f"trial_{trial.number}")
        os.makedirs(run_dir, exist_ok=True)

        try:
            metrics = run_training(
                args, tokenizer, train_ds, val_ds,
                lr=lr,
                warmup_ratio=warmup_ratio,
                weight_decay=weight_decay,
                per_device_bs=per_device_bs,
                grad_accum=grad_accum,
                scheduler_type=scheduler,
                num_epochs=args.search_epochs,
                run_output_dir=run_dir,
            )
            f1 = metrics.get("eval_f1", 0.0)
        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            f1 = 0.0
            metrics = {}

        trial.set_user_attr("metrics", {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        return f1
    return objective


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name,
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    # Datasets
    logger.info("Loading datasets …")
    train_ds = SupervisedDataset(
        os.path.join(args.data_path, "train.csv"), tokenizer, kmer=args.kmer
    )
    val_ds = SupervisedDataset(
        os.path.join(args.data_path, "dev.csv"), tokenizer, kmer=args.kmer
    )
    test_ds = SupervisedDataset(
        os.path.join(args.data_path, "test.csv"), tokenizer, kmer=args.kmer
    )

    sampler = TPESampler(seed=SEED)
    storage = args.storage or f"sqlite:///{os.path.join(args.output_dir, 'study.db')}"
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(
        make_objective(args, tokenizer, train_ds, val_ds),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    best = study.best_params
    logger.info(f"Best params: {best}")
    with open(os.path.join(args.output_dir, "best_params.json"), "w") as f:
        json.dump(best, f, indent=2)

    logger.info(f"Retraining best config for {args.full_epochs} epochs …")
    final_dir = os.path.join(args.output_dir, "best_model")
    os.makedirs(final_dir, exist_ok=True)

    final_metrics = run_training(
        args, tokenizer, train_ds, val_ds,
        lr=best["lr"],
        warmup_ratio=best["warmup_ratio"],
        weight_decay=best["weight_decay"],
        per_device_bs=best["per_device_bs"],
        grad_accum=best["grad_accum"],
        scheduler_type=best["scheduler"],
        num_epochs=args.full_epochs,
        run_output_dir=final_dir,
    )

    result = {
        "best_params": best,
        "val_metrics": {k: float(v) for k, v in final_metrics.items() if isinstance(v, (int, float))},
        "n_trials": args.n_trials,
        "search_epochs": args.search_epochs,
        "full_epochs": args.full_epochs,
    }
    with open(os.path.join(args.output_dir, "final_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
