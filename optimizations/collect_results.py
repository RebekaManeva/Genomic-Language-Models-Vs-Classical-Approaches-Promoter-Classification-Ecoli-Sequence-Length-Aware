"""
Collect optimization results into one ranked summary.

The expected output layout is:

    outputs/opt_<model>_<window>bp/final_results.json        # optuna
    outputs/opt_<model>_<window>bp/<algorithm>/final_results.json
    outputs/opt_<model>_<window>bp/<algorithm>/all_results.jsonl

If an algorithm folder has no final_results.json, the best row is derived
from all_results.jsonl.
"""

import argparse
import csv
import json
import math
import os
import re
import sys


OPT_DIR_RE = re.compile(r"^opt_(?P<model>.+)_(?P<window>\d+)bp$")
SKIP_DIRS = {"best_model", "trials", "sopt", "__pycache__"}


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def metric_value(data, *names):
    for name in names:
        if name in data:
            return data[name]

    metrics = data.get("val_metrics") or {}
    for name in names:
        if name in metrics:
            return metrics[name]

    return None


def as_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def infer_best_from_jsonl(path):
    best = None
    best_score = float("-inf")
    n_trials = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            n_trials += 1
            data = json.loads(line)
            score = as_float(metric_value(data, "val_f1", "search_val_f1", "eval_f1"), float("-inf"))
            if score > best_score:
                best = data
                best_score = score

    if best is None:
        return None

    return {
        "best_params": best.get("config") or best.get("best_params") or {},
        "search_val_f1": best_score,
        "n_trials": n_trials,
        "best_trial": best.get("trial"),
    }


def make_row(model, window_bp, algorithm, path, data, source_kind):
    metrics = data.get("val_metrics") or {}
    eval_f1 = metric_value(data, "eval_f1")
    search_val_f1 = metric_value(data, "search_val_f1", "val_f1")
    rank_score = as_float(eval_f1, as_float(search_val_f1, float("-inf")))

    return {
        "model": model,
        "window_bp": int(window_bp),
        "algorithm": data.get("algorithm") or algorithm,
        "rank_score": rank_score,
        "eval_f1": eval_f1,
        "search_val_f1": search_val_f1,
        "eval_accuracy": metrics.get("eval_accuracy"),
        "eval_precision": metrics.get("eval_precision"),
        "eval_recall": metrics.get("eval_recall"),
        "eval_roc_auc": metrics.get("eval_roc_auc"),
        "eval_loss": metrics.get("eval_loss"),
        "n_trials": data.get("n_trials"),
        "best_trial": data.get("best_trial"),
        "best_metric": data.get("best_metric"),
        "best_model_checkpoint": data.get("best_model_checkpoint"),
        "search_epochs": data.get("search_epochs"),
        "full_epochs": data.get("full_epochs"),
        "train_size": data.get("train_size"),
        "val_size": data.get("val_size"),
        "test_size": data.get("test_size"),
        "model_name": data.get("model_name"),
        "best_params": json.dumps(data.get("best_params") or {}, sort_keys=True),
        "source_kind": source_kind,
        "path": path,
    }


def collect_opt_dir(opt_dir):
    name = os.path.basename(opt_dir.rstrip(os.sep))
    match = OPT_DIR_RE.match(name)
    if not match:
        return []

    model = match.group("model")
    window_bp = match.group("window")
    rows = []

    direct_final = os.path.join(opt_dir, "final_results.json")
    if os.path.isfile(direct_final):
        rows.append(make_row(
            model=model,
            window_bp=window_bp,
            algorithm="optuna",
            path=direct_final,
            data=read_json(direct_final),
            source_kind="final_results",
        ))

    for entry in sorted(os.listdir(opt_dir)):
        algo_dir = os.path.join(opt_dir, entry)
        if entry in SKIP_DIRS or not os.path.isdir(algo_dir):
            continue

        final_path = os.path.join(algo_dir, "final_results.json")
        all_results_path = os.path.join(algo_dir, "all_results.jsonl")

        if os.path.isfile(final_path):
            rows.append(make_row(
                model=model,
                window_bp=window_bp,
                algorithm=entry,
                path=final_path,
                data=read_json(final_path),
                source_kind="final_results",
            ))
        elif os.path.isfile(all_results_path):
            data = infer_best_from_jsonl(all_results_path)
            if data is not None:
                rows.append(make_row(
                    model=model,
                    window_bp=window_bp,
                    algorithm=entry,
                    path=all_results_path,
                    data=data,
                    source_kind="all_results_best_trial",
                ))

    return rows


def sort_rows(rows):
    def key(row):
        score = as_float(row["rank_score"], float("-inf"))
        if math.isnan(score):
            score = float("-inf")
        return (score, row["model"], -row["window_bp"], row["algorithm"])

    return sorted(rows, key=key, reverse=True)


def write_csv(path, rows):
    fieldnames = [
        "model",
        "window_bp",
        "algorithm",
        "rank_score",
        "eval_f1",
        "search_val_f1",
        "eval_accuracy",
        "eval_precision",
        "eval_recall",
        "eval_roc_auc",
        "eval_loss",
        "n_trials",
        "best_trial",
        "best_metric",
        "best_model_checkpoint",
        "search_epochs",
        "full_epochs",
        "train_size",
        "val_size",
        "test_size",
        "model_name",
        "best_params",
        "source_kind",
        "path",
    ]

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows):
    headers = ["Model", "BP", "Algorithm", "Score", "Eval F1", "Search F1", "Source"]
    widths = [12, 5, 10, 10, 10, 10, 22]
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    fmt = "|" + "|".join(f" {{:<{w}}} " for w in widths) + "|"

    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(
            row["model"],
            row["window_bp"],
            row["algorithm"],
            format_score(row["rank_score"]),
            format_score(row["eval_f1"]),
            format_score(row["search_val_f1"]),
            row["source_kind"],
        ))
    print(sep)


def format_score(value):
    number = as_float(value)
    if number is None or math.isnan(number):
        return ""
    return f"{number:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Directory containing opt_<model>_<window>bp result folders.",
    )
    parser.add_argument(
        "--out-csv",
        default=os.path.join("outputs", "final_results.csv"),
        help="CSV summary path.",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Optional JSON summary path.",
    )
    args = parser.parse_args()

    rows = []
    for entry in sorted(os.listdir(args.output_root)):
        opt_dir = os.path.join(args.output_root, entry)
        if os.path.isdir(opt_dir) and OPT_DIR_RE.match(entry):
            rows.extend(collect_opt_dir(opt_dir))

    if not rows:
        print(f"No optimization results found under: {args.output_root}")
        sys.exit(0)

    rows = sort_rows(rows)
    write_csv(args.out_csv, rows)
    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        write_json(args.out_json, rows)

    print_table(rows)
    print(f"\nCSV results saved to: {args.out_csv}")
    if args.out_json:
        print(f"JSON results saved to: {args.out_json}")
    print(f"Total collected runs: {len(rows)}")


if __name__ == "__main__":
    main()
