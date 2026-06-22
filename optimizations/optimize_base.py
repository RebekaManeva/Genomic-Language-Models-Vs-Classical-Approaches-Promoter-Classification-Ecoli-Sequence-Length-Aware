"""
optimize_base.py
================
Shared optimization algorithms and utilities for DNA promoter classification.
Used by both optimize_hf_seqclf_nlp.py and optimize_dnabert2_nlp.py.

Algorithms (--algorithm):
    rs    Random Search
    ts    Tree-structured Parzen Estimator / TPE   (Optuna)
    bayes Bayesian TPE with explicit sampler        (Optuna, same as ts but named clearly)
    ga    Genetic Algorithm                         (pure Python EA)
    hc    Hill Climbing                             (local perturbation)
    sa    Simulated Annealing                       (Metropolis acceptance)
    sopt  Sequential GP-BO                          (scikit-optimize)

Grid Search (gs) is intentionally excluded.

Each algorithm shares the same interface:
    run_<algo>(objective_fn, space, n_trials, seed) -> (best_cfg, best_score, all_results)

Where:
    objective_fn : callable(cfg: dict) -> float   (higher = better)
    space        : dict[str, list]                 (categorical values per key)
    n_trials     : int
    seed         : int
    returns      : (best_cfg: dict, best_score: float, all_results: list[dict])
"""

import copy
import math
import random
import logging
from collections.abc import Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ── optional heavy imports ────────────────────────────────────────────────────
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    from skopt import gp_minimize
    from skopt.space import Categorical
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _random_cfg(space: dict, rng: random.Random) -> dict:
    return {k: rng.choice(v) for k, v in space.items()}


def json_safe(value):
    """Convert NumPy/scikit-optimize values to plain JSON-safe Python types."""
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_safe(v) for v in value]
    return value


def _record(trial_idx: int, cfg: dict, score: float) -> dict:
    return {
        "trial": int(trial_idx),
        "config": json_safe(cfg),
        "val_f1": float(score),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Random Search
# ─────────────────────────────────────────────────────────────────────────────

def run_rs(objective_fn, space: dict, n_trials: int, seed: int = 42):
    """
    Random Search — uniform sampling with no surrogate model.
    Simple baseline; surprisingly competitive with small budgets.
    """
    rng = random.Random(seed)
    best_score, best_cfg = float("-inf"), None
    all_results = []

    for i in range(n_trials):
        cfg   = _random_cfg(space, rng)
        score = objective_fn(cfg)
        all_results.append(_record(i, cfg, score))
        if score > best_score:
            best_score, best_cfg = score, cfg
        logger.info(f"[RS] trial {i:04d}  val_f1={score:.4f}  cfg={cfg}")

    return best_cfg, best_score, all_results


# ─────────────────────────────────────────────────────────────────────────────
# Hill Climbing
# ─────────────────────────────────────────────────────────────────────────────

def run_hc(objective_fn, space: dict, n_trials: int, seed: int = 42):
    """
    Hill Climbing — start from a random config, perturb one dimension per step.
    Accepts any move that is equal or better (greedy). Can get stuck in local optima
    but is cheap and works well for small spaces.
    """
    rng = random.Random(seed)
    current_cfg   = _random_cfg(space, rng)
    current_score = objective_fn(current_cfg)
    best_score, best_cfg = current_score, current_cfg
    all_results = [_record(0, current_cfg, current_score)]

    logger.info(f"[HC] trial 0000  val_f1={current_score:.4f}  cfg={current_cfg}")

    for i in range(1, n_trials):
        candidate = copy.copy(current_cfg)
        key = rng.choice(list(space.keys()))
        # Ensure the new value actually differs (avoids wasting a trial)
        choices = [v for v in space[key] if v != candidate[key]]
        candidate[key] = rng.choice(choices) if choices else candidate[key]

        score = objective_fn(candidate)
        all_results.append(_record(i, candidate, score))

        if score >= current_score:
            current_cfg, current_score = candidate, score
            if score > best_score:
                best_score, best_cfg = score, candidate

        logger.info(f"[HC] trial {i:04d}  val_f1={score:.4f}  cfg={candidate}")

    return best_cfg, best_score, all_results


# ─────────────────────────────────────────────────────────────────────────────
# Simulated Annealing
# ─────────────────────────────────────────────────────────────────────────────

def run_sa(objective_fn, space: dict, n_trials: int, seed: int = 42):
    """
    Simulated Annealing with Metropolis acceptance criterion.
    Temperature follows geometric cooling from T_start=1.0 to T_end=0.01.
    Worse moves are accepted with probability exp(delta/T), allowing escape
    from local optima early in the search.
    """
    rng = random.Random(seed)
    current_cfg   = _random_cfg(space, rng)
    current_score = objective_fn(current_cfg)
    best_score, best_cfg = current_score, current_cfg
    all_results = [_record(0, current_cfg, current_score)]

    logger.info(f"[SA] trial 0000  val_f1={current_score:.4f}  cfg={current_cfg}")

    T_start, T_end = 1.0, 0.01
    for i in range(1, n_trials):
        T = T_start * (T_end / T_start) ** (i / max(n_trials - 1, 1))

        candidate = copy.copy(current_cfg)
        key = rng.choice(list(space.keys()))
        choices = [v for v in space[key] if v != candidate[key]]
        candidate[key] = rng.choice(choices) if choices else candidate[key]

        score = objective_fn(candidate)
        all_results.append(_record(i, candidate, score))

        delta = score - current_score
        if delta > 0 or rng.random() < math.exp(delta / T):
            current_cfg, current_score = candidate, score

        if score > best_score:
            best_score, best_cfg = score, candidate

        logger.info(f"[SA] trial {i:04d}  T={T:.4f}  val_f1={score:.4f}  cfg={candidate}")

    return best_cfg, best_score, all_results


# ─────────────────────────────────────────────────────────────────────────────
# Genetic Algorithm
# ─────────────────────────────────────────────────────────────────────────────

def run_ga(objective_fn, space: dict, n_trials: int, seed: int = 42):
    """
    Genetic Algorithm with tournament selection, uniform crossover, and mutation.
    Population size = max(4, n_trials // 5).
    Mutation rate = 0.2 per gene.
    Uses worst-replacement strategy: child replaces worst member if it improves.
    """
    rng = random.Random(seed)
    pop_size  = max(4, n_trials // 5)
    trial_idx = 0
    all_results = []

    # ── Initial population ───────────────────────────────────────────────────
    population  = [_random_cfg(space, rng) for _ in range(pop_size)]
    pop_scores  = []
    for cfg in population:
        if trial_idx >= n_trials:
            break
        score = objective_fn(cfg)
        pop_scores.append(score)
        all_results.append(_record(trial_idx, cfg, score))
        logger.info(f"[GA] trial {trial_idx:04d}  val_f1={score:.4f}  cfg={cfg}")
        trial_idx += 1

    # ── Evolution loop ───────────────────────────────────────────────────────
    while trial_idx < n_trials:
        def tournament():
            a = rng.randrange(len(population))
            b = rng.randrange(len(population))
            return population[a] if pop_scores[a] >= pop_scores[b] else population[b]

        p1, p2 = tournament(), tournament()
        child  = {}
        for key in space:
            # Uniform crossover
            child[key] = p1[key] if rng.random() < 0.5 else p2[key]
            # Mutation
            if rng.random() < 0.2:
                child[key] = rng.choice(space[key])

        score = objective_fn(child)
        all_results.append(_record(trial_idx, child, score))
        logger.info(f"[GA] trial {trial_idx:04d}  val_f1={score:.4f}  cfg={child}")
        trial_idx += 1

        # Replace worst if child is better
        worst_idx = pop_scores.index(min(pop_scores))
        if score > pop_scores[worst_idx]:
            population[worst_idx]  = child
            pop_scores[worst_idx]  = score

    best_score, best_cfg = max(
        ((r["val_f1"], r["config"]) for r in all_results), key=lambda x: x[0]
    )
    return best_cfg, best_score, all_results


# ─────────────────────────────────────────────────────────────────────────────
# Optuna (TPE / Bayesian)
# ─────────────────────────────────────────────────────────────────────────────

def run_optuna(
    objective_fn,
    space: dict,
    n_trials: int,
    seed: int = 42,
    sampler_name: str = "tpe",
    study_name: str = "opt_study",
    storage: str = None,
    direction: str = "maximize",
):
    """
    Generic Optuna runner.

    sampler_name: "tpe"    — Tree Parzen Estimator (default, recommended)
                  "random" — Random sampler (for ablation)
                  "cmaes"  — CMA-ES (works best for continuous spaces)

    Note: all parameters are treated as categorical here since the search space
    is defined as discrete lists. If you extend spaces with continuous ranges,
    switch to suggest_float / suggest_int with the log= flag where appropriate.
    """
    if not HAS_OPTUNA:
        raise ImportError("optuna not installed. Run: pip install optuna")

    sampler_map = {
        "tpe":    optuna.samplers.TPESampler(seed=seed),
        "random": optuna.samplers.RandomSampler(seed=seed),
        "cmaes":  optuna.samplers.CmaEsSampler(seed=seed),
    }
    sampler = sampler_map.get(sampler_name, optuna.samplers.TPESampler(seed=seed))

    trial_counter = {"n": 0}
    all_results   = []

    def _objective(trial: "optuna.Trial") -> float:
        cfg = {}
        for key, values in space.items():
            # Use typed suggest calls to let Optuna model the space properly
            if all(isinstance(v, bool) for v in values):
                cfg[key] = trial.suggest_categorical(key, values)
            elif all(isinstance(v, int) and not isinstance(v, bool) for v in values):
                cfg[key] = trial.suggest_categorical(key, values)
            elif all(isinstance(v, float) for v in values):
                cfg[key] = trial.suggest_categorical(key, values)
            else:
                cfg[key] = trial.suggest_categorical(key, values)

        idx   = trial_counter["n"]
        score = objective_fn(cfg)
        trial_counter["n"] += 1
        all_results.append(_record(idx, cfg, score))
        trial.set_user_attr("val_f1", score)
        logger.info(f"[Optuna/{sampler_name}] trial {idx:04d}  val_f1={score:.4f}  cfg={cfg}")
        return score

    study = optuna.create_study(
        direction=direction,
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=(storage is not None),
    )
    study.optimize(_objective, n_trials=n_trials)

    best_cfg   = study.best_params
    best_score = study.best_value
    return best_cfg, best_score, all_results


def run_ts(objective_fn, space, n_trials, seed=42, **kw):
    """Tree-structured Parzen Estimator (TPE) via Optuna."""
    return run_optuna(objective_fn, space, n_trials, seed,
                      sampler_name="tpe", **kw)


def run_bayes(objective_fn, space, n_trials, seed=42, **kw):
    """
    Bayesian optimisation via Optuna CMA-ES sampler.
    CMA-ES adapts a covariance matrix over the search space, making it
    more effective than vanilla TPE on larger continuous-like spaces.
    Falls back to TPE if CMA-ES is unavailable.
    """
    try:
        return run_optuna(objective_fn, space, n_trials, seed,
                          sampler_name="cmaes", **kw)
    except Exception:
        logger.warning("CMA-ES unavailable, falling back to TPE for 'bayes'.")
        return run_optuna(objective_fn, space, n_trials, seed,
                          sampler_name="tpe", **kw)


# ─────────────────────────────────────────────────────────────────────────────
# scikit-optimize (GP-BO)
# ─────────────────────────────────────────────────────────────────────────────

def run_sopt(objective_fn, space: dict, n_trials: int, seed: int = 42, **kw):
    """
    Sequential model-based optimisation using a Gaussian Process surrogate
    (scikit-optimize / skopt). Treats every dimension as Categorical.

    GP-BO is the most sample-efficient algorithm here for small budgets
    (< 50 trials), but is also the slowest per-iteration due to GP fitting.
    """
    if not HAS_SKOPT:
        raise ImportError("scikit-optimize not installed. Run: pip install scikit-optimize")

    if n_trials < 5:
        logger.warning(
            "skopt gp_minimize requires at least 5 calls; "
            "using random search for n_trials=%d.",
            n_trials,
        )
        return run_rs(objective_fn, space, n_trials, seed)

    keys       = list(space.keys())
    dimensions = [Categorical(v, name=k) for k, v in space.items()]

    trial_counter = {"n": 0}
    all_results   = []

    def _objective(params):
        cfg   = dict(zip(keys, params))
        idx   = trial_counter["n"]
        score = objective_fn(cfg)
        trial_counter["n"] += 1
        all_results.append(_record(idx, cfg, score))
        logger.info(f"[SOpt] trial {idx:04d}  val_f1={score:.4f}  cfg={cfg}")
        return -score  # skopt minimises

    result = gp_minimize(
        _objective,
        dimensions,
        n_calls=n_trials,
        n_initial_points=max(5, n_trials // 5),
        random_state=seed,
    )

    best_cfg   = dict(zip(keys, result.x))
    best_score = -result.fun
    return best_cfg, best_score, all_results


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm registry
# ─────────────────────────────────────────────────────────────────────────────

ALGO_MAP = {
    "rs":    ("Random Search",                run_rs),
    "ts":    ("Tree Parzen Estimator (TPE)",  run_ts),
    "bayes": ("Bayesian (CMA-ES / TPE)",      run_bayes),
    "ga":    ("Genetic Algorithm",            run_ga),
    "hc":    ("Hill Climbing",                run_hc),
    "sa":    ("Simulated Annealing",          run_sa),
    "sopt":  ("Sequential GP-BO (skopt)",     run_sopt),
}

ALGO_CHOICES = list(ALGO_MAP.keys())


def dispatch(algorithm: str, objective_fn, space: dict, n_trials: int, seed: int = 42, **kw):
    """
    Run the chosen algorithm and return (best_cfg, best_score, all_results).

    Extra kwargs (e.g. study_name, storage) are forwarded to Optuna-based
    algorithms; others silently ignore them.
    """
    if algorithm not in ALGO_MAP:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from: {ALGO_CHOICES}")

    algo_name, algo_fn = ALGO_MAP[algorithm]
    logger.info(f"Running algorithm: {algo_name} for {n_trials} trials")

    # Forward kwargs only to functions that accept them
    import inspect
    sig    = inspect.signature(algo_fn)
    params = sig.parameters
    safe_kw = {k: v for k, v in kw.items() if k in params}

    return algo_fn(objective_fn, space, n_trials, seed, **safe_kw)
