"""
Computes and stores validation MSE for experiments 2 and 3.

For each result duckdb/CSV, evaluates the stored expressions on the
held-out validation fold (the KFold split that was not used for training)
and adds a `mse_val` column.
"""

import glob
import os
import pathlib
import re
import signal

import duckdb
import numpy as np
import pandas as pd
import pmlb
import ucimlrepo
from shared.dataset_config import DATASETS
from shared.task import lambdify_expression, synthetic_problem
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold
from tqdm import tqdm

PMLB_CACHE_DIRECTORY = "pmlb_cache"

# Experiment configs: dataset name -> number of folds
EXP2_DATASETS = {
    "daily_demand": 30,
    "auto_mpg": 30,
    "california_housing": 30,
    "feynman": 30,
}

EXP3_DATASETS = {
    "1_addition": 9,
    "2_division": 9,
    "3_subtraction": 9,
    "4_multiplication": 9,
    "5_square": 9,
}

# Map paper result dataset names to shared.dataset_config keys
DATASET_KEY_MAP = {
    "daily_demand": "daily_demand",
    "auto_mpg": "auto_mpg",
    "california_housing": "california_housing",
    "feynman": "feynman_I_9_18",
    "1_addition": "1_addition",
    "2_division": "2_division",
    "3_subtraction": "3_subtraction",
    "4_multiplication": "4_multiplication",
    "5_square": "5_square",
    "feynman_I_8_14": "feynman_I_8_14",
    "feynman_I_11_19": "feynman_I_11_19",
    "feynman_I_9_18": "feynman_I_9_18",
    "modular_1": "modular_1",
    "modular_2": "modular_2",
    "modular_3": "modular_3",
    "modular_4": "modular_4",
    "modular_5": "modular_5",
}


def _load_raw_dataset(dataset_key: str) -> tuple[np.ndarray, np.ndarray]:
    """Load raw (X, y) for a dataset using the same method as task.py."""
    cfg = DATASETS[dataset_key]

    match cfg.dataset_type:
        case "pmlb":
            X, y = pmlb.fetch_data(
                cfg.name,
                return_X_y=True,
                local_cache_dir=PMLB_CACHE_DIRECTORY,
            )
        case "sklearn":
            X, y = fetch_california_housing(return_X_y=True)
        case "synthetic":
            kwargs = {}
            if cfg.lb is not None:
                kwargs["lb"] = cfg.lb
            if cfg.ub is not None:
                kwargs["ub"] = cfg.ub
            X, y = synthetic_problem(
                cfg.equation, cfg.observations, noise=0.0, seed=42, **kwargs
            )
        case "uci":
            d = ucimlrepo.fetch_ucirepo(id=int(cfg.name))
            X = d.data.features.to_numpy()
            y = d.data.targets.to_numpy()

    return X, y


def get_validation_data(
    dataset_name: str, num_folds: int, seed: int = 42
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Reproduces the KFold validation splits used during experiment task creation.

    Returns a dict mapping fold index -> (X_val, y_val).
    """
    dataset_key = DATASET_KEY_MAP[dataset_name]
    X, y = _load_raw_dataset(dataset_key)

    if y.ndim > 1:
        y = y.ravel()

    # Reproduce the exact rng sequence from task_factory / create_tasks:
    # test_size=0 for both exp2 and exp3, so no train_test_split call.
    # First rng call is the KFold random_state.
    rng = np.random.default_rng(seed)
    kf_seed = int(rng.integers(2**32 - 1))

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=kf_seed)

    result = {}
    for fold, (_, val_indices) in enumerate(kf.split(X)):
        result[fold] = (X[val_indices], y[val_indices])

    return result


class _EvalTimeout(Exception):
    pass


def _eval_mse(
    expr: str, X_val: np.ndarray, y_val: np.ndarray, timeout: int = 10
) -> float | None:
    """
    Evaluate expression on validation data and return MSE, or None on failure/timeout.

    Uses SIGALRM to cap wall-clock time, guarding against sympy.sympify hanging
    on deeply nested GP expressions.
    """
    expr = expr.strip("'\"")

    def _handler(signum, frame):
        raise _EvalTimeout()

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout)
    try:
        fn = lambdify_expression(expr)
        y_pred = fn(X_val)
        if y_pred is None or np.all(np.isnan(y_pred)):
            return None
        return float(np.mean((y_pred - y_val) ** 2))
    except Exception:
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _add_mse_val_column(conn: duckdb.DuckDBPyConnection) -> None:
    """Add mse_val column to results table if it doesn't exist."""
    cols = {row[0] for row in conn.execute("DESCRIBE results").fetchall()}
    if "mse_val" not in cols:
        conn.execute("ALTER TABLE results ADD COLUMN mse_val DOUBLE")
    if "var_y_val" not in cols:
        conn.execute("ALTER TABLE results ADD COLUMN var_y_val DOUBLE")


def add_mse_val_to_duckdb(
    db_path: str | pathlib.Path,
    dataset_name: str,
    num_folds: int,
    seed: int = 42,
) -> None:
    """
    Adds mse_val and var_y_val columns to a result duckdb file by evaluating
    each stored expression on the corresponding validation fold.
    """
    print(f"  Validating {db_path}...")

    val_data = get_validation_data(dataset_name, num_folds, seed)

    conn = duckdb.connect(str(db_path))
    _add_mse_val_column(conn)

    # Build mapping: (fold, expression) -> mse_val
    rows = conn.execute(
        "SELECT DISTINCT fold, expressions FROM results WHERE expressions IS NOT NULL"
    ).fetchall()

    cache: dict[tuple[int, str], float | None] = {}
    for fold, expr in tqdm(rows, desc="  Evaluating", leave=False):
        key = (fold, expr)
        if key not in cache:
            X_val, y_val = val_data[fold]
            cache[key] = _eval_mse(expr, X_val, y_val)

    # Precompute var_y_val per fold
    var_y_val = {fold: float(np.var(y_val)) for fold, (_, y_val) in val_data.items()}

    # Build pandas DataFrame for bulk update
    mapping_rows = [
        {"fold": fold, "expressions": expr, "mse_val": mse}
        for (fold, expr), mse in cache.items()
    ]
    mapping_df = pd.DataFrame(mapping_rows)
    conn.register("_mse_val_mapping", mapping_df)
    conn.execute("""
        UPDATE results
        SET mse_val = m.mse_val
        FROM _mse_val_mapping m
        WHERE results.fold = m.fold AND results.expressions = m.expressions
    """)

    # Set var_y_val per fold
    var_df = pd.DataFrame([{"fold": f, "var_y_val": v} for f, v in var_y_val.items()])
    conn.register("_var_y_val_mapping", var_df)
    conn.execute("""
        UPDATE results
        SET var_y_val = v.var_y_val
        FROM _var_y_val_mapping v
        WHERE results.fold = v.fold
    """)

    conn.close()


def add_mse_val_to_csv(
    csv_path: str | pathlib.Path,
    dataset_name: str,
    num_folds: int,
    seed: int = 42,
) -> None:
    """
    Adds mse_val column to a result CSV file and overwrites it.
    """
    print(f"  Validating {csv_path}...")

    val_data = get_validation_data(dataset_name, num_folds, seed)

    df = pd.read_csv(csv_path)

    if "mse_val" in df.columns:
        return  # Already done

    cache: dict[tuple[int, str], float | None] = {}

    def _compute(row):
        fold = int(row["fold"])
        expr = str(row["expression"])
        key = (fold, expr)
        if key not in cache:
            X_val, y_val = val_data[fold]
            cache[key] = _eval_mse(expr, X_val, y_val)
        return cache[key]

    df["mse_val"] = df.apply(_compute, axis=1)
    df["var_y_val"] = df["fold"].map(
        {fold: float(np.var(y_val)) for fold, (_, y_val) in val_data.items()}
    )
    df.to_csv(csv_path, index=False)


def validate_experiment_2(dir: pathlib.Path, seed: int = 42) -> None:
    """Add mse_val to all source result files for experiment 2."""
    print("Validating experiment 2...")

    dataset_fold_map = {
        "daily_demand": 30,
        "auto_mpg": 30,
        "california_housing": 30,
        "feynman": 30,
    }

    # Process duckdb files (cpu_results, gpu_results)
    for db_file in glob.glob(f"{dir}/**/*.duckdb", recursive=True):
        if "backup" in db_file.split(os.sep) or "all_results" in db_file:
            continue
        dataset = pathlib.Path(db_file).stem
        if dataset not in dataset_fold_map:
            continue
        add_mse_val_to_duckdb(db_file, dataset, dataset_fold_map[dataset], seed)

    # Process CSV files (evogp, kozax, operon, etc.)
    for csv_file in glob.glob(f"{dir}/**/*.csv", recursive=True):
        if "backup" in csv_file.split(os.sep):
            continue
        # Infer dataset from path: dir/{algorithm}/{dataset}.csv
        parts = csv_file.split(os.sep)
        dataset = parts[-1].split(".")[0]
        if dataset not in dataset_fold_map:
            continue
        add_mse_val_to_csv(csv_file, dataset, dataset_fold_map[dataset], seed)


def validate_experiment_4(dir: pathlib.Path, seed: int = 42) -> None:
    """Add mse_val to all source result duckdb files for experiment 4."""
    print("Validating experiment 4...")

    dataset_fold_map = {
        "modular_1": 9,
        "modular_2": 9,
        "modular_3": 9,
        "modular_4": 9,
        "modular_5": 9,
    }

    for db_file in glob.glob(f"{dir}/**/*.duckdb", recursive=True):
        if "backup" in db_file.split(os.sep) or "all_results" in db_file:
            continue
        # Dataset name is stored inside the duckdb, not the filename
        conn = duckdb.connect(db_file)
        datasets = [
            r[0]
            for r in conn.execute("SELECT DISTINCT dataset FROM results").fetchall()
        ]
        conn.close()
        for dataset in datasets:
            if dataset not in dataset_fold_map:
                continue
            add_mse_val_to_duckdb(db_file, dataset, dataset_fold_map[dataset], seed)


def validate_experiment_5(dir: pathlib.Path, seed: int = 42) -> None:
    """Add mse_val to all source result duckdb files for experiment 5."""
    print("Validating experiment 5...")

    dataset_fold_map = {
        "daily_demand": 30,
        "auto_mpg": 30,
        "california_housing": 30,
        "feynman": 30,
    }

    for db_file in glob.glob(f"{dir}/**/*.duckdb", recursive=True):
        if "backup" in db_file.split(os.sep) or "all_results" in db_file:
            continue
        dataset = pathlib.Path(db_file).stem
        if dataset not in dataset_fold_map:
            continue
        add_mse_val_to_duckdb(db_file, dataset, dataset_fold_map[dataset], seed)


def validate_experiment_3(dir: pathlib.Path, seed: int = 42) -> None:
    """Add mse_val to all source result duckdb files for experiment 3."""
    print("Validating experiment 3...")

    dataset_fold_map = {
        "1_addition": 9,
        "2_division": 9,
        "3_subtraction": 9,
        "4_multiplication": 9,
        "5_square": 9,
    }

    for db_file in glob.glob(f"{dir}/**/*.duckdb", recursive=True):
        if "backup" in db_file.split(os.sep) or "all_results" in db_file:
            continue
        dataset = pathlib.Path(db_file).stem
        if dataset not in dataset_fold_map:
            continue
        add_mse_val_to_duckdb(db_file, dataset, dataset_fold_map[dataset], seed)
