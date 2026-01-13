import os
import re
from collections.abc import Generator
from pathlib import Path
from typing import TypedDict
from urllib.parse import quote

import numpy as np
import pmlb
import sympy as sym
from pygom import KernelVersion
from sklearn.model_selection import KFold, train_test_split

from src.config import ExperimentConfig

# REPEATS_TOTAL = 15
# NUM_FOLDS = 5

# REPEATS_PER_FOLD = REPEATS_TOTAL // NUM_FOLDS

PMLB_CACHE_DIR = "pmlb_cache"


class Task(TypedDict):
    problem: str
    accelerated: bool
    kernel: KernelVersion | None
    population_size: int
    num_observations: int
    num_features: int
    branching_factor: int
    depth: int
    operator_set: str
    target_objectives: list[float] | None
    fold: int
    num_iterations: int
    X_path: Path
    y_path: Path
    X_test_path: Path
    y_test_path: Path
    seed: int


def lambdify_expression(e: str | sym.Expr):
    """
    Converts a `sympy` compatible expression string into a function
    accepting a dataset `X`.
    """
    e = str(e)

    symbols = {x: sym.Symbol(x) for x in re.findall(r"(x\d+)", e)}
    expr = sym.sympify(e, locals=symbols)
    f = sym.lambdify(symbols.values(), expr, modules=[{"clip": np.clip}, "numpy"])

    def fn(X: np.ndarray):
        try:
            return f(*[X[:, int(s[1:])] for s in symbols])
        except Exception as e:
            print(e)
            return np.repeat(float("nan"), X.shape[0])

    return fn


def synthetic_problem(
    expr: str,
    size: int = 1000,
    lb: float = -10.0,
    ub: float = 10.0,
    noise: float = 0.01,
    seed: int | None = None,
):
    """
    Creates a synthetic problem by sampling a random dataset,
    applying the function and possibly adding noise.
    """
    assert ub > lb, "Invalid initialization bounds"

    rng = np.random.Generator(np.random.Philox(seed=seed))

    num_inputs = max([int(x) + 1 for x in re.findall(r"x(\d+)", expr)])

    X = rng.random(size=(size, num_inputs)) * (ub - lb) + lb
    y = lambdify_expression(expr)(X)

    if noise > 0:
        y += rng.standard_normal(size) * noise

    return X, y


def problems(
    rng: np.random.Generator,
    output_dir: Path,
    cfg: ExperimentConfig,
    dry_run: bool = False,
) -> Generator[Task]:
    data_dir = output_dir / "data"

    if not dry_run:
        os.makedirs(data_dir, exist_ok=True)

    for problem in cfg.iter_problems():
        eq_dir = data_dir / quote(problem.name, safe=" ")

        if not dry_run:
            eq_dir.mkdir(parents=True, exist_ok=True)

        match problem.type:
            case "synthetic":
                X, y = synthetic_problem(
                    problem.name,
                    problem.observations,
                    noise=0.0,
                    seed=rng.integers(2**32 - 1),
                )
            case "pmlb":
                X, y = pmlb.fetch_data(
                    problem.name,
                    return_X_y=True,
                    local_cache_dir=PMLB_CACHE_DIR,
                )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=rng.integers(2**32 - 1)
        )

        # the "task" needs to be transferrable across processes
        # -> save the data as .csv and load it there again
        X_test_path, y_test_path = (
            eq_dir / "X_test.npy",
            eq_dir / "y_test.npy",
        )

        if not dry_run:
            np.save(X_test_path, X_test, allow_pickle=False)
            np.save(y_test_path, y_test.reshape(-1, 1), allow_pickle=False)

        target_objectives = (
            cfg.get_target_objectives(problem) if cfg.use_target else None
        )

        kf = KFold(
            n_splits=cfg.num_folds,
            shuffle=True,
            random_state=rng.integers(2**32 - 1),
        )
        for fold, (train_indices, _) in enumerate(kf.split(X_train)):
            X_fold, y_fold = (
                X_train[train_indices, :],
                y_train[train_indices].reshape(-1, 1),
            )

            X_path, y_path = (
                eq_dir / f"X{fold:03d}.npy",
                eq_dir / f"y{fold:03d}.npy",
            )

            if not dry_run:
                np.save(X_path, X_fold, allow_pickle=False)
                np.save(y_path, y_fold, allow_pickle=False)

            for obs in cfg.iter_observations():
                # Skip datasets that do not have enough observations
                # 0.75 = due to train_test_split
                # (1 - 1 / num_folds) = due to k_fold
                if obs > problem.observations * 0.75 * (1 - 1 / cfg.num_folds):
                    continue

                for feat in cfg.iter_features():
                    # Skip datasets that do not have enough features
                    if feat > problem.features:
                        continue

                    for pop in cfg.iter_population():
                        # For larger population sizes, skip large datasets
                        if pop >= 4096 and obs >= 1e5:
                            continue

                        # (branching_factor, depth)
                        for template in cfg.iter_templates():
                            for operator_set in cfg.iter_operators():
                                seed = int(rng.integers(2**32 - 1))

                                task: Task = {
                                    "problem": problem.name,
                                    "accelerated": False,
                                    "kernel": None,
                                    "population_size": pop,
                                    "num_observations": obs,
                                    "num_features": feat,
                                    "branching_factor": template.branching_factor,
                                    "depth": template.depth,
                                    "operator_set": operator_set,
                                    "target_objectives": target_objectives,
                                    "fold": fold,
                                    "num_iterations": cfg.num_iterations,
                                    "X_path": X_path,
                                    "y_path": y_path,
                                    "X_test_path": X_test_path,
                                    "y_test_path": y_test_path,
                                    "seed": seed,
                                }

                                yield task
