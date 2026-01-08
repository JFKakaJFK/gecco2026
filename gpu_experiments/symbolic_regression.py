import os
import re
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pmlb
import sympy as sym
from pygom import KernelVersion
from sklearn.model_selection import KFold, train_test_split

from src.config import CPU_GPU_CONFIG, KERNELS_CONFIG, TEST_CONFIG, ExperimentConfig
from src.db import create_db
from src.plots import plot
from src.run import run_cpu_tasks, run_gpu_tasks

REPEATS_TOTAL = 15
NUM_FOLDS = 5

REPEATS_PER_FOLD = REPEATS_TOTAL // NUM_FOLDS

PMLB_CACHE_DIR = "pmlb_cache"


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
    assert ub > lb, "Invalid initialisation bounds"

    rng = np.random.Generator(np.random.Philox(seed=seed))

    num_inputs = max([int(x) + 1 for x in re.findall(r"x(\d+)", expr)])

    X = rng.random(size=(size, num_inputs)) * (ub - lb) + lb
    y = lambdify_expression(expr)(X)

    if noise > 0:
        y += rng.standard_normal(size) * noise

    return X, y


def problems(
    rng: Generator,
    output_directory: Path,
    config: ExperimentConfig,
    dry_run: bool = False,
):
    data_dir = output_directory / "data"

    if not dry_run:
        os.makedirs(data_dir, exist_ok=True)

    for problem in config.problems:
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

        kf = KFold(
            n_splits=NUM_FOLDS,
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

            for obs in config.num_observations:
                # Skip datasets that do not have enough observations
                # 0.75 = due to train_test_split
                # (1 - 1 / NUM_FOLDS) = due to KFOLD
                if obs > problem.observations * 0.75 * (1 - 1 / NUM_FOLDS):
                    continue

                for feat in config.num_features:
                    # Skip datasets that do not have enough features
                    if feat > problem.features:
                        continue

                    for pop in config.population_sizes:
                        # For larger population sizes, skip large datasets
                        if pop >= 4096 and obs >= 1e5:
                            continue

                        # (branching_factor, depth)
                        for template in config.templates:
                            for operator in config.operator_sets:
                                seed = int(rng.integers(2**32 - 1))

                                yield {
                                    "problem": problem.name,
                                    "fold": fold,
                                    "num_observations": obs,
                                    "num_features": feat,
                                    "population_size": pop,
                                    "template": template,
                                    "operators": operator.operators,
                                    "operator_set": operator.name,
                                    "X_path": X_path,
                                    "y_path": y_path,
                                    "X_test_path": X_test_path,
                                    "y_test_path": y_test_path,
                                    "seed": seed,
                                }


def cpu_jobs(problems):
    yield from problems


def gpu_jobs(problems, kernels: tuple[KernelVersion, ...]):
    for info in problems:
        for kernel in kernels:
            yield (info, kernel)


def all_jobs(output_directory: Path, config: ExperimentConfig, dry_run: bool = False):
    # Run CPU jobs
    if config.cpu.enabled:
        print("Starting CPU tasks...")
        run_cpu_tasks(
            output_directory,
            cpu_jobs(
                problems(
                    np.random.default_rng(seed=42), output_directory, config, dry_run
                )
            ),
            num_repeats=REPEATS_PER_FOLD,
            dry_run=dry_run,
        )
        print("Finished CPU tasks...")

    # Run GPU jobs
    print("Starting GPU tasks...")
    run_gpu_tasks(
        output_directory,
        gpu_jobs(
            problems(np.random.default_rng(seed=42), output_directory, config, dry_run),
            config.gpu.kernels,
        ),
        num_repeats=REPEATS_PER_FOLD,
        dry_run=dry_run,
    )
    print("Finished GPU tasks")


def run_experiment(dir: Path, config: ExperimentConfig, dry_run=False):
    print(f"\n\nStarting {config.name} experiment...")

    output_directory = dir / config.name

    all_jobs(output_directory, config, dry_run)

    if not dry_run:
        create_db(output_directory)
        plot(output_directory)

    print(f"Finished {config.name} experiment!\n\n")


def main():
    run_date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    # run_date = "2026-01-08_12:39:02"
    output_directory = Path("results") / run_date

    # run_experiment(output_directory, TEST_CONFIG, dry_run=False)
    run_experiment(output_directory, CPU_GPU_CONFIG, dry_run=False)
    run_experiment(output_directory, KERNELS_CONFIG, dry_run=False)


if __name__ == "__main__":
    main()
