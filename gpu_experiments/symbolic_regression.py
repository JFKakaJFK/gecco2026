import os
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pmlb
import sympy as sym
from pygom import KernelVersion
from sklearn.model_selection import KFold, train_test_split

from src.db import create_db
from src.run import run_cpu_tasks, run_gpu_tasks

REPEATS_TOTAL = 15
NUM_FOLDS = 5

REPEATS_PER_FOLD = REPEATS_TOTAL // NUM_FOLDS

PMLB_CACHE_DIR = "pmlb_cache"


def lambdify_expression(e: str | sym.Expr):
    """Converts a `sympy` compatible expression string into a function accepting a dataset `X`."""
    e = str(e)

    symbols = {x: sym.Symbol(x) for x in re.findall(r"(x\d+)", e)}
    expr = sym.sympify(e, locals=symbols)
    f = sym.lambdify(symbols.values(), expr, modules=[{"clip": np.clip}, "numpy"])

    def fn(X: np.ndarray):
        try:
            return f(*[X[:, int(s[1:])] for s in symbols.keys()])
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
    """Creates a synthetic problem by sampling a random dataset, applying the function and possibly adding noise."""
    assert ub > lb, "Invalid initialisation bounds"

    rng = np.random.Generator(np.random.Philox(seed=seed))

    num_inputs = max([int(x) + 1 for x in re.findall(r"x(\d+)", expr)])

    X = rng.random(size=(size, num_inputs)) * (ub - lb) + lb
    y = lambdify_expression(expr)(X)

    if noise > 0:
        y += rng.standard_normal(size) * noise

    return X, y


def problems(rng, output_directory):
    data_dir = output_directory / "data"
    os.makedirs(data_dir, exist_ok=True)

    problems: dict[str, dict[str, int | str]] = {
        # Synthetic problems
        "sin(3.1 * x0 + 2.7)": {
            "type": "synthetic",
            "observations": np.nan,
            "features": 1,
        },
        # PMLB datasets
        "nikuradse_2": {"type": "pmlb", "observations": 362, "features": 1},
        "feynman_I_6_2a": {"type": "pmlb", "observations": 100_000, "features": 1},
        "542_pollution": {"type": "pmlb", "observations": 60, "features": 15},
        "503_wind": {"type": "pmlb", "observations": 6574, "features": 14},
        "1191_BNG_pbc": {"type": "pmlb", "observations": 1_000_000, "features": 18},
        "505_tecator": {"type": "pmlb", "observations": 240, "features": 124},
    }

    population_sizes: list[int] = [256, 512, 1024, 2048, 4096, 8192]

    num_observations: list[int] = [10, 100, 1_000, 10_000, 100_000, 1_000_000]

    for problem, stats in problems.items():
        eq_dir = data_dir / quote(problem, safe=" ")
        eq_dir.mkdir(parents=True, exist_ok=True)

        for obs in num_observations:
            match stats["type"]:
                case "synthetic":
                    X, y = synthetic_problem(
                        problem, obs, noise=0.0, seed=rng.integers(2**32 - 1)
                    )
                case "pmlb":
                    X, y = pmlb.fetch_data(
                        problem, return_X_y=True, local_cache_dir=PMLB_CACHE_DIR
                    )
                    X = X[:obs]
                    y = y[:obs]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=rng.integers(2**32 - 1)
            )

            # the "task" needs to be transferrable across processess -> save the data as .csv and load it there again
            X_test_path, y_test_path = (
                eq_dir / "X_test.npy",
                eq_dir / "y_test.npy",
            )
            np.save(X_test_path, X_test, allow_pickle=False)
            np.save(y_test_path, y_test.reshape(-1, 1), allow_pickle=False)

            kf = KFold(
                n_splits=NUM_FOLDS, shuffle=True, random_state=rng.integers(2**32 - 1)
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
                np.save(X_path, X_fold, allow_pickle=False)
                np.save(y_path, y_fold, allow_pickle=False)

                for pop in population_sizes:
                    seed = int(rng.integers(2**32 - 1))

                    yield {
                        "problem": problem,
                        "fold": fold,
                        "num_observations": obs,
                        "population_size": pop,
                        "X_path": X_path,
                        "y_path": y_path,
                        "X_test_path": X_test_path,
                        "y_test_path": y_test_path,
                        "seed": seed,
                    }


def cpu_jobs(problems):
    for info in problems:
        yield info


def gpu_jobs(problems, include_kernels):
    kernel_versions: list[KernelVersion] = [
        KernelVersion.baseline,
        KernelVersion.restrict,
        KernelVersion.shared_memory,
        KernelVersion.block_reduce,
        KernelVersion.single_kernel,
        KernelVersion.single_kernel_fmaf,
    ]

    kernels_to_use = (
        kernel_versions if include_kernels else [KernelVersion.single_kernel]
    )

    for info in problems:
        for kernel in kernels_to_use:
            yield (info, kernel)


def all_jobs(output_directory, include_cpu=True, include_kernels=False):
    # Run CPU jobs
    if include_cpu:
        run_cpu_tasks(
            output_directory,
            cpu_jobs(problems(np.random.default_rng(seed=42), output_directory)),
            num_repeats=REPEATS_PER_FOLD,
        )

    # Run GPU jobs
    run_gpu_tasks(
        output_directory,
        gpu_jobs(
            problems(np.random.default_rng(seed=42), output_directory), include_kernels
        ),
        num_repeats=REPEATS_PER_FOLD,
    )


def main():
    run_date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    output_directory = Path("results") / run_date

    all_jobs(output_directory, include_cpu=True, include_kernels=False)

    create_db(output_directory)


if __name__ == "__main__":
    main()
