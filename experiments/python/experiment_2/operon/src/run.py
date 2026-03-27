import csv
import os
import re
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Never

import numpy as np
import sympy as sym
from pyoperon.sklearn import SymbolicRegressor
from shared.task import Task, TaskGenerator
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from tqdm import tqdm

JobQueue = list[tuple[Callable[[Task, Path], None], list[Never], dict[str, Task]]]


def lambdify_expression(e):
    """Converts a `sympy` compatible expression string into a function accepting a dataset `X`."""
    e = str(e)

    symbols = {x: sym.Symbol(x) for x in re.findall(r"([Xx]\d+)", e)}
    expr = sym.sympify(e, locals=symbols)
    f = sym.lambdify(symbols.values(), expr, modules=[{"clip": np.clip}, "numpy"])

    def fn(X: np.ndarray):
        try:
            return f(*[X[:, int(s[1:])] for s in symbols.keys()])
        except Exception as e:
            print(e)
            return np.repeat(float("nan"), X.shape[0])

    return fn


def run_one_task(task: Task) -> dict:
    X_train = np.load(task["X_path"].absolute())
    y_train = np.load(task["y_path"].absolute())

    obs = task["num_observations"]
    feat = task["num_features"]

    assert X_train.shape[0] >= obs, (
        f"Requested obs={obs}, but fold has only {X_train.shape[0]}"
    )
    assert X_train.shape[1] >= feat, (
        f"Requested feat={feat}, but fold has only {X_train.shape[1]}"
    )

    X_train = X_train[:obs, :feat]
    y_train = y_train[:obs].reshape(-1)

    if np.isnan(X_train).any():
        imputer = IterativeImputer(
            max_iter=10,
            random_state=task["seed"],
            sample_posterior=True,
        )
        X_train = imputer.fit_transform(X_train)

    temp = float(np.nanmax(np.abs(y_train)))
    const_range = (-temp, temp)

    if task["population_size"] is None:
        raise ValueError("Population size cannot be None")

    reg = SymbolicRegressor(
        allowed_symbols="constant,variable,add,sub,mul,div,sin,cos,square,sqrt,exp,log,pow",
        crossover_probability=1.0,
        crossover_internal_probability=0.9,
        mutation_probability=0.1,
        objectives=["mse"],
        max_length=task["max_tree_len"],
        max_depth=task["depth"],
        initialization_max_length=task["max_tree_len"],
        initialization_max_depth=task["depth"],
        population_size=task["population_size"],
        generations=1_000_000_000,
        max_evaluations=1_000_000_000,
        tournament_size=20,
        model_selection_criterion="mean_squared_error",
        max_time=task["max_duration"] * 60,  # assumed to be seconds, sp multiply by 60
        random_state=task["seed"],
    )

    start_time = time.perf_counter()
    reg.fit(X_train, y_train)
    elapsed_time = time.perf_counter() - start_time

    best = min(reg.pareto_front_, key=lambda s: s["objective_values"][0])
    old_mse = best["objective_values"][0]
    expr = reg.get_model_string(
        best["tree"], 12, [f"x{i}" for i in range(task["num_features"])]
    )

    y_pred = lambdify_expression(expr)(X_train)
    mse = np.mean((y_pred - y_train) ** 2)

    return {
        "total_time_seconds": elapsed_time,
        "expression": f"'{expr}'",
        "mse": float(mse),
        "old_mse": float(old_mse),
        "evaluations": reg.stats_["evaluation_count"],
        "dataset": task["dataset"],
        "fold": task["fold"],
        "num_observations": task["num_observations"],
        "num_features": task["num_features"],
        "population_size": task["population_size"],
        "operator_set": task["operator_set"],
        "template_depth": task["depth"],
        "iteration": task["iteration"],
        "seed": task["seed"],
    }


def run_cpu_tasks(
    tasks: TaskGenerator,
    output_directory: Path,
    max_workers: int | None = 30,
    dry_run: bool = False,
) -> None:
    if not dry_run:
        os.makedirs(output_directory, exist_ok=True)

    results_path = output_directory / "results.csv"
    fieldnames: list[str] = [
        "total_time_seconds",
        "expression",
        "mse",
        "old_mse",
        "evaluations",
        "dataset",
        "fold",
        "num_observations",
        "num_features",
        "population_size",
        "operator_set",
        "template_depth",
        "iteration",
        "seed",
    ]

    jobs: JobQueue = []

    for task in tasks:
        jobs.append((run_one_task, [], {"task": task}))

    n_jobs = len(jobs)

    if dry_run:
        print(f"Total CPU tasks: {n_jobs}")
        return

    with open(results_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        with (
            ProcessPoolExecutor(max_workers=max_workers) as pool,
            tqdm(total=n_jobs, leave=False, ascii=True) as progress,
        ):
            futures = {pool.submit(fn, *args, **kwargs) for fn, args, kwargs in jobs}

            try:
                for f in as_completed(futures):
                    try:
                        result = f.result()
                        writer.writerow(result)
                        csvfile.flush()

                    except Exception as e:
                        print(e)

                    finally:
                        progress.update()

            except KeyboardInterrupt:
                pool.shutdown(wait=False, cancel_futures=True)

                progress.close()

                return
