import csv
import os
import re
import subprocess
import time
from collections.abc import Callable
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Never

import numpy as np
import sympy as sym
import torch
from evogp.algorithm import (
    DefaultCrossover,
    DefaultMutation,
    GeneticProgramming,
    TournamentSelection,
)
from evogp.pipeline import StandardPipeline
from evogp.problem import SymbolicRegression
from evogp.tree import Forest, GenerateDescriptor
from shared.experiment_config import OPERATOR_SETS
from shared.task import Task, TaskGenerator
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from tqdm import tqdm

JobQueue = list[tuple[Callable[[Task, Path], None], list[Never], dict[str, Task]]]

torch.random.manual_seed(42)
torch.cuda.manual_seed(42)


def lambdify_expression(e):
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


# Cursed, but works
def get_num_cuda_devices() -> int:
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)

        return len(result.stdout.strip().splitlines())
    except Exception as e:
        raise Exception(f"No CUDA devices could be detected: {e}") from e


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
    y_train = y_train[:obs]

    if np.isnan(X_train).any():
        imputer = IterativeImputer(
            max_iter=10,
            random_state=task["seed"],
            sample_posterior=True,
        )
        X_train = imputer.fit_transform(X_train)

    X = torch.tensor(X_train, dtype=torch.float32, device="cuda").contiguous()
    y = torch.tensor(y_train, dtype=torch.float32, device="cuda").contiguous()

    temp = float(np.nanmax(np.abs(y_train)))
    const_range = (-temp, temp)

    if task["population_size"] is None:
        raise ValueError("Population size cannot be None")

    problem = SymbolicRegression(datapoints=X, labels=y)

    descriptor = GenerateDescriptor(
        max_tree_len=task["max_tree_len"],
        input_len=problem.problem_dim,
        output_len=problem.solution_dim,
        using_funcs=OPERATOR_SETS[task["operator_set"]].split(","),
        max_layer_cnt=task["depth"] + 1,  # different definition compared to GP-GOMEA
        const_range=const_range,
        sample_cnt=task["max_tree_len"],
    )

    algorithm = GeneticProgramming(
        initial_forest=Forest.random_generate(
            pop_size=task["population_size"], descriptor=descriptor
        ),
        crossover=DefaultCrossover(),
        mutation=DefaultMutation(mutation_rate=0.1, descriptor=descriptor),
        selection=TournamentSelection(
            tournament_size=20, survivor_rate=0.5, elite_rate=0.1
        ),
        enable_pareto_front=False,
    )

    pipeline = StandardPipeline(
        algorithm,
        problem,
        generation_limit=1_000_000_000,  # is not optional, so set very large
        time_limit=600,  # in seconds
        is_show_details=False,
    )

    start_time = time.perf_counter()

    best = pipeline.run()

    elapsed_time = time.perf_counter() - start_time

    expr = str(best.to_sympy_expr())
    old_mse = -pipeline.best_fitness.item()

    y_pred = lambdify_expression(expr)(X_train)
    mse = np.mean((y_pred - y_train.flatten()) ** 2)

    return {
        "total_time_seconds": elapsed_time,
        "expression": f"'{expr}'",
        "mse": float(mse),
        "evaluations": pipeline.evaluation_count,
        "dataset": task["dataset"],
        "fold": task["fold"],
        "num_observations": task["num_observations"],
        "num_features": task["num_features"],
        "population_size": task["population_size"],
        "operator_set": task["operator_set"],
        "template_depth": task["depth"],
        "iteration": task["iteration"],
        "seed": task["seed"],
        "old_mse": old_mse,
    }


def gpu_worker(
    device_id: int,
    queue: Queue,
    result_queue: Queue,
) -> None:
    # Bind this process to exactly one GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    while True:
        job = queue.get()

        if job is None:
            break

        fn, args, kwargs = job

        try:
            result = fn(*args, **kwargs)
            result_queue.put(result)

        except Exception as e:
            result_queue.put({"error": str(e), "device": device_id})


def run_gpu_tasks(
    tasks: TaskGenerator,
    output_directory: Path,
    max_workers: int | None = None,
    dry_run: bool = False,
) -> None:
    if not dry_run:
        os.makedirs(output_directory, exist_ok=True)

    results_path = output_directory / "results.csv"
    fieldnames: list[str] = [
        "total_time_seconds",
        "expression",
        "mse",
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
        "old_mse",
    ]

    jobs: JobQueue = []

    for task in tasks:
        jobs.append((run_one_task, [], {"task": task}))

    n_jobs = len(jobs)

    if dry_run:
        print(f"Total GPU tasks: {n_jobs}")
        return

    with open(results_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        if max_workers is None or max_workers > 1:
            # Determine number of cuda capable devices
            num_gpus = get_num_cuda_devices()
            # Determine the amount of workers
            num_workers = (
                num_gpus if max_workers is None else min(num_gpus, max_workers)
            )

            job_queue = Queue()
            result_queue = Queue()

            workers = [
                Process(
                    target=gpu_worker,
                    args=(i, job_queue, result_queue),
                )
                for i in range(num_workers)
            ]

            for w in workers:
                w.start()

            for job in jobs:
                job_queue.put(job)

            for _ in workers:
                job_queue.put(None)

            with tqdm(total=len(jobs), leave=False, ascii=True) as pbar:
                completed = 0

                while completed < n_jobs:
                    result = result_queue.get()

                    if "error" in result:
                        print(f"Job failed: {result}")
                    else:
                        writer.writerow(result)

                    csvfile.flush()

                    completed += 1
                    pbar.update(1)

            for w in workers:
                w.join()

        else:
            for fn, args, kwargs in tqdm(
                jobs, total=len(jobs), leave=False, ascii=True
            ):
                result = fn(*args, **kwargs)
                writer.writerow(result)
