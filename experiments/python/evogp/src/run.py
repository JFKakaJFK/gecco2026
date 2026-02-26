import csv
import os
import subprocess
from collections.abc import Callable
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Never

import numpy as np
import torch
from evogp.algorithm import (
    CombinedMutation,
    DefaultCrossover,
    DefaultMutation,
    DefaultSelection,
    DeleteMutation,
    DiversityCrossover,
    GeneticProgramming,
    TournamentSelection,
)
from evogp.pipeline import StandardPipeline
from evogp.problem import SymbolicRegression
from evogp.tree import Forest, GenerateDescriptor
from tqdm import tqdm

from src.experiment_config import OPERATOR_SETS
from src.task import Task, TaskGenerator

JobQueue = list[tuple[Callable[[Task, Path], None], list[Never], dict[str, Task]]]

torch.random.manual_seed(42)
torch.cuda.manual_seed(42)


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
        max_layer_cnt=task["depth"],
        const_range=const_range,
        sample_cnt=10_000,
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
        time_limit=60,  # in seconds
        is_show_details=False,
    )

    best = pipeline.run()
    mse = -pipeline.best_fitness.item()
    tree = best.to_sympy_expr()

    return {
        "dataset": task["dataset"],
        "population_size": task["population_size"],
        "num_observations": task["num_observations"],
        "num_features": task["num_features"],
        "depth": task["depth"],
        "operator_set": task["operator_set"],
        "fold": task["fold"],
        "iteration": task["iteration"],
        "seed": task["seed"],
        "mse": float(mse),
        "tree": f"'{tree}'",
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
            print(f"[GPU {device_id}] Job failed: {e}")


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
        "dataset",
        "population_size",
        "num_observations",
        "num_features",
        "depth",
        "operator_set",
        "fold",
        "iteration",
        "seed",
        "mse",
        "tree",
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
