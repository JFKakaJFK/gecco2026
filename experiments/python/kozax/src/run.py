import csv
import os
import subprocess
from collections.abc import Callable
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Never

import numpy as np
from tqdm import tqdm

from src.experiment_config import OPERATOR_SETS
from src.task import Task, TaskGenerator

JobQueue = list[tuple[Callable[[Task, Path], None], list[Never], dict[str, Task]]]


# Cursed, but works
def get_num_cuda_devices() -> int:
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)

        return len(result.stdout.strip().splitlines())
    except Exception as e:
        raise Exception(f"No CUDA devices could be detected: {e}") from e


def run_one_task(task: Task) -> dict:
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    from kozax.fitness_functions.base_fitness_function import BaseFitnessFunction
    from kozax.genetic_programming import GeneticProgramming

    class FitnessFunction(BaseFitnessFunction):
        def __call__(self, candidate, data, tree_evaluator):
            X, Y = data
            predictions = jax.vmap(tree_evaluator, in_axes=[None, 0])(candidate, X)
            return jnp.mean(jnp.square(predictions - Y))

    def best_solution(strategy):
        pareto_fitness, pareto_solutions = strategy.pareto_front

        valid = pareto_fitness < strategy.max_fitness
        fitness = jnp.where(valid, pareto_fitness, jnp.inf)

        best_idx = int(jnp.argmin(fitness))

        best_fitness = float(pareto_fitness[best_idx])
        best_tree = strategy.expression_to_string(pareto_solutions[best_idx])

        return best_fitness, best_tree

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

    X = X_train[:obs, :feat]
    y = y_train[:obs]

    # temp = float(np.nanmax(np.abs(y_train)))
    # const_range = (-temp, temp)

    if task["population_size"] is None:
        raise ValueError("Population size cannot be None")

    fitness_function = FitnessFunction()

    operator_set = [
        ("+", lambda x, y: jnp.add(x, y), 2),
        ("-", lambda x, y: jnp.subtract(x, y), 2),
        ("*", lambda x, y: jnp.multiply(x, y), 2),
        ("/", lambda x, y: jnp.divide(x, y), 2),
        ("sin", lambda x: jnp.sin(x), 1),
        ("cos", lambda x: jnp.cos(x), 1),
        ("square", lambda x: jnp.square(x), 1),
        ("sqrt", lambda x: jnp.sqrt(x), 1),
        ("exp", lambda x: jnp.exp(x), 1),
        ("log", lambda x: jnp.log(x), 1),
        ("pow", lambda x, y: jnp.power(x, y), 2),
    ]

    variable_list = [[f"x{i}" for i in range(task["num_features"])]]

    strategy = GeneticProgramming(
        num_generations=10,
        population_size=task["population_size"],
        fitness_function=fitness_function,
        operator_list=operator_set,
        variable_list=variable_list,
        max_init_depth=task["depth"],
        max_nodes=task["max_nodes"],
        device_type="cpu",
        tournament_size=20,
        crossover_probability_factors=0.9,
        mutation_probability_factors=0.1,
    )

    key = jr.PRNGKey(task["seed"])

    strategy.fit(key, (X, y), verbose=False)

    mse, tree = best_solution(strategy)

    del strategy

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
