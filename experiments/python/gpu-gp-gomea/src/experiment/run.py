import csv
import datetime
import math
import os
import re
import subprocess
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Process, Queue, Value
from multiprocessing.sharedctypes import Synchronized
from pathlib import Path
from typing import Never, TypedDict

import numpy as np
import pandas as pd
import pygom.gp as gp
import sympy as sym
from pygom import KernelVersion as KV
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from tqdm import tqdm

from src.experiment.experiment_config import BASELINE_KV, OPERATOR_SETS
from src.experiment.task import Task, TaskGenerator, TaskTransform

JobQueue = list[tuple[Callable[[Task, Path], None], list[Never], dict[str, Task | str]]]

# Some kernels have a limit to the amount of work they can do
BASELINE_LIMIT = 3e9  # roughly 12GB of memory required
BLOCK_REDUCE_LIMIT = 1024**2


class LogInfo(TypedDict):
    dataset: str
    kernel: str
    population_size: int
    num_observations: int
    num_features: int
    branching_factor: int
    depth: int
    operator_set: str
    fold: int
    iteration: int
    seed: int
    var_y: float


def task_to_log_info(task: Task, var_y: float) -> list[tuple[str, str]]:
    kernel_str: str = (
        str(task["kernel"]).replace("KernelVersion.", "")
        if task["kernel"] is not None
        else "cpu"
    )

    log_info: LogInfo = LogInfo(
        dataset=task["dataset"],
        kernel=kernel_str,
        population_size=task["population_size"],
        num_observations=task["num_observations"],
        num_features=task["num_features"],
        branching_factor=task["branching_factor"],
        depth=task["depth"],
        operator_set=task["operator_set"],
        fold=task["fold"],
        iteration=task["iteration"],
        seed=task["seed"],
        var_y=var_y,
    )

    return [(k, str(v)) for k, v in log_info.items()]


def task_to_file_name(task: Task) -> str:
    kernel_str: str = (
        str(task["kernel"]).replace("KernelVersion.", "")
        if task["kernel"] is not None
        else "cpu"
    )

    return (
        f"{task['dataset']}-{kernel_str}-pop{task['population_size']}"
        f"-obs{task['num_observations']}-feat{task['num_features']}"
        f"-depth{task['depth']}-op_{task['operator_set']}"
        f"-fold{task['fold']}-iter{task['iteration']}"
    )


def determine_task_success(csv_path: Path) -> bool:
    if not csv_path.exists():
        return False

    last_row = None

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            last_row = row

    return last_row is not None and last_row.get("status") == "TargetReached"


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


def run_one_task(task: Task, log_path: Path) -> None:
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

    if np.isnan(X_train).any():
        imputer = IterativeImputer(
            max_iter=10,
            random_state=task["seed"],
            sample_posterior=True,
        )
        X_train = imputer.fit_transform(X_train)

    log_info: list[tuple[str, str]] = task_to_log_info(task, float(np.var(y[:, 0])))

    if task["population_size"] is None:
        raise ValueError("Population size cannot be None")

    est = gp.SymbolicRegressor(
        gpu_accelerated=task["accelerated"],
        kernel_version=task["kernel"],
        linear_scaling=False,
        ims_kwargs={
            "initial_population_size": task["population_size"],
            "max_num_populations": 1,
            "restart_stale_populations": True,
        },
        rv_kwargs={"enabled": False},
        discrete_model_kwargs={
            "metric": "node_proximity",
            "merge_continuous": False,
            "num_continuous_bins": 25,
            "normalize_initial_linkage_bias": False,
        },
        outputs=[(task["branching_factor"], task["depth"])],
        operators=OPERATOR_SETS[task["operator_set"]],
        target_objectives=task["target_objectives"],
        seed=task["seed"],
        random_state=task["seed"],
        budget_kwargs={
            # "max_evaluations": 1_000_000,
            "max_duration": datetime.timedelta(minutes=task["max_duration"]),
        },
        tracking_kwargs={
            "logpath": log_path,
            "log_info": log_info,
            "report_intermediate_results": False,
            # "max_generations_until_next_report": 1,
            # "generation_factor": 1,
            # "max_evaluations_until_next_report": 100_000,
            # "eval_factor": 100_000,
            # "initial_time_until_next_report": datetime.timedelta(hours=1),
        },
    )

    est.fit(X, y)

    df = pd.read_csv(log_path)
    expr = df.loc[df.index[-1], "expressions"]

    y_pred = lambdify_expression(expr)(X)
    actual_mse = np.mean((y_pred - y.flatten()) ** 2)

    df.loc[df.index[-1], "mse"] = actual_mse
    df.to_csv(log_path, index=False)


def run_cpu_tasks(
    tasks: TaskGenerator,
    output_directory: Path,
    max_workers: int | None = None,
    required_rate: float | None = None,
    dry_run: bool = False,
) -> float | None:
    if not dry_run:
        os.makedirs(output_directory, exist_ok=True)

    jobs: JobQueue = []

    for task in tasks:
        task_name = task_to_file_name(task)
        log_path = output_directory / f"{task['dataset']}/cpu/{task_name}.csv"

        jobs.append((run_one_task, [], {"task": task, "log_path": log_path}))

    n_jobs = len(jobs)

    if dry_run:
        print(f"Total CPU tasks: {n_jobs}")
        return

    if required_rate is not None:
        max_failures = math.ceil(n_jobs * (1 - required_rate))

    failure_count = 0
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(fn, *args, **kwargs): kwargs["log_path"]
            for fn, args, kwargs in jobs
        }

        progress = tqdm(total=n_jobs, leave=False, ascii=True)

        try:
            for f in as_completed(futures):
                log_path = futures[f]

                try:
                    f.result()

                    if not determine_task_success(log_path):
                        failure_count += 1
                except Exception as e:
                    print(e)
                    failure_count += 1
                finally:
                    completed += 1
                    progress.update()

                if required_rate is not None and failure_count > max_failures:
                    pool.shutdown(wait=False, cancel_futures=True)

                    progress.close()

                    return 1 - failure_count / completed

        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)

            progress.close()

            return

    if required_rate is not None:
        return 1 - failure_count / completed


# Cursed, but works
def get_num_cuda_devices() -> int:
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)

        return len(result.stdout.strip().splitlines())
    except Exception as e:
        raise Exception(f"No CUDA devices could be detected: {e}") from e


def gpu_worker(
    device_id: int,
    queue: Queue,
    progress_counter: Synchronized,
    failure_counter: Synchronized,
    max_failures: int | None,
) -> None:
    # Bind this process to exactly one GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    while True:
        # Early stopping
        if max_failures is not None:
            with failure_counter.get_lock():
                if failure_counter.value > max_failures:
                    break

        job = queue.get()

        if job is None:
            break

        fn, args, kwargs = job
        log_path = kwargs["log_path"]

        try:
            fn(*args, **kwargs)

            if not determine_task_success(log_path):
                with failure_counter.get_lock():
                    failure_counter.value += 1

        except Exception as e:
            print(f"[GPU {device_id}] Job failed: {e}")
            with failure_counter.get_lock():
                failure_counter.value += 1

        finally:
            with progress_counter.get_lock():
                progress_counter.value += 1


def run_gpu_tasks(
    tasks: TaskGenerator,
    output_directory: Path,
    max_workers: int | None = None,
    required_rate: float | None = None,
    dry_run: bool = False,
) -> None:
    if not dry_run:
        os.makedirs(output_directory, exist_ok=True)

    jobs: JobQueue = []

    for task in tasks:
        task_name = task_to_file_name(task)
        kernel = task["kernel"]

        # Determine the amount of work, and if the kernel can perform all that work
        num_obs = task["num_observations"]
        num_work = task["population_size"] * num_obs
        skip_kernel = (kernel in BASELINE_KV and num_work >= BASELINE_LIMIT) or (
            kernel == KV.block_reduce and num_obs >= BLOCK_REDUCE_LIMIT
        )

        if skip_kernel:
            print(f"skipping kernel: {task['kernel']}")
            continue

        kernel_str: str = str(kernel).replace("KernelVersion.", "")

        log_path = output_directory / f"{task['dataset']}/{kernel_str}/{task_name}.csv"

        jobs.append((run_one_task, [], {"task": task, "log_path": log_path}))

    n_jobs = len(jobs)

    if dry_run:
        print(f"Total GPU tasks: {n_jobs}")
        return

    if required_rate is not None:
        max_failures = math.ceil(n_jobs * (1 - required_rate))
    else:
        max_failures = None

    if max_workers is None or max_workers > 1:
        # Determine number of cuda capable devices
        num_gpus = get_num_cuda_devices()
        # Determine the amount of workers
        num_workers = num_gpus if max_workers is None else min(num_gpus, max_workers)

        queue = Queue()
        progress_counter: Synchronized = Value("i", 0)
        failure_counter: Synchronized = Value("i", 0)

        workers = [
            Process(
                target=gpu_worker,
                args=(i, queue, progress_counter, failure_counter, max_failures),
            )
            for i in range(num_workers)
        ]

        for w in workers:
            w.start()

        for job in jobs:
            queue.put(job)

        for _ in workers:
            queue.put(None)

        with tqdm(total=len(jobs), leave=False, ascii=True) as pbar:
            while True:
                with progress_counter.get_lock():
                    completed = progress_counter.value

                with failure_counter.get_lock():
                    failures = failure_counter.value

                pbar.n = completed
                pbar.refresh()

                if completed >= n_jobs:
                    break

                if max_failures is not None and failures > max_failures:
                    break

        for w in workers:
            w.join()

    else:
        for fn, args, kwargs in tqdm(jobs, total=len(jobs), leave=False, ascii=True):
            fn(*args, **kwargs)

    if required_rate is not None:
        return 1 - failure_counter.value / completed


def run_tasks(
    tasks: TaskGenerator,
    transform: TaskTransform | None,
    runner: Callable[..., float | None],
    output_directory: Path,
    required_rate: float | None = None,
    dry_run: bool = False,
) -> float | None:
    tasks = tasks if transform is None else transform(tasks)

    return runner(tasks, output_directory, required_rate=required_rate, dry_run=dry_run)
