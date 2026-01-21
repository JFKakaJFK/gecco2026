import datetime
import os
import subprocess
from collections.abc import Callable, Generator
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Process, Queue, Value
from pathlib import Path
from typing import Never, TypedDict

import numpy as np
import pygom.gp as gp
from pygom import KernelVersion as KV
from tqdm import tqdm

from src.experiment.experiment_config import OPERATOR_SETS
from src.experiment.task import Task

JobQueue = list[
    tuple[Callable[[Task, int, Path], None], list[Never], dict[str, Task | str]]
]


class LogInfo(TypedDict):
    problem: str
    kernel: str
    population_size: int
    num_observations: int
    num_features: int
    branching_factor: int
    depth: int
    operator_set: str
    fold: int
    run: int
    seed: int
    var_y: float


def task_to_log_info(task: Task, run: int, var_y: float) -> list[tuple[str, str]]:
    kernel_str: str = (
        str(task["kernel"]).replace("KernelVersion.", "")
        if task["kernel"] is not None
        else "cpu"
    )

    log_info: LogInfo = LogInfo(
        problem=task["problem"],
        kernel=kernel_str,
        population_size=task["population_size"],
        num_observations=task["num_observations"],
        num_features=task["num_features"],
        branching_factor=task["branching_factor"],
        depth=task["depth"],
        operator_set=task["operator_set"],
        fold=task["fold"],
        run=run,
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
        f"{task['problem']}-{kernel_str}-pop{task['population_size']}"
        f"-obs{task['num_observations']}-feat{task['num_features']}"
        f"-depth{task['depth']}-op_{task['operator_set']}"
        f"-fold{task['fold']}"
    )


def run_one_task(task: Task, run: int, log_path: Path) -> None:
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

    log_info: list[tuple[str, str]] = task_to_log_info(
        task, run, float(np.var(y[:, 0]))
    )

    est = gp.SymbolicRegressor(
        gpu_accelerated=task["accelerated"],
        kernel_version=task["kernel"],
        linear_scaling=False,
        ims_kwargs={
            "initial_population_size": task["population_size"],
            "max_num_populations": 1,
        },
        rv_kwargs={"enabled": False},
        discrete_model_kwargs={
            "merge_continuous": False,
            "num_continuous_bins": 25,
            "normalize_initial_linkage_bias": True,
        },
        outputs=[(task["branching_factor"], task["depth"])],
        operators=OPERATOR_SETS[task["operator_set"]],
        target_objectives=task["target_objectives"],
        seed=task["seed"],
        random_state=task["seed"],
        budget_kwargs={
            # "max_evaluations": 1_000_000,  # TODO
            "max_duration": datetime.timedelta(hours=3),
        },
        tracking_kwargs={
            "logpath": log_path,
            "log_info": log_info,
            "max_generations_until_next_report": 1,
            "generation_factor": 1,
            "initial_evaluations_until_next_report": 100,
            "max_evaluations_until_next_report": 5000,
            "eval_factor": 1,
            "initial_time_until_next_report": datetime.timedelta(hours=1),
        },
    )

    est.fit(X, y)


def run_cpu_tasks(
    output_dir: Path,
    tasks: Generator[Task],
    max_workers: int | None = None,
    dry_run: bool = False,
) -> None:
    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)

    jobs: JobQueue = []

    for task in tasks:
        task_name = task_to_file_name(task)

        for run in range(task["num_iterations"]):
            test_name = task_name + f"-iter{run}"
            log_path = output_dir / f"{task['problem']}/cpu/{test_name}.csv"

            jobs.append(
                (run_one_task, [], {"task": task, "run": run, "log_path": log_path})
            )

    if dry_run:
        print(f"Total CPU tasks: {len(jobs)}")
        return

    if max_workers is None or max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(fn, *args, **kwargs) for fn, args, kwargs in jobs]

            progress = tqdm(total=len(futures), leave=False, ascii=True)
            for f in as_completed(futures):
                try:
                    f.result()
                    progress.update()
                except KeyboardInterrupt:
                    pool.shutdown(wait=False, cancel_futures=True)
                except Exception as e:
                    print(e)


# Cursed, but works
def get_num_cuda_devices() -> int:
    try:
        result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)

        return len(result.stdout.strip().splitlines())
    except Exception as e:
        raise Exception(f"No CUDA devices could be detected: {e}") from e


def gpu_worker(device_id: int, queue: Queue, progress_counter) -> None:
    # Bind this process to exactly one GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    while True:
        job = queue.get()

        if job is None:
            break

        fn, args, kwargs = job

        try:
            fn(*args, **kwargs)
        except Exception as e:
            print(f"[GPU {device_id}] Job failed: {e}")
        finally:
            with progress_counter.get_lock():
                progress_counter.value += 1


def run_gpu_tasks(
    output_dir: Path,
    tasks: Generator[Task],
    max_workers: int | None = None,
    dry_run: bool = False,
) -> None:
    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)

    jobs: JobQueue = []

    for task in tasks:
        task_name = task_to_file_name(task)

        # The combination of large population size, a million datapoints and these
        # kernels requires more memory than available on the GPU
        if (
            task["population_size"] >= 4096
            and task["num_observations"] >= 1e6
            and task["kernel"]
            in [
                KV.baseline,
                KV.restrict,
                KV.shared_memory,
            ]
        ):
            print(f"skipping kernel: {task['kernel']}")
            continue

        for run in range(task["num_iterations"]):
            kernel_str: str = str(task["kernel"]).replace("KernelVersion.", "")

            test_name = task_name + f"-iter{run}"
            log_path = output_dir / f"{task['problem']}/{kernel_str}/{test_name}.csv"

            jobs.append(
                (run_one_task, [], {"task": task, "run": run, "log_path": log_path})
            )

    if dry_run:
        print(f"Total GPU tasks: {len(jobs)}")
        return

    if max_workers is None or max_workers > 1:
        # Determine number of cuda capable devices
        num_gpus = get_num_cuda_devices()
        # Determine the amount of workers
        num_workers = num_gpus if max_workers is None else min(num_gpus, max_workers)

        print(f"Number of workers: {num_workers}")

        queue = Queue()
        progress_counter = Value("i", 0)

        workers = [
            Process(target=gpu_worker, args=(i, queue, progress_counter))
            for i in range(num_workers)
        ]

        for w in workers:
            w.start()

        for job in jobs:
            queue.put(job)

        for _ in workers:
            queue.put(None)

        with tqdm(total=len(jobs), leave=False, ascii=True) as pbar:
            while progress_counter.value < len(jobs):
                pbar.n = progress_counter.value
                pbar.refresh()

        for w in workers:
            w.join()

    else:
        for fn, args, kwargs in tqdm(jobs, total=len(jobs), leave=False, ascii=True):
            fn(*args, **kwargs)
