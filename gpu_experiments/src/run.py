import os
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pygom.gp as gp
from tqdm import tqdm


def run_one_cpu_task(sr_instance, X, y):
    sr_instance.fit(X, y)


def run_cpu_tasks(
    output_directory: str | pathlib.Path,
    tasks,
    num_repeats: int = 1,
    max_workers: int | None = None,
):
    odir = pathlib.Path(output_directory)
    os.makedirs(odir, exist_ok=True)

    jobs = []

    for info in tasks:
        X_train = np.load(str(info["X_path"].absolute()))
        y_train = np.load(str(info["y_path"].absolute()))

        for run in range(num_repeats):
            test_name = f"""{info["problem"]}-cpu-pop{info["population_size"]}-obs{info["num_observations"]}-fold{info["fold"]}-iter{run}"""
            logpath = f"{output_directory}/{test_name}.csv"

            est = gp.SymbolicRegressor(
                gpu_accelerated=False,
                linear_scaling=False,
                ims_kwargs={
                    "initial_population_size": info["population_size"],
                    "max_num_populations": 1,
                },
                rv_kwargs={"enabled": False},
                discrete_model_kwargs={
                    "merge_continuous": False,
                    "num_continuous_bins": 25,
                    "normalize_initial_linkage_bias": True,
                },
                seed=info["seed"],
                random_state=info["seed"],
                tracking_kwargs={"logpath": logpath},
            )

            jobs.append(
                (run_one_cpu_task, [], {"sr_instance": est, "X": X_train, "y": y_train})
            )

    if max_workers is None or max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(fn, *args, **kwargs) for fn, args, kwargs in jobs]

            progress = tqdm(total=len(futures))
            for f in as_completed(futures):
                try:
                    f.result()
                    progress.update()
                except KeyboardInterrupt:
                    pool.shutdown(wait=False, cancel_futures=True)
                except Exception as e:
                    print(e)


def run_gpu_tasks(
    output_directory: str | pathlib.Path,
    tasks,
    num_repeats: int = 1,
):
    odir = pathlib.Path(output_directory)
    os.makedirs(odir, exist_ok=True)

    for info, kernel in tqdm(tasks, desc="Tasks"):
        v = str(kernel).replace("KernelVersion.", "")

        for run in tqdm(range(num_repeats), desc="Repeats", leave=False):
            X_train = np.load(str(info["X_path"].absolute()))
            y_train = np.load(str(info["y_path"].absolute()))

            test_name = f"""{info["problem"]}-{v}-pop{info["population_size"]}-obs{info["num_observations"]}-fold{info["fold"]}-iter{run}"""
            logpath = f"{output_directory}/{test_name}.csv"

            est = gp.SymbolicRegressor(
                gpu_accelerated=True,
                kernel_version=kernel,
                linear_scaling=False,
                ims_kwargs={
                    "initial_population_size": info["population_size"],
                    "max_num_populations": 1,
                },
                rv_kwargs={"enabled": False},
                discrete_model_kwargs={
                    "merge_continuous": False,
                    "num_continuous_bins": 25,
                    "normalize_initial_linkage_bias": True,
                },
                seed=info["seed"],
                random_state=info["seed"],
                tracking_kwargs={"logpath": logpath},
            )

            est.fit(X_train, y_train)
