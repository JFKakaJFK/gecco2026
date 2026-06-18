import datetime
import hashlib
import json
import os
import pathlib
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from urllib.parse import quote

import numpy as np
import pygom
from pygom import *
from tqdm import tqdm

from .config import Config


def run_one(
    task_path: str,
    logfile: str,
    loginfo: list[tuple[str, str]],
    seed: int | None = None,
    run: int | None = None,
):
    task = Config.eval(
        Config.load_config(task_path), ctx=dict(**vars(pygom), np=vars(np))
    )

    if pathlib.Path(logfile).is_file():
        with open(logfile, "r") as f:
            *_, last = f.readlines()
            completed = len(_) > 0 and last.split(",")[0] != "Running"
            if completed:
                return

    try:
        seed = task.get("seed", seed)
        if seed and run is not None:
            seed += run
        Tracked.run(
            instance=task["instance"],
            method=task["method"],
            budget=task["budget"],
            config=TrackingOptions(  #
                logfile,  #
                loginfo,
                # report_intermediate_results=False,
                max_generations_until_next_report=1,
                generation_factor=1,
                initial_evaluations_until_next_report=100,
                max_evaluations_until_next_report=5000,
                eval_factor=1,
                initial_time_until_next_report=datetime.timedelta(hours=1),
                # report_raw_solutions=True,
            ),
            seed=seed,
            population_size=task.get("population_size", None),
        )
    except Exception as e:
        print("Task:", task_path)
        print(e)
        raise e


def run_all(
    output_directory: str | pathlib.Path,
    problems,
    methods,
    budget,
    num_repeats: int = 1,
    clean: bool = False,
    limit: int | None = None,
    max_workers: int | None = None,
):
    odir = pathlib.Path(output_directory)

    if clean and odir.exists():
        shutil.rmtree(odir)
    os.makedirs(odir, exist_ok=True)

    jobs = []
    for problem_name, dims, instance in problems():
        for method_name, method in methods():
            run_dir = odir / problem_name / f"{dims:03d}" / method_name
            os.makedirs(run_dir, exist_ok=True)
            task_path = run_dir / "task.yaml"
            task = next(
                Config.expand(dict(instance=instance, method=method, budget=budget))
            )

            for run in range(num_repeats):
                if limit is not None and len(jobs) >= limit:
                    break

                logfile = run_dir / f"{run:04d}.csv"

                # if there alread is a task and the tasks are identical, and the run completed without fail, then skip
                if task_path.is_file() and logfile.is_file():
                    existing = Config.load_config(task_path)

                    if task == existing or json.dumps(task) == json.dumps(existing):
                        with open(logfile, "r") as f:
                            *_, last_line = f.readlines()
                            status = last_line.split(",")[0]
                            if status != "Running" or status != "Aborted":
                                print(
                                    f"{problem_name} @ {dims:03d}D - {method_name}: skipping run {run}"
                                )
                                continue

                Config.save_config(task, task_path)
                jobs.append(
                    (
                        run_one,
                        [],
                        dict(
                            task_path=task_path,
                            logfile=str(logfile),
                            loginfo=list(
                                dict(
                                    problem_name=problem_name,
                                    method_name=method_name,
                                    dims=str(dims),
                                    run=str(run),
                                ).items()
                            ),
                        ),
                    )
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
    else:
        for fn, args, kwargs in tqdm(jobs):
            fn(*args, **kwargs)


def run_tasks(
    output_directory: str | pathlib.Path,
    tasks,
    num_repeats: int = 1,
    clean: bool = False,
    limit: int | None = None,
    max_workers: int | None = None,
):
    odir = pathlib.Path(output_directory)

    if clean and odir.exists():
        shutil.rmtree(odir)
    os.makedirs(odir, exist_ok=True)

    jobs = []
    for info, config in tasks:
        run_dir = odir
        if "problem_name" in info:
            run_dir /= quote(info["problem_name"], safe=" ")
        if "dims" in info:
            run_dir /= f"{info['dims']:03d}"
        if "method_name" in info:
            run_dir /= quote(info["method_name"], safe=" ")
        run_dir /= hashlib.sha256(json.dumps(info).encode()).hexdigest()[:10]

        os.makedirs(run_dir, exist_ok=True)
        task_path = run_dir / "task.yaml"
        task = next(Config.expand(config))

        for run in range(num_repeats):
            if limit is not None and len(jobs) >= limit:
                break

            logfile = run_dir / f"{run:04d}.csv"

            # if there alread is a task and the tasks are identical, and the run completed without fail, then skip
            if task_path.is_file() and logfile.is_file():
                existing = Config.load_config(task_path)

                if task == existing or json.dumps(task) == json.dumps(existing):
                    with open(logfile, "r") as f:
                        *_, last_line = f.readlines()
                        status = last_line.split(",")[0]
                        if status != "Running" or status != "Aborted":
                            print(f"{run_dir}: skipping run {run}")
                            continue

            Config.save_config(task, task_path)
            jobs.append(
                (
                    run_one,
                    [],
                    dict(
                        task_path=task_path,
                        logfile=str(logfile),
                        loginfo=[(str(k), str(v)) for k, v in info.items()]
                        + [("run", str(run))],
                        run=run,
                    ),
                )
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
    else:
        for fn, args, kwargs in tqdm(jobs):
            fn(*args, **kwargs)
