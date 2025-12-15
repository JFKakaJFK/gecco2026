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

from .config import extract, instantiate, load_config, save_config


def run_one(
    task_path: str,
    logfile: str,
    loginfo: list[tuple[str, str]],
    seed: int | None = None,
):
    task = instantiate(load_config(task_path), ctx=dict(**vars(pygom), np=vars(np)))

    # if pathlib.Path(logfile).is_file():
    #     with open(logfile, "r") as f:
    #         *_, last = f.readlines()
    #         status = last.split(",")[0]
    #         completed = len(_) > 0 and status != "Running" and status != "Aborted"
    #         if completed:
    #             return

    # print(task_path)

    try:
        Tracked.run(
            instance=task["instance"],
            method=task["method"],
            budget=task["budget"],
            config=TrackingOptions(  #
                logfile,  #
                loginfo,
                # report_intermediate_results=False,
                max_generations_until_next_report=10,
                generation_factor=2,
                initial_evaluations_until_next_report=1,
                max_evaluations_until_next_report=100000,
                eval_factor=100000,
                initial_time_until_next_report=datetime.timedelta(hours=1),
            ),
            seed=task.get("seed", seed),
            population_size=task.get("population_size", None),
        )
    except Exception as e:
        print("Task:", task_path)
        print(e)
        raise e


def compute_run_path(output_directory: str | pathlib.Path, info):
    run_dir = pathlib.Path(output_directory)
    if "problem_name" in info:
        run_dir /= quote(info["problem_name"], safe=' (){}$_+-"')
    if "dims" in info:
        run_dir /= f"{info['dims']:03d}D"
    if "method_name" in info:
        run_dir /= quote(info["method_name"], safe=' (){}$_+-"')
    if "fold" in info:
        run_dir /= f"fold{info['fold']:03d}"
    if "run" in info:
        run_dir /= f"run{info['run']:03d}"
    run_dir /= hashlib.sha256(json.dumps(info).encode()).hexdigest()[:10]
    return run_dir


def run_tasks(
    output_directory: str | pathlib.Path,
    tasks,
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
        run_dir = compute_run_path(odir, info)
        os.makedirs(run_dir, exist_ok=True)

        task_path = run_dir / "task.yaml"
        task = extract(config)

        if limit is not None and len(jobs) >= limit:
            break

        logfile = run_dir / "stats.csv"

        # if there alread is a task and the tasks are identical, and the run completed without fail, then skip
        if task_path.is_file() and logfile.is_file():
            existing = load_config(task_path)

            if task == existing or json.dumps(task) == json.dumps(existing):
                with open(logfile, "r") as f:
                    *_, last_line = f.readlines()
                    status = last_line.split(",")[0]
                    if status != "Running" or status != "Aborted":
                        print(f"Skipping {run_dir} ...")
                        continue

        save_config(task, task_path)
        jobs.append(
            (
                run_one,
                [],
                dict(
                    task_path=task_path,
                    logfile=str(logfile),
                    loginfo=[(str(k), str(v)) for k, v in info.items()],
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
