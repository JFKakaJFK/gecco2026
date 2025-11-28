import json
import os
import pathlib
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

import pygom
from pygom import *
from tqdm import tqdm

from .config import Config


def run_one(
    task_path: str,
    logfile: str,
    loginfo: list[tuple[str, str]],
    seed: int | None = None,
):
    task = Config.eval(Config.load(task_path), ctx=vars(pygom))

    if pathlib.Path(logfile).is_file():
        with open(logfile, "r") as f:
            *_, last = f.readlines()
            completed = len(_) > 0 and last.split(",")[0] != "Running"
            if completed:
                return

    try:
        Tracked.run(
            instance=task["instance"],
            method=task["method"],
            budget=task["budget"],
            config=TrackingOptions(logfile, loginfo, report_intermediate_results=False),
            seed=task.get("seed", seed),
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

                logfile = run_dir / f"{run:03d}.csv"

                # if there alread is a task and the tasks are identical, and the run completed without fail, then skip
                if task_path.is_file() and logfile.is_file():
                    existing = Config.load(task_path)

                    if task == existing or json.dumps(task) == json.dumps(existing):
                        with open(logfile, "r") as f:
                            *_, last_line = f.readlines()
                            status = last_line.split(",")[0]
                            if status != "Running" or status != "Aborted":
                                print(
                                    f"{problem_name} @ {dims:03d}D - {method_name}: skipping run {run}"
                                )
                                continue

                Config.save(task, task_path)
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
