import datetime
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pygom.gp as gp
from pygom import KernelVersion as KV
from tqdm import tqdm


def run_one_task(options):
    X_train = np.load(options["X_path"].absolute())
    y_train = np.load(options["y_path"].absolute())

    obs = options["num_observations"]
    feat = options["num_features"]

    assert X_train.shape[0] >= obs, (
        f"Requested obs={obs}, but fold has only {X_train.shape[0]}"
    )
    assert X_train.shape[1] >= feat, (
        f"Requested feat={feat}, but fold has only {X_train.shape[1]}"
    )

    X = X_train[:obs, :feat]
    y = y_train[:obs]

    # Determine info that needs to be logged
    loginfo = [
        (str(k), str(v))
        for k, v in options.items()
        if k
        not in [
            "kernel",
            "operators",
            "template",
            "X_path",
            "y_path",
            "X_test_path",
            "y_test_path",
            "logpath",
        ]
    ] + [
        ("var_y", str(float(np.var(y[:, 0])))),
        ("kernel", str(options["kernel"]).replace("KernelVersion.", "")),
    ]

    est = gp.SymbolicRegressor(
        gpu_accelerated=options["gpu_accelerated"],
        kernel_version=options["kernel"],
        linear_scaling=False,
        ims_kwargs={
            "initial_population_size": options["population_size"],
            "max_num_populations": 1,
        },
        rv_kwargs={"enabled": False},
        discrete_model_kwargs={
            "merge_continuous": False,
            "num_continuous_bins": 25,
            "normalize_initial_linkage_bias": True,
        },
        outputs=options["template"],
        operators=options["operators"],
        seed=options["seed"],
        random_state=options["seed"],
        budget_kwargs={
            "max_evaluations": 1_000_000,  # TODO
            "max_duration": datetime.timedelta(minutes=15),
        },
        tracking_kwargs={
            "logpath": options["logpath"],
            "log_info": loginfo,
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
    output_directory: str | pathlib.Path,
    tasks,
    num_repeats: int = 1,
    max_workers: int | None = None,
    dry_run: bool = False,
):
    if not dry_run:
        odir = pathlib.Path(output_directory)
        os.makedirs(odir, exist_ok=True)

    jobs = []

    for task in tasks:
        prob = task["problem"]
        obs = task["num_observations"]
        feat = task["num_features"]
        pop = task["population_size"]
        template = task["template"]
        op_set = task["operator_set"]
        fold = task["fold"]

        h = template[0][1] + 1

        task_name = (
            f"{prob}-cpu-pop{pop}-obs{obs}-feat{feat}-height{h}-op_{op_set}-fold{fold}"
        )

        for run in range(num_repeats):
            test_name = task_name + f"-iter{run}"
            logpath = f"{output_directory}/{prob}/cpu/{test_name}.csv"

            options = dict(task)
            options["gpu_accelerated"] = False
            options["kernel"] = "cpu"
            options["run"] = run
            options["logpath"] = logpath
            options["template_height"] = h

            jobs.append((run_one_task, [], {"options": options}))

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


def run_gpu_tasks(
    output_directory: str | pathlib.Path,
    tasks,
    num_repeats: int = 1,
    dry_run: bool = False,
):
    if not dry_run:
        odir = pathlib.Path(output_directory)
        os.makedirs(odir, exist_ok=True)

    jobs = []

    for task, kernel in tasks:
        kv = str(kernel).replace("KernelVersion.", "")

        prob = task["problem"]
        obs = task["num_observations"]
        feat = task["num_features"]
        pop = task["population_size"]
        template = task["template"]
        op_set = task["operator_set"]
        fold = task["fold"]

        h = template[0][1] + 1

        task_name = (
            f"{prob}-{kv}-pop{pop}-obs{obs}-feat{feat}-height{h}-op_{op_set}-fold{fold}"
        )

        # The combination of a million datapoints and these kernels requires more
        # memory than available on the GPU
        if obs >= 1e6 and kernel in [KV.baseline, KV.restrict, KV.shared_memory]:
            continue

        for run in range(num_repeats):
            test_name = task_name + f"-iter{run}"
            logpath = f"{output_directory}/{prob}/{kv}/{test_name}.csv"

            options = dict(task)
            options["gpu_accelerated"] = True
            options["kernel"] = kernel
            options["run"] = run
            options["logpath"] = logpath
            options["template_height"] = h

            jobs.append((run_one_task, [], {"options": options}))

    if dry_run:
        print(f"Total GPU tasks: {len(jobs)}")
        return

    for fn, args, kwargs in tqdm(jobs, total=len(jobs), leave=False, ascii=True):
        fn(*args, **kwargs)
