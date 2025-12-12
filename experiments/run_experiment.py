import csv
import pmlb

import numpy as np
import pygom.gp as gp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path
from pygom import KernelVersion
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import plots


def run_single_experiment(
    X_train, X_test, y_train, y_test, 
    pop_size: int, gpu_accelerated: bool, kernel_version: KernelVersion, 
    seed: int, logpath: Path
):
    est = gp.SymbolicRegressor(
        gpu_accelerated=gpu_accelerated,
        kernel_version=kernel_version,
        linear_scaling=False,
        # budget_kwargs=dict(
        #     max_time_seconds=5,
        # ),
        ims_kwargs=dict(
            initial_population_size=pop_size, max_num_populations=1
        ),
        rv_kwargs=dict(enabled=False),  # disable rv optimization
        discrete_model_kwargs=dict(
            merge_continuous=False,
            num_continuous_bins=25,
            normalize_initial_linkage_bias=True,
        ),
        seed=seed,
        random_state=seed,
        tracking_kwargs=dict(
            logpath=logpath,
        ),
    )

    est.fit(X_train, y_train)

    r2_train = r2_score(y_train, est.predict(X_train))
    r2_test = r2_score(y_test, est.predict(X_test))

    mse_train = mean_squared_error(y_train, est.predict(X_train))
    mse_test = mean_squared_error(y_test, est.predict(X_test))

    return [r2_train, r2_test, mse_train, mse_test]


def run_experiments(
        benchmarks: list[str], 
        pop_sizes: list[int], 
        results_dir: Path, 
        include_cpu=False, 
        iterations=3
    ):
    results = [
        [
            "benchmark",
            "pop_size",
            "device",
            "iteration",
            "r2_train",
            "r2_test",
            "mse_train",
            "mse_test",
        ]
    ]

    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 2**16, size=iterations)

    accelerated = [True, False] if include_cpu else [True]

    for b in benchmarks:
        X, y = pmlb.fetch_data(
            b, return_X_y=True, local_cache_dir="./plmb_cache"
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        for p in pop_sizes:
            for i, seed in enumerate(seeds):
                for acc in accelerated:
                    test_name = f"{b}-{"gpu" if acc else "cpu"}-pop{p}-iter{i}"
                    logpath = f"{results_dir}/{test_name}.csv"
                    print(f"Starting experiment: {test_name}")

                    temp = run_single_experiment(
                        X_train, X_test, y_train, y_test, p, 
                        acc, KernelVersion.block_reduce, seed, logpath
                    )
                    results.append([b, p, "gpu" if acc else "cpu", i] + temp)

    with open(f"{results_dir}/error_result.csv", "w", newline="") as csvfile:
        csvfile.truncate()
        writer = csv.writer(csvfile)
        writer.writerows(results)

def run_kernel_version_experiment(
        benchmarks: list[str], 
        kernel_versions: list[KernelVersion], 
        results_dir: Path, 
        iterations=5
    ):
    p = 512

    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 2**16, size=iterations)

    for b in benchmarks:
        print("Fetching dataset")
        X, y = pmlb.fetch_data(
            b, return_X_y=True, local_cache_dir="./plmb_cache"
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        for kernel_version in kernel_versions:
            v = str(kernel_version).replace("KernelVersion.", "")

            for obs in [10, 100, 1_000, 10_000, 100_000, 1_000_000]:
                for i, seed in enumerate(seeds):
                    test_name = f"{b}-{v}-pop{p}-obs{obs}-iter{i}"
                    logpath = f"{results_dir}/{test_name}.csv"
                    
                    print(f"Starting experiment: {test_name}")

                    run_single_experiment(
                        X_train[:obs],
                        X_test[:obs],
                        y_train[:obs],
                        y_test[:obs],
                        p,
                        True,
                        kernel_version,
                        seed,
                        logpath
                    )

def main():
    run_date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    results_dir = Path("experiments/results") / run_date

    kernel_versions = [
        KernelVersion.baseline,
        KernelVersion.restrict,
        KernelVersion.shared_memory,
        KernelVersion.block_reduce,
    ]

    dataset_sizes = {
        "first_principles_kepler": {"observations": 6, "features": 1},
        "nikuradse_2": {"observations": 362, "features": 1},
        "feynman_I_6_2a": 100_000,
        "542_pollution": 60,
        "503_wind": 6574,
        "1191_BNG_pbc": 1_000_000,
        "505_tecator": 240,
    }

    benchmarks = [
        "first_principles_kepler",  # 6 obs, 1 feat
        "nikuradse_2",  # 362 obs, 1 feat
        "feynman_I_6_2a",  # 100_000 obs, 1 feat
        "542_pollution",  # 60 obs, 15 feat
        "503_wind",  # 6574 obs, 14 feat
        "1191_BNG_pbc",  # 1_000_000 obs, 18 feat
        "505_tecator",  # 240 obs, 124 feat
    ]
    pop_sizes = [256, 512, 1024, 2048]

    run_experiments(benchmarks, pop_sizes, results_dir, include_cpu=True)
    plots.plot_cpu_gpu("2025-12-10_09:02:23", dataset_sizes)
    
    run_kernel_version_experiment(benchmarks, kernel_versions, results_dir, iterations=5)
    plots.plot_kernel_versions("2025-12-11_09:56:21")


if __name__ == "__main__":
    main()
