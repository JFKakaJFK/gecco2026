import csv
import pmlb

import numpy as np
import pygom.gp as gp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import plots


def run_single_experiment(
    X_train, X_test, y_train, y_test, pop_size, gpu_accelerated, seed, logpath
):
    est = gp.SymbolicRegressor(
        gpu_accelerated=gpu_accelerated,
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
        random_state=seed,
        seed=seed,
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


def run_experiments(benchmarks, pop_sizes, results_dir, include_cpu=False):
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

    accelerated = [True, False] if include_cpu else [True]

    for b in benchmarks:
        X, y = pmlb.fetch_data(
            b, return_X_y=True, local_cache_dir="./plmb_cache"
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        for p in pop_sizes:
            for i in range(3):
                seed = np.random.randint(2**16)

                for acc in accelerated:
                    test_name = f"{b}-{"gpu" if acc else "cpu"}-pop{p}-iter{i}"
                    logpath = f"{results_dir}/{test_name}.csv"
                    print(f"Starting experiment: {test_name}")

                    temp = run_single_experiment(
                        X_train, X_test, y_train, y_test, p, acc, seed, logpath
                    )
                    results.append([b, p, "gpu" if acc else "cpu", i] + temp)

    with open(f"{results_dir}/error_result.csv", "w", newline="") as csvfile:
        csvfile.truncate()
        writer = csv.writer(csvfile)
        writer.writerows(results)

def run_kernel_experiment(kernel_version, results_dir, iterations=5):
    b = "1191_BNG_pbc"
    p = 512

    print("Fetching dataset")
    X, y = pmlb.fetch_data(
        b, return_X_y=True, local_cache_dir="./plmb_cache"
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    rng = np.random.default_rng(42)
    seeds = rng.integers(0, 2**16, size=iterations)

    for obs in [10, 100, 1_000, 10_000, 100_000, 1_000_000]:
        for i, seed in enumerate(seeds):
            test_name = f"{b}-{kernel_version}-pop{p}-obs{obs}-iter{i}"
            logpath = f"{results_dir}/{test_name}.csv"
            print(f"Starting experiment: {test_name}")

            run_single_experiment(
                X_train[:obs],
                X_test[:obs],
                y_train[:obs],
                y_test[:obs],
                p,
                True,
                seed,
                logpath
            )

def main():
    run_date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    results_dir = Path("experiments/results") / run_date
    # results_dir.mkdir(parents=True, exist_ok=True)

    dataset_sizes = {
        "first_principles_kepler": 6,
        "nikuradse_2": 362,
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
        # "1191_BNG_pbc",  # 1_000_000 obs, 18 feat
        # "505_tecator",  # 240 obs, 124 feat
    ]
    pop_sizes = [256, 512, 1024, 2048]

    # run_experiments(benchmarks, pop_sizes, results_dir, include_cpu=True)
    # plots.plot_cpu_gpu("2025-12-10_09:02:23", dataset_sizes)
    
    # run_kernel_experiment("baseline", results_dir)
    plots.plot_gpu_versions(["2025-12-10_12:44:03", "2025-12-10_12:24:59"])


if __name__ == "__main__":
    main()
