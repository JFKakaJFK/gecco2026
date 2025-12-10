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
        # Get benchmark data and split
        X, y = pmlb.fetch_data(
            b, return_X_y=True, local_cache_dir="./plmb_cache"
        )

        # assert not np.any(np.isnan(X))
        # assert not np.any(np.isnan(y))
        # assert np.all(np.isfinite(X))
        # assert np.all(np.isfinite(y))

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


def plot_error():
    error = pd.read_csv("results/error_result.csv")

    # Melt into long format for train/test split
    error_melted = error.melt(
        id_vars=["benchmark", "device"],
        value_vars=["r2_train", "r2_test"],
        var_name="split",
        value_name="r2",
    )

    # Plot: device as hue, optional split as facet
    r2 = sns.catplot(
        data=error_melted,
        kind="bar",
        x="benchmark",
        y="r2",
        hue="device",  # CPU vs GPU as main highlight
        col="split",  # optional: separate train/test columns
        sharey=True,
    )

    r2.despine()
    r2.set_axis_labels("", "R² score")
    r2.set_titles("{col_name}")  # show only train/test in column title
    r2._legend.set_title("Device")  # legend clearly shows CPU/GPU

    # plt.tight_layout()
    r2.savefig("r2_device.png")
    # plt.show()


def plot_convergence():
    path = Path("experiments/results/")

    dfs = []

    for file in path.glob("*.csv"):
        df = pd.read_csv(file)

        # Extract metadata (you can adjust this to your filenames)
        # Example filename: "benchmark1_pop100_cpu.csv"
        parts = file.stem.split("_")
        if len(parts) < 3:
            continue
        benchmark = "_".join(parts[:2])
        device = parts[2]
        population = int(parts[3])

        # Filter converged row only
        df = df[df["status"] == "Converged"].copy()

        # Add metadata columns
        df["benchmark"] = benchmark
        df["population"] = population
        df["device"] = device
        df["ratio"] = df["evaluations"] / df["eval_time_seconds"]
        dfs.append(df)

    # Merge all files
    data = pd.concat(dfs, ignore_index=True)

    # Melt data into long format
    long = data.melt(
        id_vars=["benchmark", "population", "device"],
        value_vars=["evaluations", "eval_time_seconds", "ratio"],
        var_name="metric",
        value_name="value",
    )

    metric_names = {
        "evaluations": "Evaluations",
        "eval_time_seconds": "Evaluation Time (s)",
        "ratio": "Evaluations per Second",
    }

    long["metric_label"] = long["metric"].map(metric_names)

    # Plot
    g = sns.catplot(
        data=long,
        kind="bar",
        x="population",
        y="value",
        hue="device",
        row="benchmark",
        col="metric_label",
        sharey=False,
        margin_titles=True,
    )

    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    g.set_axis_labels("Population Size", "Value")
    g.set(yscale="log")

    g.tight_layout()
    g.savefig("eval.png")


def main():
    run_date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    results_dir = Path("experiments/results") / run_date
    results_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = [
        # "first_principles_kepler",  # 6 obs, 1 feat
        "nikuradse_2",  # 362 obs, 1 feat
        # "feynman_I_6_2a",  # 100_000 obs, 1 feat
        # "542_pollution",  # 60 obs, 15 feat
        # "503_wind",  # 6574 obs, 14 feat
        # "1191_BNG_pbc",  # 1_000_000 obs, 18 feat
        # "505_tecator",  # 240 obs, 124 feat
    ]
    pop_sizes = [256, 512, 1024, 2048]

    run_experiments(benchmarks, pop_sizes, results_dir, include_cpu=True)

    # plot_error()
    # plot_convergence()


if __name__ == "__main__":
    main()
