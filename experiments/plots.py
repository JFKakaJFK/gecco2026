import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path

def plot_cpu_gpu(datetime, dataset_sizes):
    path = Path("experiments/results/") / datetime

    records = []

    # Get information from result files
    for file in path.glob("*-*-pop*-iter*.csv"):
        df = pd.read_csv(file)

        # Extract metadata
        dataset, device, population, iteration = file.stem.split("-")

        population = int(population.replace("pop", ""))
        iteration = int(iteration.replace("iter", ""))

        # Select converged row
        converged = df[df["status"] == "Converged"]
        if converged.empty:
            continue

        row = converged.iloc[0]

        evals_per_sec = row["evaluations"] / row["eval_time_seconds"]

        records.append({
            "dataset": dataset,
            "device": device,
            "population": population,
            "iteration": iteration,
            "evals_per_sec": evals_per_sec,
        })

    df_all = pd.DataFrame(records)

    # Faceted grouped bar plot of evals per sec (CPU vs GPU)
    g = sns.catplot(
        data=df_all,
        x="population",
        y="evals_per_sec",
        hue="device",
        col="dataset",
        kind="bar",
        height=4,
        aspect=1,
        errorbar=("ci", 95),
    )

    g.set_axis_labels("Population Size", "Evaluations per Second")
    g.set_titles("Dataset: {col_name}")
    g._legend.set_title("Device")

    plt.yscale("log")

    plt.tight_layout()
    plt.savefig(f"{path}/evals_per_second_barplot.png", dpi=300)

    # Speedup plot (GPU / CPU)
    df_mean = (
        df_all
        .groupby(["dataset", "device", "population"], as_index=False)
        .agg(evals_per_sec=("evals_per_sec", "mean"))
    )

    cpu = df_mean[df_mean["device"] == "cpu"]
    gpu = df_mean[df_mean["device"] == "gpu"]

    df_speedup = pd.merge(
        cpu,
        gpu,
        on=["dataset", "population"],
        suffixes=("_cpu", "_gpu")
    )

    df_speedup["speedup"] = (
        df_speedup["evals_per_sec_gpu"] /
        df_speedup["evals_per_sec_cpu"]
    )

    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df_speedup,
        x="population",
        y="speedup",
        hue="dataset",
        marker="o"
    )

    plt.xscale("log", base=2)
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("Population Size")
    plt.ylabel("Speedup (GPU / CPU)")
    plt.title("Acceleration Speedup vs Population Size")
    plt.tight_layout()
    plt.savefig(f"{path}/speedup_plot.png", dpi=300)


def plot_gpu_versions(datetime: list[str]):
    records = []

    for dt in datetime:
        path = Path("experiments/results/") / dt

        for file in path.glob("*-*-pop*-obs*-iter*.csv"):
            df = pd.read_csv(file)

            # Extract metadata
            _, version, population, observations, iterations = file.stem.split("-")

            population = int(population.replace("pop", ""))
            observations = int(observations.replace("obs", ""))
            iterations = int(iterations.replace("iter", ""))

            converged = df[df["status"] == "Converged"]
            if converged.empty:
                continue

            row = converged.iloc[0]
            threads_per_sec =  (row["evaluations"] * observations * population) / row["eval_time_seconds"]

            records.append({
                "version": version,
                "observations": observations,
                "iteration": iterations,
                "threads_per_sec": threads_per_sec
            })

    df_all = pd.DataFrame(records)

    df_mean = (
        df_all
        .groupby(["version", "observations"], as_index=False)
        .agg(threads_per_sec=("threads_per_sec", "mean"))
    )

    sns.scatterplot(
        data=df_all,
        x="observations",
        y="threads_per_sec",
        hue="version",
        alpha=0.7
    )

    for version in df_mean["version"].unique():
        subset = df_mean[df_mean["version"] == version]
        sns.regplot(
            data=subset,
            x="observations",
            y="threads_per_sec",
            scatter=False,
            logx=True,
            label=f"{version} trend"
        )

    plt.xscale("log")
    plt.xlabel("Number of Observations (log scale)")
    plt.ylabel("Threads per Second")
    plt.title("Kernel Versions vs Number of Observations")
    plt.tight_layout()
    plt.savefig("gpu_scaling_log_scatter_regression.png", dpi=300)

    # Pivot baseline
    baseline = df_mean[df_mean["version"] == "baseline"][["observations", "threads_per_sec"]]
    baseline = baseline.rename(columns={"threads_per_sec": "baseline_threads"})

    # Merge baseline back
    df_relative = pd.merge(df_mean, baseline, on="observations")
    df_relative["relative_perf"] = df_relative["threads_per_sec"] / df_relative["baseline_threads"]

    # Plot
    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=df_relative,
        x="observations",
        y="relative_perf",
        hue="version",
        marker="o"
    )

    plt.xscale("log")
    plt.xlabel("Number of Observations (log scale)")
    plt.ylabel("Relative Performance vs Baseline")
    plt.title("GPU Versions Relative Performance")
    plt.axhline(1.0, linestyle="--", color="gray", label="Baseline")
    plt.legend()
    plt.tight_layout()
    plt.savefig("gpu_relative_performance.png", dpi=300)
    # plt.show()