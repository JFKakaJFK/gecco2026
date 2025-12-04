import csv
import pygom.gp as gp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from pmlb import fetch_data
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def run_single_experiment(X_train, X_test, y_train, y_test, pop_size, gpu_accelerated, test):
    est = gp.SymbolicRegressor(
        gpu_accelerated=gpu_accelerated,
        linear_scaling=False,
        budget_kwargs=dict(
            max_time_seconds=300,
        ),
        ims_kwargs=dict(initial_population_size=pop_size, max_num_populations=1),
        rv_kwargs=dict(enabled=False),  # disable rv optimization
        discrete_model_kwargs=dict(
            merge_continuous=False,
            num_continuous_bins=25,
            normalize_initial_linkage_bias=True,
        ),
        random_state=42,
        tracking_kwargs=dict(
            logpath=f"results/{test}.csv",
        ),
    )

    print("Fit")
    est.fit(X_train, y_train)

    print("r2")
    r2_train = r2_score(y_train, est.predict(X_train))
    r2_test = r2_score(y_test, est.predict(X_test))

    print("mse")
    mse_train = mean_squared_error(y_train, est.predict(X_train))
    mse_test = mean_squared_error(y_test, est.predict(X_test))

    return [r2_train, r2_test, mse_train, mse_test]


def run_experiments(benchmarks, pop_sizes):
    results = [["benchmark", "pop_size", "device", "r2_train", "r2_test", "mse_train", "mse_test"]]

    for b in benchmarks:
        # Get benchmark data and split
        X, y = fetch_data(b, return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        for p in pop_sizes:
            # print(f"Starting benchmark: {b} with population size: {p}")
            for acc in [True, False]:
                test_name = f"{b}_{"gpu" if acc else "cpu"}_{p}"
                print(f"Starting experiment: {test_name}")
                temp = run_single_experiment(X_train, X_test, y_train, y_test, p, acc, test_name)
                results.append([b, p, "gpu" if acc else "cpu"] + temp)

    with open("results/error_result.csv", 'w', newline='') as csvfile:
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
        value_name="r2"
    )

    # Plot: device as hue, optional split as facet
    r2 = sns.catplot(
        data=error_melted,
        kind="bar",
        x="benchmark",
        y="r2",
        hue="device",       # CPU vs GPU as main highlight
        col="split",        # optional: separate train/test columns
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
    path = Path("results/")

    dfs = []

    for file in path.glob("*.csv"):
        df = pd.read_csv(file)

        # Extract metadata (you can adjust this to your filenames)
        # Example filename: "benchmark1_pop100_cpu.csv"
        parts = file.stem.split("_")
        if len(parts) < 3:
            continue
        benchmark = "_".join(parts[:2])
        device = parts [2]
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
        value_name="value"
    )

    # Optional: nicer labels
    benchmark_names = {
        "210_cloud": "210_cloud",
        "505_tecator": "505_tecator",
        "294_satellite-image": "294_satellite_image"
    }

    metric_names = {
        "evaluations": "Evaluations",
        "eval_time_seconds": "Evaluation Time (s)",
        "ratio": "Evaluations per Second"
    }

    long["benchmark_label"] = long["benchmark"].map(benchmark_names)
    long["metric_label"] = long["metric"].map(metric_names)

    # Plot
    g = sns.catplot(
        data=long,
        kind="bar",
        x="population",
        y="value",
        hue="device",
        row="benchmark_label",
        col="metric_label",
        sharey=False,
        margin_titles=True
    )

    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    g.set_axis_labels("Population Size", "Value")
    g.set(yscale="log")

    g.tight_layout()
    g.savefig("eval.png")

def main():
    benchmarks = ["210_cloud", "505_tecator", "294_satellite_image"]
    pop_sizes = [256, 512, 1024, 2048]

    # run_experiments(benchmarks, pop_sizes)
    
    plot_error()
    plot_convergence()

    

if __name__ == "__main__":
    main()