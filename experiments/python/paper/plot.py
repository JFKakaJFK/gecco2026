import os
import pathlib

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot(var: str, dir: pathlib.Path):
    db_path = dir / "all_results.duckdb"
    plot_dir = dir / "plots"

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    conn = duckdb.connect(db_path)

    df = conn.execute(f"""
    SELECT
        dataset,
        template_depth,
        population_size,
        algorithm,
        run,
        {var}
    FROM results
    """).df()

    dataset_order = [
        "daily_demand",
        "auto_mpg",
        "california_housing",
        "feynman",
    ]

    g = sns.relplot(
        data=df,
        x="population_size",
        y=var,
        hue="algorithm",
        row="dataset",
        col="template_depth",
        kind="line",
        errorbar=("pi", 50),
        estimator="median",
        row_order=dataset_order,
        marker="o",
        facet_kws={"sharey": "row"},
    )

    for ax in g.axes.flat:
        ax.set_xscale("log", base=2)

    g.set(yscale="log")
    g.set_axis_labels("Population Size", var)
    plt.savefig(plot_dir / f"{var}_test.svg")
    plt.savefig(plot_dir / f"{var}_test.png")
    plt.savefig(plot_dir / f"{var}_test.eps")


if __name__ == "__main__":
    plot("mse", pathlib.Path("results"))
    # plot("evaluation", pathlib.Path("results"))
