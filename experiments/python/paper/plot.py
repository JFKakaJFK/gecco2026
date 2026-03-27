import os
import pathlib

import duckdb
import matplotlib.pyplot as plt
import seaborn as sns

ALGORITHM_LABELS = {
    "cpu_results": "GP-GOMEA (CPU)",
    "gpu_results": "GP-GOMEA (GPU)",
    "evogp_results": "EvoGP",
    "operon_results": "Operon",
    "kozax_results": "Kozax",
}

DATASET_LABELS = {
    "daily_demand": "Daily Demand",
    "auto_mpg": "Auto MPG",
    "california_housing": "California Housing",
    "feynman": "Feynman",
}

VAR_LABELS = {
    "mse": "MSE",
    "evaluations": "#Evaluations / Minute",
}

TIME_BUDGET_MINUTES = {
    "GP-GOMEA (CPU)": 60,
    "GP-GOMEA (GPU)": 10,
    "EvoGP": 10,
    "Operon": 60,
    "Kozax": 10,
}


def plot(var: str, dir: pathlib.Path, sharey: str = "col"):
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

    df["algorithm"] = df["algorithm"].map(ALGORITHM_LABELS).fillna(df["algorithm"])
    df["dataset"] = df["dataset"].map(DATASET_LABELS).fillna(df["dataset"])

    if var == "evaluations":
        df["evaluations"] /= df["algorithm"].map(TIME_BUDGET_MINUTES)

    dataset_order = [
        DATASET_LABELS[d]
        for d in [
            "daily_demand",
            "auto_mpg",
            "california_housing",
            "feynman",
        ]
    ]

    algorithm_order = [
        v for v in ALGORITHM_LABELS.values() if v in df["algorithm"].unique()
    ]

    var_label = VAR_LABELS.get(var, var)

    palette = sns.color_palette("colorblind", n_colors=len(algorithm_order))

    g = sns.relplot(
        data=df,
        x="population_size",
        y=var,
        hue="algorithm",
        hue_order=algorithm_order,
        col="dataset",
        row="template_depth",
        kind="line",
        errorbar=("pi", 50),
        estimator="median",
        col_order=dataset_order,
        marker="o",
        palette=palette,
        facet_kws={"sharey": sharey, "margin_titles": True},
        height=2.5,
        aspect=1.0,
    )

    g.set_titles(col_template="{col_name}", row_template="")
    for i, row_name in enumerate(g.row_names):
        g.axes[i, 0].annotate(
            f"{var_label}\nDepth {row_name}",
            xy=(0, 0.5),
            xycoords="axes fraction",
            xytext=(-50, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            rotation=90,
            multialignment="center",
        )

    for ax in g.axes.flat:
        ax.set_xscale("log", base=2)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)

    g.set(yscale="log")
    g.set_axis_labels("", "")

    sns.move_legend(
        g,
        loc="lower center",
        bbox_to_anchor=(0.42, -0.05),
        ncol=len(algorithm_order),
        title=None,
    )

    g.figure.canvas.draw()
    positions = [ax.get_position() for ax in g.axes.flat]
    grid_left = min(p.x0 for p in positions)
    grid_right = max(p.x1 for p in positions)
    g.figure.supxlabel("Population Size", x=(grid_left + grid_right) / 2, y=0.025)

    plt.savefig(plot_dir / f"{var}.svg", bbox_inches="tight")
    plt.savefig(plot_dir / f"{var}.png", bbox_inches="tight", dpi=150)
    plt.savefig(plot_dir / f"{var}.eps", bbox_inches="tight")


def plot_experiment_1(dir: pathlib.Path):
    db_path = dir / "all_results.duckdb"
    plot_dir = dir / "plots"

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    conn = duckdb.connect(db_path)
    df = conn.execute("SELECT algorithm, run, mse FROM results").df()

    algorithm_order = ["GP-GOMEA (original)", "GP-GOMEA (new)"]
    palette = sns.color_palette("colorblind", n_colors=2)

    fig, ax = plt.subplots(figsize=(4, 4))

    sns.boxplot(
        data=df,
        x="algorithm",
        y="mse",
        hue="algorithm",
        order=algorithm_order,
        hue_order=algorithm_order,
        palette=palette,
        width=0.4,
        flierprops={"marker": ""},
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x="algorithm",
        y="mse",
        hue="algorithm",
        order=algorithm_order,
        hue_order=algorithm_order,
        palette=palette,
        size=4,
        alpha=0.5,
        jitter=True,
        legend=False,
        ax=ax,
    )

    ax.set_yscale("log")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("MSE")
    ax.set_title("Feynman I.9.18")
    ax.grid(True, which="both", axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(plot_dir / "mse.svg", bbox_inches="tight")
    fig.savefig(plot_dir / "mse.png", bbox_inches="tight", dpi=150)
    fig.savefig(plot_dir / "mse.eps", bbox_inches="tight")


if __name__ == "__main__":
    plot("mse", pathlib.Path("results/experiment_2"))
    plot("evaluations", pathlib.Path("results/experiment_2"), sharey="row")
    plot_experiment_1(pathlib.Path("results/experiment_1"))
