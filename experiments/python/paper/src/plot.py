import os
import pathlib

import duckdb
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocess import interpolate_onto_time_grid

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


def plot_experiment_2(dir: pathlib.Path, var="mse"):
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
        facet_kws={"sharey": var == "evaluations", "margin_titles": True},
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


def plot_experiment_6(dir: pathlib.Path, var: str = "mse"):
    db_path = dir / "all_results.duckdb"
    plot_dir = dir / "plots"

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    conn = duckdb.connect(db_path)

    df = conn.execute(f"""
    SELECT
        algorithm,
        cutoff,
        dataset,
        {var},
        population_size,
        template_depth,
        run
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

    cutoff_order = ["log_2", "log_10"]
    palette = sns.color_palette("colorblind", n_colors=len(cutoff_order))

    var_label = VAR_LABELS.get(var, var)

    g = sns.relplot(
        data=df,
        x="population_size",
        y=var,
        hue="cutoff",
        hue_order=cutoff_order,
        col="dataset",
        row="template_depth",
        kind="line",
        errorbar=("pi", 50),
        estimator="median",
        col_order=dataset_order,
        marker="o",
        palette=palette,
        facet_kws={"sharey": False, "margin_titles": True},
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
        ncol=len(cutoff_order),
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


SYNTHETIC_DATASET_LABELS = {
    "1_addition": "Addition",
    "2_division": "Division",
    "3_subtraction": "Subtraction",
    "4_multiplication": "Multiplication",
    "5_square": "Square",
}


def plot_experiment_3(dir: pathlib.Path):
    db_path = dir / "all_results.duckdb"
    plot_dir = dir / "plots"

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    conn = duckdb.connect(db_path)

    df = conn.execute("""
    SELECT
        log_config,
        dataset,
        total_time_seconds,
        nmse,
        fold,
        template_depth,
        population_size,
    FROM results
    """).df()

    df["dataset"] = df["dataset"].map(SYNTHETIC_DATASET_LABELS).fillna(df["dataset"])

    # Build row labels from (log_config, template_depth, population_size)
    pop_per_config = df.groupby("log_config")["population_size"].first()

    def _row_label(log_config: str, template_depth: int) -> str:
        log_base = log_config.split("_")[1]
        pop = pop_per_config[log_config]
        pop_str = str(int(pop))
        return f"$C = \\log_{{{log_base}}}$\n$P = {pop_str}$\n$D = {template_depth}$"

    df["row_label"] = df.apply(
        lambda r: _row_label(r["log_config"], r["template_depth"]), axis=1
    )

    log_config_order = sorted(df["log_config"].unique())
    depth_order = sorted(df["template_depth"].unique())
    row_meta = [
        {"log_config": lc, "depth": d, "label": _row_label(lc, d)}
        for lc in log_config_order
        for d in depth_order
    ]
    row_order = [m["label"] for m in row_meta]
    dataset_order = [
        SYNTHETIC_DATASET_LABELS[d]
        for d in [
            "1_addition",
            "2_division",
            "3_subtraction",
            "4_multiplication",
            "5_square",
        ]
        if SYNTHETIC_DATASET_LABELS[d] in df["dataset"].unique()
    ]

    df_interp = interpolate_onto_time_grid(df)
    df_interp["row_label"] = df_interp.apply(
        lambda r: _row_label(r["log_config"], r["template_depth"]), axis=1
    )

    color = sns.color_palette("colorblind")[0]

    g = sns.FacetGrid(
        data=df,
        col="dataset",
        row="row_label",
        col_order=dataset_order,
        row_order=row_order,
        sharey=False,
        sharex=True,
        margin_titles=True,
        height=2.5,
        aspect=1.25,
    )

    for (row_val, col_val), ax in g.axes_dict.items():
        raw = df[(df["row_label"] == row_val) & (df["dataset"] == col_val)]

        final_nmse = (
            raw.sort_values("total_time_seconds").groupby("fold")["nmse"].last()
        )
        best_fold = final_nmse.idxmin()
        worst_fold = final_nmse.idxmax()

        for fold, fold_data in raw.groupby("fold"):
            fold_data = fold_data.sort_values("total_time_seconds")
            if fold == best_fold:
                c, alpha, lw, zorder = "green", 0.9, 1.2, 3
            elif fold == worst_fold:
                c, alpha, lw, zorder = "red", 0.9, 1.2, 3
            else:
                c, alpha, lw, zorder = color, 0.2, 0.8, 1
            ax.plot(
                fold_data["total_time_seconds"],
                fold_data["nmse"],
                color=c,
                alpha=alpha,
                linewidth=lw,
                zorder=zorder,
            )

        interp = df_interp[
            (df_interp["row_label"] == row_val) & (df_interp["dataset"] == col_val)
        ]
        grouped = interp.groupby("time_grid_seconds")["nmse"]
        median = grouped.median()
        q25 = grouped.quantile(0.25)
        q75 = grouped.quantile(0.75)

        ax.fill_between(median.index, q25, q75, color=color, alpha=0.25)
        ax.plot(median.index, median, color=color, linewidth=1.5)

    SOLVED_THRESHOLD = 10e-6

    g.set_titles(col_template="{col_name}", row_template="")

    # Per-row D label in axes coords — unaffected by later figure resize
    for i, meta in enumerate(row_meta):
        g.axes[i, 0].annotate(
            f"$D = {meta['depth']}$",
            xy=(-0.075, 0.5),
            xycoords="axes fraction",
            xytext=(-30, 0),
            textcoords="offset points",
            ha="center",
            va="center",
            rotation=90,
        )

    for ax in g.axes.flat:
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)

    g.set_axis_labels("", "")

    # Compute solved counts per (log_config, template_depth, dataset)
    min_nmse = df.groupby(["log_config", "template_depth", "dataset", "fold"])[
        "nmse"
    ].min()
    solved = (min_nmse <= SOLVED_THRESHOLD).reset_index(name="solved")
    counts = (
        solved.groupby(["log_config", "template_depth", "dataset"])["solved"]
        .sum()
        .reset_index()
    )
    n_folds_total = df["fold"].nunique()

    # Draw to get axis positions before resize
    g.figure.canvas.draw()
    col_positions_pre = [
        g.axes[-1, j].get_position() for j in range(len(dataset_order))
    ]

    # Resize figure to add an extra row at the bottom
    fig = g.figure
    w_in, h_in = fig.get_size_inches()
    extra_h_in = h_in * 0.3
    h_new_in = h_in + extra_h_in
    fig.set_size_inches(w_in, h_new_in)

    y_scale = h_in / h_new_in
    y_shift = extra_h_in / h_new_in
    for ax in fig.get_axes():
        pos = ax.get_position()
        ax.set_position(
            [pos.x0, pos.y0 * y_scale + y_shift, pos.width, pos.height * y_scale]
        )

    # Add success axes aligned with each dataset column
    gap = 0.05
    new_row_h = y_shift - gap * 2
    setup_palette = sns.color_palette("colorblind", n_colors=len(row_meta))
    bar_width = 0.8 / len(row_meta)
    x_centers = [
        (k - (len(row_meta) - 1) / 2) * bar_width for k in range(len(row_meta))
    ]

    success_axes = []
    for j, col_pos in enumerate(col_positions_pre):
        ax = fig.add_axes([col_pos.x0, gap, col_pos.width, new_row_h])
        success_axes.append(ax)

        for k, meta in enumerate(row_meta):
            subset = counts[
                (counts["log_config"] == meta["log_config"])
                & (counts["template_depth"] == meta["depth"])
                & (counts["dataset"] == dataset_order[j])
            ]
            count = int(subset["solved"].iloc[0]) if len(subset) > 0 else 0
            ax.barh(x_centers[k], count, height=bar_width * 0.9, color=setup_palette[k])

        ax.set_ylim(-0.5, 0.5)
        ax.set_xlim(0, n_folds_total)
        ax.set_xticks(range(0, n_folds_total + 1, 3))
        ax.set_yticks([])
        ax.grid(True, axis="x", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)

    # All figure-level text added after resize so coordinates are correct
    fig.canvas.draw()

    # C/P spanning labels for the main grid rows
    for lc in log_config_order:
        indices = [i for i, m in enumerate(row_meta) if m["log_config"] == lc]
        positions = [g.axes[i, 0].get_position() for i in indices]
        y_mid = sum((p.y0 + p.y1) / 2 for p in positions) / len(positions)
        x_left = positions[0].x0
        parts = row_meta[indices[0]]["label"].split("\n")
        fig.text(
            x_left - 0.06,
            y_mid,
            f"{parts[0]}\n{parts[1]}",
            ha="center",
            va="center",
            rotation=90,
            multialignment="center",
            fontsize=plt.rcParams["font.size"],
        )

    # Row label for the success row
    success_y = (
        success_axes[0].get_position().y0 + success_axes[0].get_position().y1
    ) / 2
    x_left = success_axes[0].get_position().x0
    fig.text(
        x_left - 0.06,
        success_y,
        f"Solved (NMSE $\\leq$ 10e-6)",
        ha="center",
        va="center",
        rotation=90,
        multialignment="center",
        fontsize=plt.rcParams["font.size"],
    )

    main_grid_positions = [ax.get_position() for ax in g.axes.flat]
    bar_positions = [ax.get_position() for ax in success_axes]
    all_positions = main_grid_positions + bar_positions
    grid_left = min(p.x0 for p in all_positions)
    grid_right = max(p.x1 for p in all_positions)
    x_center = (grid_left + grid_right) / 2

    main_grid_bottom = min(p.y0 for p in main_grid_positions)
    bar_top = max(p.y1 for p in bar_positions)
    bar_bottom = min(p.y0 for p in bar_positions)
    between_y = (main_grid_bottom + bar_top) / 2

    fig.supxlabel("Time [s]", x=x_center, y=between_y + 0.005)
    fig.supylabel("NMSE", x=-0.05)

    # NMSE legend between main grid and bar plots
    fold_line = plt.Line2D([], [], color=color, alpha=0.3, linewidth=0.8, label="Fold")
    best_line = plt.Line2D([], [], color="green", linewidth=1.2, label="Best fold")
    worst_line = plt.Line2D([], [], color="red", linewidth=1.2, label="Worst fold")
    median_line = plt.Line2D([], [], color=color, linewidth=1.5, label="Median")
    iqr_patch = plt.matplotlib.patches.Patch(
        color=color, alpha=0.25, label="IQR (25-75%)"
    )
    fig.legend(
        handles=[fold_line, best_line, worst_line, median_line, iqr_patch],
        loc="center",
        bbox_to_anchor=(x_center, between_y - 0.01),
        ncol=5,
        title=None,
    )

    # Setup legend below the bar plots
    setup_handles = [
        plt.matplotlib.patches.Patch(
            color=setup_palette[k],
            label=(
                f"$\\log_{{{row_meta[k]['log_config'].split('_')[1]}}}$"
                f", $D={row_meta[k]['depth']}$"
            ),
        )
        for k in range(len(row_meta))
    ]
    fig.legend(
        handles=setup_handles,
        loc="center",
        bbox_to_anchor=(x_center, bar_bottom / 2),
        ncol=len(row_meta),
        title=None,
    )

    plt.savefig(plot_dir / "nmse_over_time.svg", bbox_inches="tight")
    plt.savefig(plot_dir / "nmse_over_time.png", bbox_inches="tight", dpi=150)


if __name__ == "__main__":
    # plot("mse", pathlib.Path("results/experiment_2"))
    # plot("evaluations", pathlib.Path("results/experiment_2"), sharey="row")
    plot_experiment_1(pathlib.Path("results/experiment_1"))
