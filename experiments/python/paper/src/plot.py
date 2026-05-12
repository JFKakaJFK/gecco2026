import os
import pathlib

import duckdb
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocess import interpolate_onto_time_grid

ALGORITHM_LABELS = {
    "cpu_results": "GP-GOMEA (CPU)",
    "gpu_results": "GP-GOMEA (GPU)",
    "evogp_results": "EvoGP (GPU)",
    "kozax_results": "Kozax (GPU)",
    "operon_results": "Operon (CPU)",
}

DATASET_LABELS = {
    "daily_demand": "Daily Demand",
    "auto_mpg": "Auto MPG",
    "california_housing": "California Housing",
    "feynman": "Feynman I.9.18",
    "1_addition": "Addition",
    "2_division": "Division",
    "3_subtraction": "Subtraction",
    "4_multiplication": "Multiplication",
    "5_square": "Squaring",
    "feynman_I_8_14": "Distance",
    "feynman_I_11_19": "Dot Product",
    "feynman_I_9_18": "Gravity",
    "modular_1": "Modular 1",
    "modular_2": "Modular 2",
    "modular_3": "Modular 3",
    "modular_4": "Modular 4",
    "modular_5": "Modular 5",
}

DEVICE_LABELS = {
    "cpu": "CPU",
    "gpu": "GPU",
}

DEVICE_TIME_BUDGET_MINUTES = {
    "CPU": 60,
    "GPU": 10,
}

VAR_LABELS = {
    "mse": "Training MSE",
    "mse_val": "Validation MSE",
    "nmse": "Training NMSE",
    "nmse_val": "Validation NMSE",
    "r2": "Training R²",
    "r2_val": "Validation R²",
    "evaluations": "#Evaluations / Minute",
}

TIME_BUDGET_MINUTES = {
    "GP-GOMEA (CPU)": 60,
    "GP-GOMEA (GPU)": 10,
    "EvoGP (GPU)": 10,
    "Operon (CPU)": 60,
    "Kozax (GPU)": 10,
}

KERNEL_LABELS = {
    "single_kernel_inplace": "Single-Block",
    "hybrid": "Dynamic-Block",
}

TEMPLATE_DEPTH_LABELS = {
    4: "4x4",
    7: "7x1",
}


def plot_experiment_2(dir: pathlib.Path, var="mse", exclude: list[str] | None = None):
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

    if exclude:
        df = df[~df["algorithm"].isin(exclude)]

    if var == "evaluations":
        df["evaluations"] /= df["algorithm"].map(TIME_BUDGET_MINUTES)

    # Replace exact zeros with a small positive value so they remain visible
    # on the log scale (zero is undefined on log axes).
    if var in ("mse", "mse_val"):
        df[var] = df[var].replace(0, 1e-28)

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
        facet_kws={
            "sharey": True if var == "evaluations" else "col",
            "margin_titles": True,
        },
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

    suffix = (
        ("_no_" + "_".join(e.replace(" ", "_") for e in exclude)) if exclude else ""
    )
    plt.savefig(plot_dir / f"{var}{suffix}.svg", bbox_inches="tight")
    plt.savefig(plot_dir / f"{var}{suffix}.png", bbox_inches="tight", dpi=150)


def plot_experiment_2_gomea(dir: pathlib.Path, var="mse"):
    """Like plot_experiment_2 but restricted to CPU/GPU GP-GOMEA only.
    Both template depths are shown in a single row; algorithm x depth combinations
    are distinguished by colour using tab20 pairs (one pair per algorithm,
    one shade per depth)."""
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

    gomea_algorithms = ["GP-GOMEA (CPU)", "GP-GOMEA (GPU)"]
    df = df[df["algorithm"].isin(gomea_algorithms)]

    if var == "evaluations":
        df["evaluations"] /= df["algorithm"].map(TIME_BUDGET_MINUTES)

    if var in ("mse", "mse_val"):
        df[var] = df[var].replace(0, 1e-28)

    dataset_order = [
        DATASET_LABELS[d]
        for d in [
            "daily_demand",
            "auto_mpg",
            "california_housing",
            "feynman",
        ]
    ]

    algorithm_order = [v for v in gomea_algorithms if v in df["algorithm"].unique()]
    depth_order = sorted(df["template_depth"].unique())

    # Combined hue: one tab20 pair per algorithm, one shade per depth
    _tab20 = sns.color_palette("tab20", n_colors=20)
    palette = {
        f"{alg} (D={d})": _tab20[i * 2 + j]
        for i, alg in enumerate(algorithm_order)
        for j, d in enumerate(depth_order)
    }
    df["hue_label"] = df["algorithm"] + " (D=" + df["template_depth"].astype(str) + ")"
    hue_order = [f"{alg} (D={d})" for alg in algorithm_order for d in depth_order]

    var_label = VAR_LABELS.get(var, var)

    g = sns.relplot(
        data=df,
        x="population_size",
        y=var,
        hue="hue_label",
        hue_order=hue_order,
        col="dataset",
        col_order=dataset_order,
        kind="line",
        errorbar=("pi", 50),
        estimator="median",
        marker="o",
        palette=palette,
        facet_kws={
            "sharey": True if var == "evaluations" else "col",
            "margin_titles": True,
        },
        height=2.5,
        aspect=1.0,
    )

    g.set_titles(col_template="{col_name}")
    g.axes[0, 0].annotate(
        var_label,
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

    if g.legend:
        sns.move_legend(
            g,
            loc="lower center",
            bbox_to_anchor=(0.42, -0.05),
            ncol=len(hue_order),
            title=None,
        )

    g.figure.canvas.draw()
    positions = [ax.get_position() for ax in g.axes.flat]
    grid_left = min(p.x0 for p in positions)
    grid_right = max(p.x1 for p in positions)
    g.figure.supxlabel("Population Size", x=(grid_left + grid_right) / 2, y=0.065)

    plt.savefig(plot_dir / f"{var}_gomea.svg", bbox_inches="tight")
    plt.savefig(plot_dir / f"{var}_gomea.png", bbox_inches="tight", dpi=150)


def plot_experiment_3(dir: pathlib.Path, nmse_col: str = "nmse"):
    db_path = dir / "all_results.duckdb"
    plot_dir = dir / "plots"

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    conn = duckdb.connect(db_path)

    df = conn.execute(f"""
    SELECT
        log_config,
        dataset,
        total_time_seconds,
        {nmse_col} AS nmse,
        nmse_val,
        fold,
        template_depth,
        population_size,
    FROM results
    WHERE {nmse_col} IS NOT NULL
    """).df()

    df["dataset"] = df["dataset"].map(DATASET_LABELS).fillna(df["dataset"])

    # Build row labels from (log_config, template_depth, population_size)
    pop_per_config = df.groupby("log_config")["population_size"].first()

    def _row_label(log_config: str, template_depth: int) -> str:
        log_base = log_config.split("_")[1]
        pop = pop_per_config[log_config]
        pop_str = f"{int(pop):,}"
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
        DATASET_LABELS[d]
        for d in [
            "1_addition",
            "2_division",
            "3_subtraction",
            "4_multiplication",
            "5_square",
        ]
        if DATASET_LABELS[d] in df["dataset"].unique()
    ]

    df_interp = interpolate_onto_time_grid(df)
    df_interp["row_label"] = df_interp.apply(
        lambda r: _row_label(r["log_config"], r["template_depth"]), axis=1
    )

    # Tab20 assigns pairs of shades; one pair per log_config, one shade per depth
    _tab20 = sns.color_palette("tab20", n_colors=20)
    setting_colors = {
        (lc, d): _tab20[i * 2 + j]
        for i, lc in enumerate(log_config_order)
        for j, d in enumerate(depth_order)
    }

    g = sns.FacetGrid(
        data=df,
        col="dataset",
        row="row_label",
        col_order=dataset_order,
        row_order=row_order,
        sharey=True,
        sharex=True,
        margin_titles=True,
        height=2,
        aspect=1.6,
    )
    g.figure.set_layout_engine("none")
    g.figure.subplots_adjust(hspace=0.05)

    def _plot_fold_lines(target_ax, raw, best_fold, color, linewidth_scale=1.0):
        for fold, fold_data in raw.groupby("fold"):
            fold_data = fold_data.sort_values("total_time_seconds")
            if fold == best_fold:
                ls, lw, alpha, zorder = "--", 1.5 * linewidth_scale, 0.9, 3
            else:
                ls, lw, alpha, zorder = ":", 0.8 * linewidth_scale, 0.6, 1
            target_ax.plot(
                fold_data["total_time_seconds"] / 60,
                fold_data["nmse"],
                color=color,
                alpha=alpha,
                linewidth=lw,
                linestyle=ls,
                zorder=zorder,
            )

    def _plot_median_band(target_ax, interp_data, color, linewidth_scale=1.0):
        grouped = interp_data.groupby("time_grid_seconds")["nmse"]
        median = grouped.median()
        q25 = grouped.quantile(0.25)
        q75 = grouped.quantile(0.75)
        time_minutes = median.index / 60
        target_ax.fill_between(time_minutes, q25, q75, color=color, alpha=0.25)
        target_ax.plot(
            time_minutes,
            median,
            color=color,
            linewidth=1.5 * linewidth_scale,
            linestyle="-",
        )
        return median

    for (row_val, col_val), ax in g.axes_dict.items():
        raw = df[(df["row_label"] == row_val) & (df["dataset"] == col_val)]

        final_nmse = (
            raw.sort_values("total_time_seconds").groupby("fold")["nmse"].last()
        )
        best_fold = final_nmse.idxmin()

        cell_meta = next(m for m in row_meta if m["label"] == row_val)
        cell_color = setting_colors[(cell_meta["log_config"], cell_meta["depth"])]

        _plot_fold_lines(ax, raw, best_fold, color=cell_color)

        interp = df_interp[
            (df_interp["row_label"] == row_val) & (df_interp["dataset"] == col_val)
        ]
        _plot_median_band(ax, interp, color=cell_color)

    SOLVED_THRESHOLD = 10e-6

    g.set_titles(
        col_template="{col_name}", row_template="", size=plt.rcParams["font.size"] + 5
    )

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
        ax.axhline(1e-6, color="black", linewidth=0.8, linestyle="-.", zorder=2)

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
    bar_width = 0.8 / len(row_meta)
    x_centers = [
        (k - (len(row_meta) - 1) / 2) * bar_width for k in range(len(row_meta))
    ]

    success_axes = []
    for j, col_pos in enumerate(col_positions_pre):
        ax = fig.add_axes([col_pos.x0, gap, col_pos.width, new_row_h])
        success_axes.append(ax)

        for k, meta in enumerate(reversed(row_meta)):
            subset = counts[
                (counts["log_config"] == meta["log_config"])
                & (counts["template_depth"] == meta["depth"])
                & (counts["dataset"] == dataset_order[j])
            ]
            count = int(subset["solved"].iloc[0]) if len(subset) > 0 else 0
            pct = count / n_folds_total * 100
            ax.barh(
                x_centers[k],
                pct,
                height=bar_width * 0.9,
                color=setting_colors[(meta["log_config"], meta["depth"])],
            )

        ax.set_ylim(-0.5, 0.5)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter())
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
        x_left - 0.02,
        success_y,
        "%Solved\n (NMSE $\\leq$ 10e-6)",
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
    main_grid_top = max(p.y1 for p in main_grid_positions)
    bar_top = max(p.y1 for p in bar_positions)
    between_y = (main_grid_bottom + bar_top) / 2

    fig.supxlabel("Time [min]", x=x_center, y=between_y)
    supylabel = "Training NMSE" if nmse_col == "nmse" else "Validation NMSE"
    fig.supylabel(supylabel, x=-0.05, y=(main_grid_bottom + main_grid_top) / 2)

    # NMSE legend between main grid and bar plots
    _legend_color = "gray"
    fold_line = plt.Line2D(
        [],
        [],
        color=_legend_color,
        alpha=0.6,
        linewidth=0.8,
        linestyle=":",
        label="Fold",
    )
    best_worst_line = plt.Line2D(
        [],
        [],
        color=_legend_color,
        linewidth=1.5,
        linestyle="--",
        label="Best fold",
    )
    median_line = plt.Line2D(
        [], [], color=_legend_color, linewidth=1.5, linestyle="-", label="Median"
    )
    iqr_patch = plt.matplotlib.patches.Patch(
        color=_legend_color, alpha=0.25, label="IQR (25-75%)"
    )
    setup_handles = [
        plt.matplotlib.patches.Patch(
            color=setting_colors[(row_meta[k]["log_config"], row_meta[k]["depth"])],
            label=(
                f"$C=\\log_{{{row_meta[k]['log_config'].split('_')[1]}}}$"
                f", $D={row_meta[k]['depth']}$"
            ),
        )
        for k in range(len(row_meta))
    ]
    base_handles = [fold_line, best_worst_line, median_line, iqr_patch]
    all_handles = base_handles + setup_handles
    fig.legend(
        handles=all_handles,
        loc="center",
        bbox_to_anchor=(x_center, between_y - 0.02),
        ncol=len(all_handles),
        title=None,
        fontsize=plt.rcParams["font.size"] + 2,
    )

    plt.savefig(plot_dir / f"{nmse_col}_over_time.svg", bbox_inches="tight")
    plt.savefig(plot_dir / f"{nmse_col}_over_time.png", bbox_inches="tight", dpi=150)


def _plot_experiment_4(
    dir: pathlib.Path,
    var: str,
    dataset_keys: list[str],
    filename_suffix: str,
) -> None:
    db_path = dir / "all_results.duckdb"
    plot_dir = dir / "plots"

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    conn = duckdb.connect(db_path)

    df = conn.execute(f"""
    SELECT
        device,
        dataset,
        template_depth,
        population_size,
        run,
        {var}
    FROM results
    """).df()

    # Replace exact zeros with a small positive value so they remain visible
    # on the log scale (zero is undefined on log axes).
    if var in ("mse", "mse_val", "nmse", "nmse_val"):
        df[var] = df[var].replace(0, 1e-31)

    df["device"] = df["device"].map(DEVICE_LABELS).fillna(df["device"])
    df["dataset"] = df["dataset"].map(DATASET_LABELS).fillna(df["dataset"])

    if var == "evaluations":
        df["evaluations"] /= df["device"].map(DEVICE_TIME_BUDGET_MINUTES)

    dataset_order = [DATASET_LABELS[d] for d in dataset_keys]

    df_subset = df[df["dataset"].isin(dataset_order)]
    device_order = [
        v for v in DEVICE_LABELS.values() if v in df_subset["device"].unique()
    ]
    depth_order = sorted(df_subset["template_depth"].unique())
    var_label = VAR_LABELS.get(var, var)

    _tab20 = sns.color_palette("tab20", n_colors=20)
    palette = {
        f"{dev} ({TEMPLATE_DEPTH_LABELS.get(d, d)})": _tab20[i * 2 + j]
        for i, dev in enumerate(device_order)
        for j, d in enumerate(depth_order)
    }
    df["hue_label"] = (
        df["device"]
        + " ("
        + df["template_depth"]
        .map(TEMPLATE_DEPTH_LABELS)
        .fillna(df["template_depth"].astype(str))
        + ")"
    )
    hue_order = [
        f"{dev} ({TEMPLATE_DEPTH_LABELS.get(d, d)})"
        for dev in device_order
        for d in depth_order
    ]

    g = sns.relplot(
        data=df,
        x="population_size",
        y=var,
        hue="hue_label",
        hue_order=hue_order,
        col="dataset",
        kind="line",
        errorbar=("pi", 50),
        estimator="median",
        col_order=dataset_order,
        marker="o",
        palette=palette,
        facet_kws={
            "sharey": True if (var == "evaluations") else "col",
            "margin_titles": True,
        },
        height=2.5,
        aspect=1.0,
    )

    g.set_titles(col_template="{col_name}")
    g.axes[0, 0].annotate(
        var_label,
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

    if var in ("r2", "r2_val"):
        g.set(ylim=(0.8, 1.0))
    else:
        g.set(yscale="log")
    g.set_axis_labels("", "")

    if g.legend:
        sns.move_legend(
            g,
            loc="lower center",
            bbox_to_anchor=(0.42, -0.05),
            ncol=len(hue_order),
            title=None,
        )

    g.figure.canvas.draw()
    positions = [ax.get_position() for ax in g.axes.flat]
    grid_left = min(p.x0 for p in positions)
    grid_right = max(p.x1 for p in positions)
    g.figure.supxlabel("Population Size", x=(grid_left + grid_right) / 2, y=0.065)

    plt.savefig(plot_dir / f"{var}{filename_suffix}.svg", bbox_inches="tight")
    plt.savefig(plot_dir / f"{var}{filename_suffix}.png", bbox_inches="tight", dpi=150)
    plt.close()


def _plot_experiment_4_time_to_threshold(
    dir: pathlib.Path,
    dataset_keys: list[str],
    filename_suffix: str,
    threshold: float = 10e-6,
) -> None:
    db_path = dir / "all_results.duckdb"
    plot_dir = dir / "plots"

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    conn = duckdb.connect(db_path)

    df = conn.execute(f"""
    SELECT
        device,
        dataset,
        template_depth,
        population_size,
        run,
        fold,
        MIN(total_time_seconds) AS time_to_threshold
    FROM results
    WHERE nmse_val <= {threshold}
    GROUP BY device, dataset, template_depth, population_size, run, fold
    """).df()

    df["device"] = df["device"].map(DEVICE_LABELS).fillna(df["device"])
    df["dataset"] = df["dataset"].map(DATASET_LABELS).fillna(df["dataset"])

    dataset_order = [DATASET_LABELS[d] for d in dataset_keys]
    df_subset = df[df["dataset"].isin(dataset_order)]
    device_order = [
        v for v in DEVICE_LABELS.values() if v in df_subset["device"].unique()
    ]
    depth_order = sorted(df_subset["template_depth"].unique())

    _tab20 = sns.color_palette("tab20", n_colors=20)
    palette = {
        f"{dev} ({TEMPLATE_DEPTH_LABELS.get(d, d)})": _tab20[i * 2 + j]
        for i, dev in enumerate(device_order)
        for j, d in enumerate(depth_order)
    }
    df["hue_label"] = (
        df["device"]
        + " ("
        + df["template_depth"]
        .map(TEMPLATE_DEPTH_LABELS)
        .fillna(df["template_depth"].astype(str))
        + ")"
    )
    hue_order = [
        f"{dev} ({TEMPLATE_DEPTH_LABELS.get(d, d)})"
        for dev in device_order
        for d in depth_order
    ]

    g = sns.relplot(
        data=df,
        x="population_size",
        y="time_to_threshold",
        hue="hue_label",
        hue_order=hue_order,
        col="dataset",
        kind="line",
        errorbar=("pi", 50),
        estimator="median",
        col_order=dataset_order,
        marker="o",
        palette=palette,
        facet_kws={
            "sharey": True,
            "margin_titles": True,
        },
        height=2.5,
        aspect=1.0,
    )

    g.set_titles(col_template="{col_name}")
    g.axes[0, 0].annotate(
        "Time to Threshold (s)",
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

    if g.legend:
        sns.move_legend(
            g,
            loc="lower center",
            bbox_to_anchor=(0.42, -0.05),
            ncol=len(hue_order),
            title=None,
        )

    g.figure.canvas.draw()
    positions = [ax.get_position() for ax in g.axes.flat]
    grid_left = min(p.x0 for p in positions)
    grid_right = max(p.x1 for p in positions)
    g.figure.supxlabel("Population Size", x=(grid_left + grid_right) / 2, y=0.065)

    plt.savefig(
        plot_dir / f"time_to_threshold{filename_suffix}.svg", bbox_inches="tight"
    )
    plt.savefig(
        plot_dir / f"time_to_threshold{filename_suffix}.png",
        bbox_inches="tight",
        dpi=150,
    )
    plt.close()


def _plot_experiment_4_success_rate(
    dir: pathlib.Path,
    dataset_keys: list[str],
    filename_suffix: str,
    threshold: float = 10e-6,
) -> None:
    db_path = dir / "all_results.duckdb"
    plot_dir = dir / "plots"

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    conn = duckdb.connect(db_path)

    df = conn.execute(f"""
    SELECT
        device,
        dataset,
        template_depth,
        population_size,
        run,
        100.0 * SUM(CASE WHEN min_nmse_val <= {threshold} THEN 1 ELSE 0 END)
            / COUNT(*) AS pct_reached
    FROM (
        SELECT device, dataset, template_depth, population_size, run, fold,
            MIN(nmse_val) AS min_nmse_val
        FROM results
        GROUP BY device, dataset, template_depth, population_size, run, fold
    )
    GROUP BY device, dataset, template_depth, population_size, run
    """).df()

    df["device"] = df["device"].map(DEVICE_LABELS).fillna(df["device"])
    df["dataset"] = df["dataset"].map(DATASET_LABELS).fillna(df["dataset"])

    dataset_order = [DATASET_LABELS[d] for d in dataset_keys]
    pop_size_order = sorted(df["population_size"].unique())
    device_order = [v for v in DEVICE_LABELS.values() if v in df["device"].unique()]
    depth_order = sorted(df["template_depth"].unique())
    config_order = [(dev, d) for dev in device_order for d in depth_order]

    pop_colors = dict(
        zip(
            pop_size_order,
            ["#7B2D8B", "#1F77B4", "#17BECF", "#2CA02C", "#BCBD22", "#FF7F0E"],
        )
    )
    config_hatches = dict(zip(config_order, ["", "\\\\", "xxx", "////"]))
    config_labels = {
        (dev, d): f"{dev} ({TEMPLATE_DEPTH_LABELS.get(d, d)})"
        for dev, d in config_order
    }

    bar_width = 0.15
    pop_gap = 0.08
    dataset_gap = 0.6
    n_configs = len(config_order)
    pop_slot = n_configs * bar_width + pop_gap
    dataset_width = len(pop_size_order) * pop_slot - pop_gap
    total_width = (
        len(dataset_order) * dataset_width + (len(dataset_order) - 1) * dataset_gap
    )

    fig, ax = plt.subplots(figsize=(max(12, total_width * 0.85), 5))

    dataset_centers = []
    for di, dataset in enumerate(dataset_order):
        group_x = di * (dataset_width + dataset_gap)
        dataset_centers.append(group_x + dataset_width / 2)

        df_ds = df[df["dataset"] == dataset]

        for pi, pop_size in enumerate(pop_size_order):
            pop_x = group_x + pi * pop_slot
            df_ps = df_ds[df_ds["population_size"] == pop_size]

            for ci, cfg in enumerate(config_order):
                dev, depth = cfg
                bar_x = pop_x + ci * bar_width + bar_width / 2
                row = df_ps[
                    (df_ps["device"] == dev) & (df_ps["template_depth"] == depth)
                ]
                height = float(row["pct_reached"].iloc[0]) if len(row) > 0 else 0.0
                ax.bar(
                    bar_x,
                    height,
                    width=bar_width * 0.92,
                    color=pop_colors[pop_size],
                    hatch=config_hatches[cfg],
                    edgecolor="black",
                    linewidth=0.5,
                )

    ax.set_xticks(dataset_centers)
    ax.set_xticklabels(dataset_order)
    ax.set_ylim(0, 100)
    ax.set_ylabel("%Solved (NMSE $\\leq 10^-6$)")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    pop_handles = [
        mpatches.Patch(
            facecolor=pop_colors[ps],
            edgecolor="black",
            linewidth=0.5,
            label=f"$2^{{{int(ps).bit_length() - 1}}}$",
        )
        for ps in pop_size_order
    ]
    config_handles = [
        mpatches.Patch(
            facecolor="white",
            hatch=config_hatches[cfg],
            edgecolor="black",
            linewidth=0.5,
            label=config_labels[cfg],
        )
        for cfg in config_order
    ]

    pop_legend = ax.legend(
        handles=pop_handles,
        title="Population size",
        loc="upper left",
    )
    ax.add_artist(pop_legend)
    ax.legend(
        handles=config_handles,
        title="Configuration",
        loc="upper right",
    )

    plt.tight_layout()
    plt.savefig(plot_dir / f"success_rate{filename_suffix}.svg", bbox_inches="tight")
    plt.savefig(
        plot_dir / f"success_rate{filename_suffix}.png", bbox_inches="tight", dpi=150
    )
    plt.close()


def plot_experiment_4(dir: pathlib.Path) -> None:
    for var in ("mse", "mse_val", "nmse", "nmse_val", "r2", "r2_val", "evaluations"):
        _plot_experiment_4(
            dir,
            var,
            dataset_keys=[
                "modular_1",
                "modular_2",
                "modular_3",
                "modular_4",
                "modular_5",
            ],
            filename_suffix="_modular",
        )
    _plot_experiment_4_time_to_threshold(
        dir,
        dataset_keys=[
            "modular_1",
            "modular_2",
            "modular_3",
            "modular_4",
            "modular_5",
        ],
        filename_suffix="_modular",
    )
    _plot_experiment_4_success_rate(
        dir,
        dataset_keys=[
            "modular_1",
            "modular_2",
            "modular_3",
            "modular_4",
            "modular_5",
        ],
        filename_suffix="_modular",
    )


def plot_experiment_5(dir: pathlib.Path, var: str = "mse"):
    db_path = dir / "all_results.duckdb"
    plot_dir = dir / "plots"

    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    conn = duckdb.connect(db_path)

    df = conn.execute(f"""
    SELECT
        kernel,
        dataset,
        template_depth,
        population_size,
        run,
        {var}
    FROM results
    """).df()

    df["kernel"] = df["kernel"].map(KERNEL_LABELS).fillna(df["kernel"])
    df["dataset"] = df["dataset"].map(DATASET_LABELS).fillna(df["dataset"])

    if var == "evaluations":
        df["evaluations"] /= 10  # 10-minute time budget for GPU runs

    dataset_order = [
        DATASET_LABELS[d]
        for d in [
            "daily_demand",
            "auto_mpg",
            "california_housing",
            "feynman",
        ]
    ]

    kernel_order = list(KERNEL_LABELS.values())
    depth_order = sorted(df["template_depth"].unique())
    var_label = VAR_LABELS.get(var, var)

    # Combined hue: one tab20 pair per kernel, one shade per depth
    _tab20 = sns.color_palette("tab20", n_colors=20)
    palette = {
        f"{k} (D={d})": _tab20[i * 2 + j]
        for i, k in enumerate(kernel_order)
        for j, d in enumerate(depth_order)
    }
    df["hue_label"] = df["kernel"] + " (D=" + df["template_depth"].astype(str) + ")"
    hue_order = [f"{k} (D={d})" for k in kernel_order for d in depth_order]

    g = sns.relplot(
        data=df,
        x="population_size",
        y=var,
        hue="hue_label",
        hue_order=hue_order,
        col="dataset",
        kind="line",
        errorbar=("pi", 50),
        estimator="median",
        col_order=dataset_order,
        marker="o",
        palette=palette,
        facet_kws={
            "sharey": True if (var == "evaluations") else "col",
            "margin_titles": True,
        },
        height=2.5,
        aspect=1.0,
    )

    g.set_titles(col_template="{col_name}")
    g.axes[0, 0].annotate(
        var_label,
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

    if g.legend:
        sns.move_legend(
            g,
            loc="lower center",
            bbox_to_anchor=(0.42, -0.05),
            ncol=len(hue_order),
            title=None,
        )

    g.figure.canvas.draw()
    positions = [ax.get_position() for ax in g.axes.flat]
    grid_left = min(p.x0 for p in positions)
    grid_right = max(p.x1 for p in positions)
    g.figure.supxlabel("Population Size", x=(grid_left + grid_right) / 2, y=0.065)

    plt.savefig(plot_dir / f"{var}.svg", bbox_inches="tight")
    plt.savefig(plot_dir / f"{var}.png", bbox_inches="tight", dpi=150)


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
