import duckdb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

sns.set_style("whitegrid")


def plot_scalability(
    conn: duckdb.DuckDBPyConnection,
    problems: list[str] | None = None,
    methods: list[str] | None = None,
    metrics=("evaluations", "total_time_seconds", "generation"),
    log_x: bool = True,
    log_y: bool = True,
    show_success_rates: bool = False,
):
    sr_thresholds = [
        (0.9, "solid"),
        (0.5, "dashed"),
        (0.1, "dotted"),
        (0.0, (0, (1, 10))),
    ]

    metric_names = dict(
        evaluations="Evaluations",
        total_time_seconds="Time [s]",
        generation="Generations",
    )

    if methods is None:
        methods = sorted(
            [
                m
                for m, *_ in conn.sql(
                    "SELECT DISTINCT(method_name) AS method FROM results"
                ).fetchall()
                if m is not None
            ]
        )

    if problems is None:
        problems = sorted(
            [
                p
                for p, *_ in conn.sql(
                    "SELECT DISTINCT(problem_name) AS problem FROM results"
                ).fetchall()
                if p is not None
            ]
        )

    if len(methods) == 0 or len(problems) == 0:
        print("Not enough data?")
        return

    nrows = len(metrics)
    ncols = len(problems)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows), sharex="col"
    )
    axes = axes.reshape(nrows, ncols)

    methods = sorted(methods)
    hues = sns.color_palette(n_colors=len(methods))  # , palette="colorblind")

    method_handles = dict()
    for mi, metric in enumerate(metrics):
        for pi, problem_name in enumerate(problems):
            ax = axes[mi, pi]

            num_runs = conn.execute(
                "SELECT count(DISTINCT run) FROM results WHERE problem_name = $1",
                [problem_name],
            ).fetchone()[0]

            sr_ax = None
            if show_success_rates:
                sr_ax = ax.twinx()

            all_dims = []
            for i, method in enumerate(methods):
                hue = hues[i]

                if method not in method_handles:
                    method_handles[method] = ax.plot([], [], label=method, color=hue)[0]

                df = conn.execute(
                    f"""
                    SELECT dims, count(status)::DOUBLE / {float(num_runs)} AS success_rate, quantile({metric}, 0.1) AS lower, median({metric}) AS median, quantile({metric}, 0.9) AS upper FROM results
                    WHERE problem_name = $1 AND method_name = $2 AND status = 'TargetReached'
                    GROUP BY dims
                    ORDER BY dims ASC
                """,
                    [problem_name, method],
                ).df()

                # pyplot expects the errors to be positive deviations only...
                df["lower"] = df["median"] - df["lower"]
                df["upper"] = df["upper"] - df["median"]

                if show_success_rates:
                    sr_ax.plot(
                        df["dims"], df["success_rate"], color=hue, lw=0.75, alpha=0.5
                    )

                all_dims += df["dims"].tolist()

                start, end = 0, 0
                for sr_threshold, ls in sr_thresholds:
                    start = end
                    while (
                        end < len(df) and df["success_rate"].iloc[end] >= sr_threshold
                    ):
                        end += 1

                    if start >= end:
                        # skip empy
                        continue

                    if start > 0:
                        # plot the connecting line between two success rate thresholds
                        ax.plot(
                            df["dims"].iloc[start - 1 : start + 1],
                            df["median"].iloc[start - 1 : start + 1],
                            color=hue,
                            ls=ls,
                        )

                    if sr_threshold > 0.0:
                        ax.errorbar(
                            df["dims"].iloc[start:end],
                            df["median"].iloc[start:end],
                            yerr=(
                                df["lower"].iloc[start:end],
                                df["upper"].iloc[start:end],
                            ),
                            color=hue,
                            capsize=5,
                            ls=ls,
                        )
                    else:
                        ax.plot(
                            df["dims"].iloc[start:end],
                            df["median"].iloc[start:end],
                            color=hue,
                            ls=ls,
                        )

            if log_y:
                ax.set_yscale("log")
            if log_x:
                ax.set_xscale("log")
            sdims = sorted(set(all_dims))
            ax.set_xticks(sdims)
            # make sure all ticks are shown
            ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
            assert len(sdims) > 0, sdims
            ax.set_xlim(xmin=sdims[0] - 1)  # , xmax=sdims[-1] + 1)

            if mi == 0:
                ax.set_title(problem_name)
            elif mi == nrows - 1:
                ax.set_xlabel("Dimensions")

            if pi == 0:
                ax.set_ylabel(metric_names.get(metric, metric))

            if show_success_rates:
                sr_ax.set_ylim((0, 1))
                sr_ax.set_ylabel("Success Rate")

    # add the thresholds to the legend
    sr_handles = []
    for i, (sr_threshold, ls) in enumerate(sr_thresholds):
        is_first = i == 0
        prev = 1.0 if is_first else sr_thresholds[i - 1][0]
        is_zero = sr_threshold == 0.0
        label = f"{'(['[is_first]}{prev}, {sr_threshold}{'])'[is_zero]}"

        sr_handles.append(
            axes.flat[0].plot([], [], ls=ls, color="black", label=label)[0]
        )
    l1 = fig.legend(
        title="Success Rate",
        handles=sr_handles,
        loc="center right",
        # ncols=len(sr_thresholds),
        # mode="expand",
        borderaxespad=0.0,
        frameon=False,
    )
    fig.legend(
        handles=list(method_handles.values()),
        loc="lower center",
        ncols=min(len(problems), len(methods)),
        mode="expand",
        borderaxespad=0.0,
        frameon=False,
    )
    fig.add_artist(l1)

    return fig
