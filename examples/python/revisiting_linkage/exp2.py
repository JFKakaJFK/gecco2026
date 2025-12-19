import pathlib
import re

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygom
import seaborn as sns
from matplotlib.colors import Normalize
from pygom import *
from sklearn.externals._packaging.version import SubLocalType

from src.config import c, instantiate
from src.data import prepare_problem, problem_info
from src.plots import plot_convergence_so
from src.postprocessing import load_results
from src.run import compute_run_path, run_tasks

REPEATS_PER_DATASET = 30
NUM_FOLDS = 5

REPEATS_PER_FOLD = REPEATS_PER_DATASET // NUM_FOLDS

RESULT_DIR = pathlib.Path("results") / "linkage_details2"
DATA_DIR = RESULT_DIR / "data"
LOG_DIR = RESULT_DIR / "raw"
PARQUET_DIR = RESULT_DIR / "processed"
PLOT_DIR = RESULT_DIR / "plots"

BUDGET = c.Budget(
    max_generations=41
    # max_evaluations=int(5e5)
    # max_evaluations=int(1e6)
    # max_evaluations=int(1e7)
)


def problems(rng):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for problem in [  #
        "Airfoil",
        # "Dow Chemical",
        # "Tower",
    ]:
        for (
            fold,
            X_path,
            y_path,
            X_test_path,
            y_test_path,
            X_fold,
            y_fold,
            is_synthetic,
        ) in prepare_problem(
            rng,
            problem,
            NUM_FOLDS,
            DATA_DIR,
        ):
            # problem_info(f"{problem} (Fold {fold})", X_fold, y_fold[:, 0])

            min_y, max_y = (
                float(np.nanmin(y_fold[:, 0])),
                float(np.nanmax(y_fold[:, 0])),
            )

            for height in [4]:
                template = c.Template(
                    [c.TemplateNode.full_nary(branching_factor=2, depth=height - 1)], []
                )
                for operator_set, operators in [
                    (
                        "small",
                        [c.OpAdd(), c.OpSub(), c.OpMul(), c.OpDiv(), c.OpSin()],
                    )
                ]:
                    for linear_scaling in [  #
                        # False,
                        True,
                    ]:
                        ctx = c.GPContext(
                            num_inputs=int(X_fold.shape[1]),
                            expression_template=template,
                            operators=operators,
                            constant_representation="ercs",
                        )

                        for run in range(REPEATS_PER_FOLD):
                            seed = int(rng.integers(2**32))
                            yield (
                                dict(
                                    problem_name=problem,
                                    fold=fold,
                                    run=run,
                                    template_height=height,
                                    operator_set=operator_set,
                                    linear_scaling=linear_scaling,
                                ),
                                seed,
                                c.SRProblem(
                                    ctx,
                                    x_train=c.np.load(str(X_path.absolute())),
                                    y_train=c.np.load(str(y_path.absolute())),
                                    x_test=c.np.load(str(X_test_path.absolute())),
                                    y_test=c.np.load(str(y_test_path.absolute())),
                                    objectives="nmse",  # = MSE / var(y_train)
                                    linear_scaling=linear_scaling,
                                    init=c.HalfHalfInit(),
                                    constant_init_lower_bound=min_y,
                                    constant_init_upper_bound=max_y,
                                    # early termination condition for "perfect" expression recovery
                                    target_objectives=[
                                        # R2 >= 0.999 for black-box problems
                                        # and (N)MSE < 1e-8 for synthetic problems
                                        0.0001 if not is_synthetic else 1e-8
                                    ],
                                ),
                                ctx,
                            )


def methods(info, ctx):
    # the GPContext provides lookup tables, one of them being the normalized pairwise node proximity
    ctx = instantiate(ctx, ctx=vars(pygom))

    # IMS options
    initial_population_size = 1024  # 4096  # 2048
    max_num_populations = 1
    restart_stale_populations = False  # restart the last population if it has converged

    for similarity in [  #
        "$MI$",  # plain MI
        "$NMI$",  # plain MI
        "$MI_{adjusted}$",  # adjusted MI as per https://arxiv.org/pdf/1904.02050
        # r"$MI_{any\ active}$",  # Mask inactive and normalize, keeping only at least partialy active pairs
        r"$MI_{mask\ inactive}$",  # Mask inactive
        # r"$NMI_{mask\ inactive}$",
        "Node",  # Normalized pairwise node proximity
        # r"Node + $MI_{mask\ inactive}$",
        "Random",  # Random similiarty
    ]:
        discrete_model_kwargs = dict(
            # linkage learning parameters
            metric="random"
            if "Random" in similarity
            else ("nmi" if "+" in similarity or "nmi" in similarity.lower() else "mi"),
            intron_strategy=(
                "weighted_any_active" if "any" in similarity else "mark_only"
            )
            if "active" in similarity
            else "none",
            normalize_initial_linkage_bias="adjusted" in similarity,
            custom_similarity=c.np.array(ctx.normalized_node_proximity().tolist())
            if "Node" in similarity
            else None,
            # the full FOS is excluded
            filter_root=True,
            # treat continuous nodes semantically by binning them into 25 bins
            # as per https://arxiv.org/pdf/1904.02050
            merge_continuous=False,
            num_continuous_bins=25,
            eta_custom_similarity=0.5 if "+" in similarity else None,
        )

        yield (
            f'"GP-GOMEA ({similarity})"',
            lambda run_path: c.MixedGOMEA(
                discrete_model=c.LinkageTreeFOS(**discrete_model_kwargs),
                population_options=c.PopulationOptions(
                    forced_improvements=False,  # not used per default as per https://arxiv.org/pdf/1904.02050
                    subset_logfile=str(
                        run_path / "subset_stats.csv"
                    ),  # provide the run-dependent logfile via a callback
                ),
                # disable evolutionary constant optimization as per https://ir.cwi.nl/pub/34425/paper_115.pdf
                rv_options=c.RvOptions(enabled=False),
                continuous_model=c.FullFOS(),
                sampling_model=c.AMaLGaMSamplingModel(),
                # IMS options
                ims_options=c.IMSOptions(
                    initial_population_size=initial_population_size,
                    max_num_populations=max_num_populations,
                    restart_stale_populations=restart_stale_populations,
                ),
            ),
        )


def all_tasks():
    rng = np.random.default_rng(seed=42)
    for info, seed, instance, ctx in problems(rng):
        # for the same run, all methods get the same seed...
        for method_name, method in methods(info, ctx):
            run_info = dict(method_name=method_name, **info)
            run_path = compute_run_path(LOG_DIR, run_info)
            yield (
                run_info,
                dict(
                    instance=instance, method=method(run_path), budget=BUDGET, seed=seed
                ),
            )


def analyze_subset_stats(conn, odir):
    print(conn.sql("DESCRIBE fos_stats;"))

    problems = sorted(
        [
            p
            for p, *_ in conn.sql(
                "SELECT DISTINCT(problem_name) FROM fos_stats"
            ).fetchall()
        ]
    )

    for problem in problems:
        pdir = odir / problem
        pdir.mkdir(parents=True, exist_ok=True)

        methods = sorted(
            [
                m
                for m, *_ in conn.execute(
                    "SELECT DISTINCT(method_name) FROM fos_stats WHERE problem_name = $1",
                    [problem],
                ).fetchall()
            ]
        )

        generations = sorted(
            [
                g
                for g, *_ in conn.execute(
                    "SELECT DISTINCT(generation::UINTEGER) FROM fos_stats WHERE problem_name = $1",
                    [problem],
                ).fetchall()
            ]
        )

        nrows = int(np.ceil(len(methods) / 2))
        ncols = len(generations)
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=True,
            sharey=True,
            figsize=(ncols * 3, nrows * 3),
            gridspec_kw=dict(wspace=0.1, hspace=0.05),
        )

        cmap = "Blues"  # "plasma"  # sns.diverging_palette(230, 20, as_cmap=True)

        cmap = plt.get_cmap(cmap)
        norm = Normalize(0, 1)
        cbar = cm.ScalarMappable(cmap=cmap, norm=norm)

        cbar_ax = axes.ravel().tolist()

        for col, generation in enumerate(generations):
            for row in range(nrows):
                ax = axes[row, col]

                avg_sim, mask = None, None

                ax_methods = methods[2 * row : 2 * row + 2]
                for is_lower, method in enumerate(ax_methods):
                    stats = conn.execute(
                        """
                        SELECT
                            population_size,
                            cluster,
                            similarity,
                            subsets,
                            usage_count,
                            evaluation_rate,
                            acceptance_rate,
                            avg_improvement,
                            solution_activation_rate,
                            variables_activation_rate
                        FROM fos_stats
                        WHERE
                            problem_name = $1
                            AND method_name = $2
                            AND generation = $3
                        """,
                        [problem, method, generation],
                    ).df()

                    similarities = [
                        np.array([r.tolist() for r in s]) for s in stats["similarity"]
                    ]
                    avg_similarity = similarities[0]
                    for s in similarities[1:]:
                        avg_similarity += s
                    avg_similarity /= len(similarities)

                    if avg_sim is None:
                        avg_sim = np.zeros_like(avg_similarity, dtype=np.float64)
                        avg_sim += np.triu(avg_similarity, k=0)
                        mask = np.zeros_like(avg_sim, dtype=bool)
                        mask[np.tril_indices(avg_sim.shape[0], k=0)] = True
                    else:
                        avg_sim += np.tril(avg_similarity, k=0)
                        mask[np.tril_indices(avg_sim.shape[0], k=-1)] = False

                if (avg_sim[~mask] > 1.0).any():
                    print(problem, ax_methods, "Pairwise similarity > 1.0 found!")
                    print(np.max(avg_sim[~mask]))

                sns.heatmap(
                    avg_sim,
                    mask=mask,
                    cmap=cmap,
                    norm=norm,
                    square=True,
                    linewidths=0.5,
                    cbar=False,
                    ax=ax,
                )

                def fmt_name(name):
                    return re.match(r"GP-GOMEA \(([^)]+)\)", name).group(1)

                title = (
                    f"Generation {generation:>3d}\n" if row == 0 else ""
                ) + fmt_name(ax_methods[0])
                ax.set_title(title)

                if col == 0 and len(ax_methods) > 1:
                    ax.set_ylabel(fmt_name(ax_methods[1]))

                ticks, ticklabels = zip(
                    *[
                        (
                            t + 0.5,  # center the tick
                            str(t),  # + 1) # 0/1 indexing
                        )
                        for t in range(0, avg_sim.shape[0], 2)
                    ]
                )
                ax.set_xticks(ticks, labels=ticklabels)
                ax.set_yticks(ticks, labels=ticklabels)

        cb = fig.colorbar(
            cbar,
            pad=0.025,
            shrink=0.4,
            aspect=30,
            drawedges=False,
            # extend="max",
            label="Similarity",
            ax=cbar_ax,
        )
        cb.outline.set_linewidth(0.0)

        fig.savefig(
            pdir / "similarities.pdf",
            dpi=600,
            bbox_inches="tight",
            transparent=True,
        )


def main():
    # TODO add dry run option that only checks how many jobs would be run (per cpu)
    run_tasks(
        LOG_DIR,
        all_tasks(),
        clean=True,
        # limit=1,
        # max_workers=1,
    )

    with load_results(
        LOG_DIR,
        file_pattern="stats",
        # enable pre-processing the .csv logs into .parquet files
        preprocess=True,
        parquet_dir=PARQUET_DIR / "stats",
    ) as conn:
        load_results(
            LOG_DIR,
            file_pattern="subset_stats",
            table_name="fos_stats",
            types=dict(
                population_size="INTEGER",
                cluster="INTEGER",
                similarity="DOUBLE[][]",
                subsets="UINTEGER[][]",
                usage_count="UINTEGER[]",
                evaluation_rate="DOUBLE[]",
                acceptance_rate="DOUBLE[]",
                avg_improvement="DOUBLE[]",
                solution_activation_rate="DOUBLE[]",
                variables_activation_rate="DOUBLE[]",
            ),
            # enable pre-processing the .csv logs into .parquet files
            preprocess=True,
            parquet_dir=PARQUET_DIR / "fos_stats",
            conn=conn,
        )

        PLOT_DIR.mkdir(parents=True, exist_ok=True)

        with sns.axes_style("white"):
            analyze_subset_stats(conn, PLOT_DIR / "stats")

        plot_convergence_so(
            PLOT_DIR / "convergence",
            conn,
            y_var="1.0 - objectives[1]::DOUBLE",  # transform NMSE into R2
            y_agg="MAX",  # higher R^2 is better
            y_label="$R^2$ Train",
            ymin="auto",
            ymax="auto",
            # merge folds and runs into one seaborn "unit"
            unit_query="format('{}.{}', fold, run)",
            # split up the plot into the following rows
            modifier_query="[template_height::STRING,operator_set::STRING,IF(linear_scaling, 'Yes', 'No')::STRING]",
            modifier_labels=[
                "Template Height",
                "Operators",
                "Linear Scaling",
            ],
            nsamples=100,
        )


if __name__ == "__main__":
    main()
