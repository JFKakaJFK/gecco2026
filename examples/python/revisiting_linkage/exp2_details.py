import pathlib
import re
from urllib.parse import quote

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygom
import seaborn as sns
from matplotlib.colors import AsinhNorm, Normalize, SymLogNorm
from matplotlib.lines import Line2D
from pygom import *
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg._isolve.lsqr import eps

from src.config import c, instantiate
from src.data import prepare_problem, problem_info
from src.plots import plot_convergence_so
from src.postprocessing import load_results
from src.run import compute_run_path, run_tasks

sns.set_theme(
    context="paper",
    style="ticks",
    font_scale=1.5,
    rc={
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    },
)

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

REPEATS_PER_DATASET = 30
NUM_FOLDS = 5

REPEATS_PER_FOLD = REPEATS_PER_DATASET // NUM_FOLDS

RESULT_DIR = pathlib.Path("results") / "linkage_details"
RESULT_DIR = pathlib.Path("results") / "linkage_details_large"
DATA_DIR = RESULT_DIR / "data"
LOG_DIR = RESULT_DIR / "raw"
PARQUET_DIR = RESULT_DIR / "processed"
PLOT_DIR = RESULT_DIR / "plots"

SEED = 42

BUDGET = c.Budget(max_generations=41)


def problems(rng):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    seeds = [int(rng.integers(2**32)) for _ in range(REPEATS_PER_FOLD)]

    for problem in [  #
        "Airfoil",
        "Bike Sharing",
        "Concrete Compressive Strength",
        "Dow Chemical",
        "Tower",
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

            for height in [
                4
                # 5
            ]:
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
                        False,
                        True,
                    ]:
                        ctx = c.GPContext(
                            num_inputs=int(X_fold.shape[1]),
                            expression_template=template,
                            operators=operators,
                            constant_representation="ercs",
                        )

                        for run, seed in enumerate(seeds):
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
        "$MI_{adjusted}$",  # adjusted MI as per https://arxiv.org/pdf/1904.02050
        r"$MI_{mask\ inactive}$",  # Mask inactive
        # r"$MI_{any\ active}$",  # Mask inactive + only consider partially active variables/variable pairs
        # r"$MI_{all\ active}$",  # Mask inactive + only consider fully active variables
        "Node",  # Normalized pairwise node proximity
        # "Node (static)",  # same, but first LT is kept throughout
        # r"Node * $MI_{mask\ inactive}$",
        "Random",  # Random similiarty
        "Node (wVIG)",
        # "Node (wVIG, static)",
        "Node (peter)",
        # "Node (peter, static)",
        # "Univariate",
    ]:
        custom_similarity = None
        if "Node" in similarity:
            if "wVIG" in similarity:
                custom_similarity = c.np.array(ctx.normalized_w_vig().tolist())
            elif "peter" in similarity:
                custom_similarity = c.np.array(ctx.subtree_co_occurrences().tolist())
            else:
                custom_similarity = c.np.array(ctx.normalized_node_proximity().tolist())
        discrete_model_kwargs = dict(
            # linkage learning parameters
            metric="random" if "Random" in similarity else "mi",
            intron_strategy=(
                "any_active"
                if "any" in similarity
                else ("all_active" if "all" in similarity else "mark_only")
            )
            if "active" in similarity
            else "none",
            normalize_initial_linkage_bias="adjusted" in similarity,
            custom_similarity=custom_similarity,
            custom_similarity_agg="mul"
            if "*" in similarity
            else (
                "max" if "max" in similarity else ("add" if "+" in similarity else None)
            ),
            # the full FOS is excluded
            filter_root=True,
            # treat continuous nodes semantically by binning them into 25 bins
            # as per https://arxiv.org/pdf/1904.02050
            merge_continuous=False,
            num_continuous_bins=25,
            freeze="static"
            in similarity,  # only build the LT in the first generation, then stick with it
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
    rng = np.random.default_rng(seed=SEED)
    for info, seed, instance, ctx in problems(rng):
        # for the same run, all methods get the same seed...
        for method_name, method in methods(info, ctx):
            run_info = dict(method_name=method_name, **info)
            run_path = compute_run_path(LOG_DIR, run_info)
            yield (
                run_info,
                dict(
                    instance=instance,
                    method=method(run_path),
                    budget=BUDGET,
                    seed=seed,
                    tracking_options=dict(
                        max_generations_until_next_report=1,  # log every generation
                        initial_evaluations_until_next_report=1024,  # and after initialization
                        max_evaluations_until_next_report=int(
                            1e9
                        ),  # but other than that don't trigger logs due to the number of evaluations done...
                    ),
                ),
            )


def method2name(m: str) -> str:
    return {
        "$MI_{mask\\ inactive}$": "$MI_{masked}$",
        "Node (peter)": "#Common Subfunctions",
    }.get(m, m)


m_order = {
    m: i
    for i, m in enumerate(
        [
            "Random",
            "$MI$",
            "$MI_{adjusted}$",
            "$MI_{masked}$",
            "Node",
            "Node (static)",
        ]
    )
}


def analyze_subset_stats3(
    conn,
    odir,
    problem_query="format('{}{}', problem_name, IF(linear_scaling, ' LS', ''))",
    where_query: str = r"method_name NOT SIMILAR TO '.*(\*|any|all|wVIG|static).*'",
    max_generation: int = 20,
    runs=("avg", 0),
    cmap="Blues",
    side_methods=("Random", "#Common Subfunctions", "Node / Node (static)"),
):
    def fmt_name(name):
        rm = re.match(r"^GP-GOMEA \((.+)\)$", name)
        n = rm.group(1) if rm else name
        n = method2name(n)
        return "Node / Node (static)" if n == "Node" else n

    problems = sorted(
        [
            p
            for p, *_ in conn.sql(
                f"SELECT DISTINCT({problem_query}) FROM fos_stats WHERE {where_query}"
            ).fetchall()
        ]
    )

    for run in runs:
        for problem in problems:
            methods = sorted(
                [
                    m
                    for m, *_ in conn.execute(
                        f"SELECT DISTINCT(method_name) FROM fos_stats WHERE {problem_query} = $1 AND {where_query}",
                        [problem],
                    ).fetchall()
                ]
            )

            generations = sorted(
                [
                    g
                    for g, *_ in conn.execute(
                        f"SELECT DISTINCT(generation::UINTEGER) FROM fos_stats WHERE {problem_query} = $1 AND {where_query} AND generation::UINTEGER <= $2",
                        [problem, max_generation],
                    ).fetchall()
                ]
            )

            masks = {}
            norms = {}
            similarities = {}
            n2m = {}
            for method in methods:
                n = fmt_name(method)
                n2m[n] = method

                run_query = "true"
                if isinstance(run, int):
                    target_run = [
                        r
                        for r, *_ in conn.execute(
                            f"SELECT DISTINCT(format('{{}}.{{}}', fold, seed)) AS run FROM fos_stats WHERE {problem_query} = $1 AND {where_query} AND generation::UINTEGER <= $2 AND method_name = $3",
                            [problem, max_generation, method],
                        ).fetchall()
                    ][run]
                    run_query = f"format('{{}}.{{}}', fold, seed) = '{target_run}'"

                sims = []
                for _, generation in enumerate(generations):
                    stats = conn.execute(
                        f"""
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
                            variables_activation_rate,
                            format('{{}}.{{}}', fold, seed) as run
                        FROM fos_stats
                        WHERE
                            {problem_query} = $1
                            AND method_name = $2
                            AND generation = $3
                            AND {where_query}
                            AND {run_query}
                        ORDER BY run
                        """,
                        [problem, method, generation],
                    ).df()

                    if len(stats) == 0:
                        continue

                    num_vars = len(stats["similarity"][0])
                    num_subsets = len(stats["subsets"][0])
                    trees = stats["subsets"]
                    num_trees = len(trees)

                    def lt_to_matrix(lt):
                        M = np.zeros((num_subsets, num_vars))
                        for s, subset in enumerate(lt):
                            for v in subset:
                                M[s, v] = 1.0
                        return M

                    def jaccard(A, B, e=1e-9):
                        intersection = A @ B.T
                        union = (
                            A.sum(axis=1)[:, None]
                            + B.sum(axis=1)[None, :]
                            - intersection
                            + eps
                        )
                        return intersection / union

                    freqs = lt_to_matrix(trees[0])

                    for lt in trees[1:]:
                        M = lt_to_matrix(lt)

                        # similarity between current tree nodes and accumulated rows
                        S = jaccard(M, freqs)

                        # Hungarian maximizes total similarity
                        row_ind, col_ind = linear_sum_assignment(-S)

                        # add aligned subsets
                        for i, j in zip(row_ind, col_ind):
                            freqs[j] += M[i]

                    # normalize if you want frequencies
                    freqs /= num_trees

                    # freqs = np.zeros((num_subsets, num_vars))
                    # for lt in stats["subsets"]:
                    #     for s, subset in enumerate(lt):
                    #         for v in subset:
                    #             freqs[s, v] += 1.0
                    # freqs /= num_trees

                    if n not in masks:
                        masks[n] = np.zeros_like(freqs, dtype=np.bool_)
                        masks[n][:num_vars, :] = True  # hide univariate portion

                    sims.append(freqs)
                similarities[n] = sims

                vmin, vmax = np.inf, -np.inf

                ix = ~masks[n]
                for s in sims:
                    vmin = min(vmin, np.min(s[ix]))
                    vmax = max(vmax, np.max(s[ix]))

                norms[n] = Normalize(vmin=vmin, vmax=vmax)

            # print([m for _m in methods if (m := fmt_name(_m)) not in side_methods])
            rows = sorted(
                [m for _m in methods if (m := fmt_name(_m)) not in side_methods],
                key=lambda m: m_order.get(m, np.inf),
            )
            assert len(side_methods) <= len(rows)

            nrows = len(rows)
            ncols = len(generations) + int(len(side_methods) > 0)
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                sharex=True,
                sharey=True,
                figsize=(ncols * 4, nrows * 3),
                gridspec_kw=dict(wspace=0.1, hspace=0.1),
            )

            heatmap_kw = dict(
                square=True,
                linewidths=0,
                cbar=False,
            )

            def add_cbar(cax, norm, **kwargs):
                cbar = cm.ScalarMappable(cmap=cmap, norm=norm)
                kw = dict(
                    shrink=0.6,
                    aspect=30,
                    drawedges=False,
                )
                for k, v in kwargs.items():
                    kw[k] = v
                cb = fig.colorbar(
                    cbar,
                    ax=cax,
                    **kw,
                )
                cb.outline.set_linewidth(0.0)
                cb.ax.tick_params(
                    labelsize="xx-small",
                    pad=1,
                    length=2,
                    width=1,
                )

            def add_ticks(ax, sim):
                # ticks, ticklabels = zip(
                #     *[(t + 0.5, str(t)) for t in range(sim.shape[1], sim.shape[0], 5)]
                # )
                # ax.set_yticks(ticks, labels=ticklabels, fontsize="xx-small")
                # ticks, ticklabels = zip(
                #     *[(t + 0.5, str(t)) for t in range(0, sim.shape[1], 5)]
                # )
                # ax.set_xticks(ticks, labels=ticklabels, fontsize="xx-small")

                ax.tick_params(
                    axis="both",
                    which="major",
                    bottom=True,
                    left=True,
                    length=2,
                    width=1,
                )

            for ri, m in enumerate(rows):
                sims, norm, mask = similarities[m], norms[m], masks[m]
                for ci, g in enumerate(generations):
                    ax = axes[ri, ci]
                    if ci < len(sims):
                        sns.heatmap(
                            sims[ci][sims[ci].shape[1] :, :],
                            # mask=mask,
                            cmap=cmap,
                            norm=norm,
                            ax=ax,
                            **heatmap_kw,
                        )
                        add_ticks(ax, sims[ci])
                    else:
                        ax.set_axis_off()

                    ax.set_title(m if ri > 0 else f"Generation {g}\n{m}")

                    # if ri == nrows - 1:
                    #     ax.set_xlabel(f"Generation {g}", fontsize="medium")

                add_cbar(axes[ri, :], norm, location="left", pad=0.0175)

                if ri < len(side_methods):
                    sm = side_methods[ri]
                    sims, norm, mask = similarities[sm], norms[sm], masks[sm]

                    ax = axes[ri, -1]
                    sns.heatmap(
                        sims[0][sims[0].shape[1] :, :],
                        # mask=mask,
                        cmap=cmap,
                        norm=norm,
                        ax=ax,
                        **heatmap_kw,
                    )
                    add_ticks(ax, sims[0])

                    ax.set_title(sm)

                    add_cbar(axes[ri, :], norm, location="right", pad=0.005)

            if len(side_methods) > 0:
                ptl = axes[0, -2].get_position()
                ptr = axes[0, -1].get_position()
                pbl = axes[-1, -2].get_position()
                yr = ptl.y1 - pbl.y0
                p = 0.01

                x = (ptl.x1 + ptr.x0) / 2
                fig.add_artist(
                    Line2D(
                        [x, x],
                        [pbl.y0 - p * yr, ptl.y1 + p * yr],
                        transform=fig.transFigure,
                        color="black",
                        lw=2,
                    )
                )

            pdir = odir / f"lts{f'_run{run:03d}' if isinstance(run, int) else ''}"
            pdir.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                pdir / f"""lts_{quote(problem, safe=' (){}$_+-"')}.pdf""",
                dpi=600,
                bbox_inches="tight",
                transparent=True,
            )

            plt.close(fig)


def analyze_subset_stats2(
    conn,
    odir,
    problem_query="format('{}{}', problem_name, IF(linear_scaling, ' LS', ''))",
    max_generation: int = 20,
    runs=(0, "avg"),
    cmap="Blues",
    # where_query: str = r"method_name NOT SIMILAR TO '.*(\*|any|all|wVIG|static|Random|peter).*'",
    # side_methods=(
    #     "Node / Node (static)",
    #     "Node / Node (static)",
    #     "Node / Node (static)",
    # ),
    where_query: str = r"method_name NOT SIMILAR TO '.*(\*|any|all|wVIG|static).*'",
    side_methods=("Random", "#Common Subfunctions", "Node / Node (static)"),
):
    def fmt_name(name):
        rm = re.match(r"^GP-GOMEA \((.+)\)$", name)
        n = rm.group(1) if rm else name
        n = method2name(n)
        return "Node / Node (static)" if n == "Node" else n

    problems = sorted(
        [
            p
            for p, *_ in conn.sql(
                f"SELECT DISTINCT({problem_query}) FROM fos_stats WHERE {where_query}"
            ).fetchall()
        ]
    )

    for run in runs:
        for problem in problems:
            methods = sorted(
                [
                    m
                    for m, *_ in conn.execute(
                        f"SELECT DISTINCT(method_name) FROM fos_stats WHERE {problem_query} = $1 AND {where_query}",
                        [problem],
                    ).fetchall()
                ]
            )

            generations = sorted(
                [
                    g
                    for g, *_ in conn.execute(
                        f"SELECT DISTINCT(generation::UINTEGER) FROM fos_stats WHERE {problem_query} = $1 AND {where_query} AND generation::UINTEGER <= $2",
                        [problem, max_generation],
                    ).fetchall()
                ]
            )

            masks = {}
            norms = {}
            similarities = {}
            n2m = {}
            for method in methods:
                n = fmt_name(method)
                n2m[n] = method

                run_query = "true"
                if isinstance(run, int):
                    target_run = [
                        r
                        for r, *_ in conn.execute(
                            f"SELECT DISTINCT(format('{{}}.{{}}', fold, seed)) AS run FROM fos_stats WHERE {problem_query} = $1 AND {where_query} AND generation::UINTEGER <= $2 AND method_name = $3",
                            [problem, max_generation, method],
                        ).fetchall()
                    ][run]
                    run_query = f"format('{{}}.{{}}', fold, seed) = '{target_run}'"

                sims = []
                for _, generation in enumerate(generations):
                    stats = conn.execute(
                        f"""
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
                            variables_activation_rate,
                            format('{{}}.{{}}', fold, seed) as run
                        FROM fos_stats
                        WHERE
                            {problem_query} = $1
                            AND method_name = $2
                            AND generation = $3
                            AND {where_query}
                            AND {run_query}
                        ORDER BY run
                        """,
                        [problem, method, generation],
                    ).df()

                    if len(stats) == 0:
                        continue

                    _similarities = [
                        np.array([r.tolist() for r in s]) for s in (stats["similarity"])
                    ]
                    if len(_similarities) < 1:
                        continue
                    avg_similarity = _similarities[0]
                    for s in _similarities[1:]:
                        avg_similarity += s
                    avg_similarity /= len(_similarities)
                    sims.append(avg_similarity)
                similarities[n] = sims

                vmin, vmax = np.inf, -np.inf
                masks[n] = np.eye(sims[0].shape[0], dtype=np.bool_)
                ix = ~masks[n]
                for s in sims:
                    vmin = min(vmin, np.min(s[ix]))
                    vmax = max(vmax, np.max(s[ix]))

                norms[n] = Normalize(vmin=vmin, vmax=vmax)

            # print([m for _m in methods if (m := fmt_name(_m)) not in side_methods])
            rows = sorted(
                [m for _m in methods if (m := fmt_name(_m)) not in side_methods],
                key=lambda m: m_order.get(m, np.inf),
            )
            assert len(side_methods) <= len(rows)

            nrows = len(rows)
            ncols = len(generations) + int(len(side_methods) > 0)
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                sharex=True,
                sharey=True,
                figsize=(ncols * 4, nrows * 3),
                gridspec_kw=dict(wspace=0.1, hspace=0.1),
            )

            heatmap_kw = dict(
                square=True,
                linewidths=0,
                cbar=False,
            )

            def add_cbar(cax, norm, **kwargs):
                cbar = cm.ScalarMappable(cmap=cmap, norm=norm)
                kw = dict(
                    shrink=0.6,
                    aspect=30,
                    drawedges=False,
                )
                for k, v in kwargs.items():
                    kw[k] = v
                cb = fig.colorbar(
                    cbar,
                    ax=cax,
                    **kw,
                )
                cb.outline.set_linewidth(0.0)
                cb.ax.tick_params(
                    labelsize="xx-small",
                    pad=1,
                    length=2,
                    width=1,
                )

            def add_ticks(ax, sim):
                ticks, ticklabels = zip(
                    *[(t + 0.5, str(t)) for t in range(0, sim.shape[0], 5)]
                )
                ax.set_xticks(ticks, labels=ticklabels, fontsize="xx-small")
                ax.set_yticks(ticks, labels=ticklabels, fontsize="xx-small")
                ax.tick_params(
                    axis="both",
                    which="major",
                    bottom=True,
                    left=True,
                    length=2,
                    width=1,
                )

            for ri, m in enumerate(rows):
                sims, norm, mask = similarities[m], norms[m], masks[m]
                for ci, g in enumerate(generations):
                    ax = axes[ri, ci]
                    if ci < len(sims):
                        sns.heatmap(
                            sims[ci],
                            mask=mask,
                            cmap=cmap,
                            norm=norm,
                            ax=ax,
                            **heatmap_kw,
                        )
                        add_ticks(ax, sims[ci])
                    else:
                        ax.set_axis_off()

                    ax.set_title(m if ri > 0 else f"Generation {g}\n{m}")

                    # if ri == nrows - 1:
                    #     ax.set_xlabel(f"Generation {g}", fontsize="medium")

                add_cbar(axes[ri, :], norm, location="left", pad=0.0175)

                if ri < len(side_methods):
                    sm = side_methods[ri]
                    sims, norm, mask = similarities[sm], norms[sm], masks[sm]

                    ax = axes[ri, -1]
                    sns.heatmap(
                        sims[0],
                        mask=mask,
                        cmap=cmap,
                        norm=norm,
                        ax=ax,
                        **heatmap_kw,
                    )
                    add_ticks(ax, sims[0])

                    ax.set_title(sm)

                    add_cbar(axes[ri, :], norm, location="right", pad=0.005)

            if len(side_methods) > 0:
                ptl = axes[0, -2].get_position()
                ptr = axes[0, -1].get_position()
                pbl = axes[-1, -2].get_position()
                yr = ptl.y1 - pbl.y0
                p = 0.01

                x = (ptl.x1 + ptr.x0) / 2
                fig.add_artist(
                    Line2D(
                        [x, x],
                        [pbl.y0 - p * yr, ptl.y1 + p * yr],
                        transform=fig.transFigure,
                        color="black",
                        lw=2,
                    )
                )

            pdir = (
                odir / f"similarities{f'_run{run:03d}' if isinstance(run, int) else ''}"
            )
            pdir.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                pdir
                / f"""similarities_{quote(problem.lower().replace(" ", "-"), safe=' (){}$_+-"')}.pdf""",
                dpi=600,
                bbox_inches="tight",
                transparent=True,
            )

            plt.close(fig)


def analyze_subset_stats(
    conn,
    odir,
    problem_query="format('{}{}', problem_name, IF(linear_scaling, ' LS', ''))",
    where_query: str = r"method_name NOT SIMILAR TO '.*(\*|any|all|wVIG).*'",
    max_generation: int = 20,
    side_methods=("Random", "Fun", "Node"),
):
    print(conn.sql("DESCRIBE fos_stats;"))

    problems = sorted(
        [
            p
            for p, *_ in conn.sql(
                f"SELECT DISTINCT({problem_query}) FROM fos_stats WHERE {where_query}"
            ).fetchall()
        ]
    )

    def fmt_name(name):
        n = re.match(r"^GP-GOMEA \((.+)\)$", name).group(1)
        n = method2name(n)
        return "Node / Node (static)" if n == "Node" else n

    for run in [  #
        0,
        # 1,
        # 2,
        # 3,
        None,
    ]:  # average or single run...
        for ylog in [
            # False, True,
            # None,
            "measure"
        ]:
            joint_cbar = isinstance(ylog, bool)
            per_measure_cbar = ylog == "measure"

            for problem in problems:
                suffix = {False: "linear", True: "log", "measure": "measure"}.get(
                    ylog, "separate"
                )
                if run is not None:
                    suffix += f"_run{run:03d}"
                pdir = odir / suffix
                pdir.mkdir(parents=True, exist_ok=True)

                methods = sorted(
                    [
                        m
                        for m, *_ in conn.execute(
                            f"SELECT DISTINCT(method_name) FROM fos_stats WHERE {problem_query} = $1 AND {where_query}",
                            [problem],
                        ).fetchall()
                        if "static" not in m
                    ]
                )

                generations = sorted(
                    [
                        g
                        for g, *_ in conn.execute(
                            f"SELECT DISTINCT(generation::UINTEGER) FROM fos_stats WHERE {problem_query} = $1 AND {where_query} AND generation::UINTEGER <= $2",
                            [problem, max_generation],
                        ).fetchall()
                    ]
                )

                nrows = int(np.ceil(len(methods) / 2))
                ngens = len(generations)
                ncols = ngens + +int(len(side_methods) > 0)
                fig, axes = plt.subplots(
                    nrows=nrows,
                    ncols=ncols,
                    sharex=True,
                    sharey=True,
                    figsize=(ncols * 3, nrows * 3),
                    gridspec_kw=dict(wspace=0.1, hspace=0.05),
                )
                axes = axes.reshape(nrows, ncols)

                cmap = (
                    "Blues"
                    # "plasma"  # sns.diverging_palette(230, 20, as_cmap=True)
                )

                vmax = (
                    1.04 if "Airfoil" in problem else 2.1
                )  # 2  # 1.5  # 2  # 2  # 0  # 1
                cmap = plt.get_cmap(cmap)
                norm = Normalize(0, vmax)
                if ylog:
                    norm = SymLogNorm(0.05, vmin=0, vmax=vmax)
                    norm = AsinhNorm(vmin=0, vmax=vmax, linear_width=0.05)

                actual_vmax = 0

                data = []
                for col, generation in enumerate(generations):
                    row_data = []
                    for row in range(nrows):
                        avg_sim, mask = None, None

                        ax_methods = methods[2 * row : 2 * row + 2]
                        for is_lower, method in enumerate(ax_methods):
                            stats = conn.execute(
                                f"""
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
                                    variables_activation_rate,
                                    format('{{}}.{{}}', fold, seed) as run
                                FROM fos_stats
                                WHERE
                                    {problem_query} = $1
                                    AND method_name = $2
                                    AND generation = $3
                                    AND {where_query}
                                ORDER BY run
                                """,
                                [problem, method, generation],
                            ).df()

                            if run is not None:
                                runs = sorted(stats["run"].unique().tolist())
                                sim = stats[stats["run"] == runs[run]][
                                    "similarity"
                                ].values[0]
                                avg_similarity = np.array([r.tolist() for r in sim])
                            else:
                                similarities = [
                                    np.array([r.tolist() for r in s])
                                    for s in (stats["similarity"])
                                ]
                                if len(similarities) < 1:
                                    continue
                                avg_similarity = similarities[0]
                                for s in similarities[1:]:
                                    avg_similarity += s
                                avg_similarity /= len(similarities)

                            if avg_sim is None:
                                avg_sim = np.zeros_like(
                                    avg_similarity, dtype=np.float64
                                )
                                avg_sim += np.triu(avg_similarity, k=0)
                                mask = np.zeros_like(avg_sim, dtype=bool)
                                mask[np.tril_indices(avg_sim.shape[0], k=0)] = True
                            else:
                                avg_sim += np.tril(avg_similarity, k=0)
                                mask[np.tril_indices(avg_sim.shape[0], k=-1)] = False

                        if (avg_sim[~mask] > vmax).any() and joint_cbar:
                            print(
                                problem,
                                ax_methods,
                                f"Pairwise similarity > {vmax} found!",
                            )
                            print(np.max(avg_sim[~mask]))

                            actual_vmax = max(
                                actual_vmax, float(np.max(avg_sim[~mask]))
                            )
                        row_data.append((avg_sim, mask))
                    data.append(row_data)

                m_scale = {
                    # "GP-GOMEA ($MI$)": "log",
                    # "GP-GOMEA ($MI_{adjusted}$)": "log",
                    # "GP-GOMEA ($MI_{mask\\ inactive}$)": "log",
                    # 'GP-GOMEA (Node (peter))',
                    # "GP-GOMEA (Node (wVIG))": "log",
                    # 'GP-GOMEA (Node)'
                }
                m_norm = dict()
                row_norm = [None for _ in range(nrows)]
                for row in range(nrows):
                    row_vmax = 0

                    ax_methods = methods[2 * row : 2 * row + 2]
                    for col, _ in enumerate(generations):
                        avg_sim, mask = data[col][row]
                        row_vmax = max(row_vmax, float(np.max(avg_sim[~mask])))

                        for is_lower, m in enumerate(ax_methods):
                            vm = np.ones_like(avg_sim, dtype=np.bool_)
                            if is_lower:
                                vm[np.triu_indices_from(vm)] = False
                            else:
                                vm[np.tril_indices_from(vm)] = False
                            m_norm[m] = max(
                                m_norm.get(m, 0.0), float(np.max(avg_sim[vm]))
                            )

                    for is_lower, m in enumerate(ax_methods):
                        # print(m, m_scale.get(m, "linear"))
                        # print(m, m_norm[m])
                        m_norm[m] = (
                            Normalize(vmin=0, vmax=m_norm[m])
                            if m_scale.get(m, "linear") != "log"
                            else SymLogNorm(0.1, vmin=0, vmax=vmax)
                        )

                    row_norm[row] = Normalize(0, row_vmax)
                    if row == 0 and "Airfoil" in problem:  # hack
                        row_norm[row] = SymLogNorm(0.1, vmin=0, vmax=vmax)
                        # row_norm[row] = AsinhNorm(vmin=0, vmax=vmax, linear_width=0.05)
                # print(m_norm)

                for col, generation in enumerate(generations):
                    for row in range(nrows):
                        ax = axes[row, col]

                        ax_methods = methods[2 * row : 2 * row + 2]

                        avg_sim, mask = data[col][row]

                        linewidths = 0.125
                        if per_measure_cbar:
                            mmask = np.zeros_like(avg_sim, dtype=np.bool_)
                            mmask[np.tril_indices_from(mask)] = True
                            sns.heatmap(
                                avg_sim,
                                mask=mmask,
                                cmap=cmap,
                                norm=m_norm[ax_methods[0]],
                                square=True,
                                linewidths=linewidths,
                                cbar=False,
                                ax=ax,
                            )
                            if len(ax_methods) > 1:
                                mmask = np.zeros_like(avg_sim, dtype=np.bool_)
                                mmask[np.triu_indices_from(mask)] = True
                                sns.heatmap(
                                    avg_sim,
                                    mask=mmask,
                                    cmap=cmap,
                                    norm=m_norm[ax_methods[1]],
                                    square=True,
                                    linewidths=linewidths,
                                    cbar=False,
                                    ax=ax,
                                )
                        else:
                            sns.heatmap(
                                avg_sim,
                                mask=mask,
                                cmap=cmap,
                                norm=norm if joint_cbar else row_norm[row],
                                square=True,
                                linewidths=0.5,
                                cbar=False,  # not joint_cbar and col + 1 == len(generations),
                                # xticklabels=[],
                                # yticklabels=[],
                                ax=ax,
                            )

                        title = (
                            f"Generation {generation:>3d}\n" if row == 0 else ""
                        ) + fmt_name(ax_methods[0])
                        ax.set_title(title)

                        if col == 0 and len(ax_methods) > 1:
                            if per_measure_cbar:
                                ax.set_ylabel(fmt_name(ax_methods[1]), labelpad=35)
                            else:
                                ax.set_ylabel(fmt_name(ax_methods[1]))

                        ticks, ticklabels = zip(
                            *[
                                (
                                    t + 0.5,  # center the tick
                                    str(t),  # + 1) # 0/1 indexing
                                )
                                for t in range(0, avg_sim.shape[0], 5)  # 2)
                            ]
                        )
                        ax.set_xticks(ticks, labels=ticklabels, fontsize="xx-small")
                        ax.set_yticks(ticks, labels=ticklabels, fontsize="xx-small")
                        ax.tick_params(
                            axis="both",
                            which="major",
                            bottom=True,
                            left=True,
                            length=0,
                        )
                        # sns hides the ticks, this would enable them again...
                        # ax.spines[["left", "bottom"]].set_visible(True)
                        # ax.tick_params(
                        #     axis="both",
                        #     which="major",
                        #     bottom=True,
                        #     left=True,
                        #     length=3,
                        #     color="black",
                        # )

                if per_measure_cbar:
                    for row in range(nrows):
                        cbar_ax = axes[row, :].ravel().tolist()

                        ax_methods = methods[2 * row : 2 * row + 2]

                        for is_lower, m in enumerate(ax_methods):
                            cbar = cm.ScalarMappable(cmap=cmap, norm=m_norm[m])
                            cb = fig.colorbar(
                                cbar,
                                # pad=0.025,
                                # shrink=0.4,
                                location="left" if is_lower else "right",
                                pad=0.023 if is_lower else 0.01,
                                shrink=0.6,
                                aspect=30,
                                drawedges=False,
                                extend="max" if actual_vmax > vmax else None,
                                # label="Similarity",
                                ax=cbar_ax,
                            )
                            cb.outline.set_linewidth(0.0)
                            cb.ax.tick_params(
                                # direction="in",
                                labelsize="xx-small",
                                pad=1,
                                length=3,
                                width=1,
                            )
                elif joint_cbar:
                    cbar = cm.ScalarMappable(cmap=cmap, norm=norm)

                    cbar_ax = axes.ravel().tolist()

                    cb = fig.colorbar(
                        cbar,
                        # pad=0.025,
                        # shrink=0.4,
                        pad=0.01,
                        shrink=0.6,
                        aspect=30,
                        drawedges=False,
                        extend="max" if actual_vmax > vmax else None,
                        label="Similarity",
                        ax=cbar_ax,
                    )
                    cb.outline.set_linewidth(0.0)
                else:
                    for row in range(nrows):
                        cbar = cm.ScalarMappable(cmap=cmap, norm=row_norm[row])

                        cbar_ax = axes[row, :].ravel().tolist()

                        cb = fig.colorbar(
                            cbar,
                            # pad=0.025,
                            # shrink=0.4,
                            pad=0.01,
                            shrink=0.75,
                            # aspect=30,
                            drawedges=False,
                            extend="max" if actual_vmax > vmax else None,
                            # label="Similarity",
                            ax=cbar_ax,
                        )
                        cb.outline.set_linewidth(0.0)

                fig.savefig(
                    pdir / f"similarities_{problem.lower().replace(' ', '-')}.pdf",
                    dpi=600,
                    bbox_inches="tight",
                    transparent=True,
                )

                plt.close(fig)


def main():
    # TODO add dry run option that only checks how many jobs would be run (per cpu)
    # run_tasks(
    #     LOG_DIR,
    #     all_tasks(),
    #     # clean=True,
    #     # limit=1,
    #     # max_workers=44,
    # )

    preprocess = False
    with load_results(
        LOG_DIR,
        file_pattern="stats",
        # enable pre-processing the .csv logs into .parquet files
        preprocess=preprocess,
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
            preprocess=preprocess,
            parquet_dir=PARQUET_DIR / "fos_stats",
            conn=conn,
        )

        PLOT_DIR.mkdir(parents=True, exist_ok=True)

        with sns.axes_style("white"):
            # analyze_subset_stats3(conn, PLOT_DIR / "linkage_trees")
            analyze_subset_stats2(conn, PLOT_DIR / "stats")
            # analyze_subset_stats(conn, PLOT_DIR / "stats")

        exit()
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
