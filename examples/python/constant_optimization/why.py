import pathlib

import numpy as np
import pygom
from pygom import *
from tqdm import tqdm

from src.config import c, extract, instantiate, load_config
from src.data import prepare_problem, problem_info
from src.plots import plot_convergence_so
from src.postprocessing import load_results
from src.run import compute_run_path, run_tasks

REPEATS_PER_DATASET = 30  # 5
NUM_FOLDS = 5

REPEATS_PER_FOLD = REPEATS_PER_DATASET // NUM_FOLDS
assert REPEATS_PER_FOLD > 0, "No runs scheduled!"

RESULT_DIR = pathlib.Path("results") / "rvterm8"
DATA_DIR = RESULT_DIR / "data"
LOG_DIR = RESULT_DIR / "raw"
PARQUET_DIR = RESULT_DIR / "processed"
PLOT_DIR = RESULT_DIR / "plots"

BUDGET = c.Budget(
    # max_generations=300,
    # max_evaluations=int(5e5)
    # max_evaluations=int(1e6)
    # max_evaluations=int(2e6)
    # max_evaluations=int(5e6)
    max_evaluations=int(1e7)
    # max_time_seconds=30 * 60
)


def problems(rng):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    synthetic_problems = [
        # "-4.2 * x0 + 1.141 * x1 + 2.72 * x2",
        # "0.1 * x0 + 0.2 * x1 + 2.4 * x2",
        # "sin(3.141 * x0)/(3.141 * x0)",
        # "sin(3.141 * x0 + 2.72)",
        # "sin(1.772 * x0) + sin(2.035 * x2)",
        # "sin(1.57 * x0 + 1.04 * x1)",
    ]

    noise_levels = [
        0.0,  # 0.05, 0.1
    ]

    problems = [
        (
            p,
            dict(
                synthetic_problem_kwargs=dict(
                    size=1000,
                    noise=n,
                )
            ),
        )
        for p in synthetic_problems
        for n in noise_levels
    ] + [
        # "1028_SWD",
        # # "1089_USCrime",
        # "210_cloud",
        # "522_pm10",
        "Airfoil",
        "Bike Sharing",
        "Concrete Compressive Strength",
        # "Dow Chemical",
        # "Tower",
        # "Energy Cooling",
        # "Energy Heating",
        # "Yacht Hydrodynamics",
    ]

    for problem in problems:
        kwargs = {}
        if isinstance(problem, tuple):
            problem, kwargs = problem
        for (
            fold,
            X_path,
            y_path,
            X_test_path,
            y_test_path,
            X_fold,
            y_fold,
            is_synthetic,
        ) in prepare_problem(rng, problem, NUM_FOLDS, DATA_DIR, **kwargs):
            # problem_info(f"{problem} (Fold {fold})", X_fold, y_fold[:, 0])

            min_y, max_y = (
                float(np.nanmin(y_fold[:, 0])),
                float(np.nanmax(y_fold[:, 0])),
            )

            for height in [
                5  # ,
                # 7
            ]:
                template = c.Template(
                    [c.TemplateNode.full_nary(branching_factor=2, depth=height - 1)], []
                )
                for operator_set, operators in [
                    (
                        "small",
                        [c.OpAdd(), c.OpSub(), c.OpMul(), c.OpDiv(), c.OpSin()],
                    ),
                    # (
                    #     "large",
                    #     [
                    #         c.OpAdd(),
                    #         c.OpSub(),
                    #         c.OpMul(),
                    #         c.OpDiv(),
                    #         c.OpSin(),
                    #         c.OpCos(),
                    #         c.OpExp(),
                    #         c.OpLog(),
                    #         c.OpSqrt(),
                    #         c.OpSquare(),
                    #     ],
                    # ),
                ]:
                    for linear_scaling in [  #
                        False,
                        # True
                    ]:
                        for constant_representation in ["ercs", "pool"]:
                            ctx = c.GPContext(
                                num_inputs=int(X_fold.shape[1]),
                                expression_template=template,
                                operators=operators,
                                constant_representation=constant_representation,
                                use_apply_buf=True,
                            )

                            for batch_size in [  #
                                # None,
                                # 32,
                                256
                            ]:
                                for run in range(REPEATS_PER_FOLD):
                                    seed = int(rng.integers(2**32))

                                    instance = c.SRProblem(
                                        ctx,
                                        X_train=c.np.load(str(X_path.absolute())),
                                        Y_train=c.np.load(str(y_path.absolute())),
                                        X_test=c.np.load(str(X_test_path.absolute())),
                                        Y_test=c.np.load(str(y_test_path.absolute())),
                                        objectives="nmse",  # = MSE / var(y_train)
                                        linear_scaling=linear_scaling,
                                        init=c.HalfHalfInit(
                                            p_terminal=0.5, p_constant=0.5
                                        ),
                                        constant_init_lower_bound=min_y,  # TODO - max y?
                                        constant_init_upper_bound=max_y,
                                        # early termination condition for "perfect" expression recovery
                                        # target_objectives=[1e-8]
                                        # if is_synthetic
                                        # else [0.0],
                                        # target_objectives=[
                                        #     # R2 >= 0.999 for black-box problems
                                        #     # and (N)MSE < 1e-8 for synthetic problems
                                        #     0.0001 if not is_synthetic else 1e-8
                                        # ],  # if is_synthetic else None,
                                        gradient_mode="forward",
                                        # gradient_mode="central",
                                        archive_epsilon=0.0 if is_synthetic else 1e-3,
                                        batch_size=batch_size,  # <= 64 is too noisy...
                                    )
                                    yield (
                                        dict(
                                            problem_name=f"{problem}{'N=' + str(kwargs.get('synthetic_problem_kwargs', {}).get('noise', 0.0)) if len(kwargs) else ''}",
                                            fold=fold,
                                            run=run,
                                            template_height=height,
                                            operator_set=operator_set,
                                            linear_scaling=linear_scaling,
                                            constant_representation=constant_representation,
                                            batch_size=batch_size,
                                        ),
                                        seed,
                                        # c.cached(
                                        instance,
                                        # cache_size=10_000,
                                        # ),
                                        ctx,
                                    )


def methods(info, ctx):
    # the GPContext provides lookup tables, one of them being the normalized pairwise node proximity
    ctx = instantiate(ctx, ctx=vars(pygom))

    # IMS options
    initial_population_size = 64
    # initial_population_size = 256
    subgeneration_factor = 6
    max_num_populations = 25  # int(np.log2(2048 / initial_population_size)) + 1
    restart_stale_populations = True  # restart the last population if it has converged

    initial_population_size = 1024
    # initial_population_size = 512
    max_num_populations = 1
    restart_stale_populations = False
    restart_stale_populations = True

    constant_representation = info["constant_representation"]
    variants = []
    if constant_representation == "ercs":
        variants += [
            ", ERCs",
            # ", ERCs + Mut",
            # ", ERCs + LM",
            # ", ERCs + LM (mut)",
            # ", ERCs + LM (central)",
        ]
    if constant_representation == "pool":
        variants += [
            ", $Pool_{10}$ + LM",
            ", $Pool_{10}$ + RV (1:1)",
            # ", $Pool_{10}$ + RV (1:2)",
            # ", $Pool_{10}$ + RVIA",
            # ", $Pool_{10}$ + RV (iu)",
            # ", $Pool_{10}$ + RV (ai)",
            # ", $Pool_{10}$ + RV (me,ce)",
            # ", $Pool_{10}$ + RV (iu,me,ce)",
            # ", $Pool_{10}$ + RV (nSDR)",
            # ", $Pool_{10}$ + RV (nrvfi)",
            # ", $Pool_{10}$ + RV (nrvfi,nmfi)",
            # ", $Pool_{10}$ + RV (nrvfi,nfi)",
            # ", $Pool_{10}$ + LM (mut)",
            # ", $Pool_{10}$ + RV (nfa)",
        ]

    for similarity in [  #
        # "$MI$",  # plain MI
        # "$MI_{adjusted}$",  # adjusted MI as per https://arxiv.org/pdf/1904.02050
        # r"$MI_{mask\ inactive}$",  # Mask inactive
        "Node",  # Normalized pairwise node proximity
        # "Node (static)",  # same, but first LT is kept throughout
        # r"Node * $MI_{mask\ inactive}$",
        # "Random",  # Random similiarty
        # r"Node * $NMI_{mask\ inactive}$",
    ]:
        # for clinkage in [
        #     "none", # do not handle constants specially, i.e. == merge for ERCs, index for pool
        #     "bin", # bin them, i.e. group them by semantic value, effectively makes constant optimization part of what linkeage learning has to do
        #     "merge" # effectively say "should this be a constant or not?" What constant etc is for the optimizer to figure out...
        # ]:
        #     pass
        copt_model_kwargs = dict(
            # treat continuous nodes semantically by binning them into 25 bins
            # as per https://arxiv.org/pdf/1904.02050
            merge_continuous=True,
            num_continuous_bins=None,
        )

        for copt in variants:
            copt_population_kwargs = dict()
            if "Mut" in copt:
                copt_population_kwargs = dict(
                    continuous_mutation_probability=1.0, **copt_population_kwargs
                )
            if "LM" in copt:
                copt_population_kwargs = dict(
                    gradient_step_frequency=1,
                    mutate_before_gradient_step="mut" in copt,
                    **copt_population_kwargs,
                )

            # evolutionary constant optimization as per https://ir.cwi.nl/pub/34425/paper_115.pdf
            rv_options = dict(
                enabled="RV" in copt,
                p_accept=0.0,
                max_nis=20,  # 25,  # + nc
                num_forced_improvement_tries=8 if "nrvfi" not in copt else 0,
                intron_aware="RVIA" in copt,  # TODO!!!
                intron_aware_intermediate_updates="RV" in copt and "iu" in copt,
                intron_aware_mean_estimation="RV" in copt and "me" in copt,
                intron_aware_cov_estimation="RV" in copt and "ce" in copt,
                intron_aware_ams="RV" in copt and "ai" in copt,
                enable_partial_ams="nfa"
                in copt,  # full FOS does not need partial AMS...
                enable_full_ams="nfa" not in copt,
                init_ams_from_population_mean=False,
                generations_until_full_evaluation=None,
            )

            discrete_model_kwargs = dict(
                # linkage learning parameters
                metric="random"
                if "Random" in similarity
                else ("nmi" if "NMI" in similarity else "mi"),
                intron_strategy="mark_only" if "active" in similarity else "none",
                normalize_initial_linkage_bias="adjusted" in similarity,
                custom_similarity=c.np.array(ctx.normalized_node_proximity().tolist())
                if "Node" in similarity
                else None,
                custom_similarity_agg="mul"
                if "*" in similarity
                else (
                    "max"
                    if "max" in similarity
                    else ("add" if "+" in similarity else None)
                ),
                # the full FOS is excluded
                filter_root=True,
                freeze="static" in similarity,
                **copt_model_kwargs,
            )

            yield (
                f'"{similarity} {copt}"',
                lambda run_path: c.MixedGOMEA(
                    discrete_model=c.LinkageTreeFOS(**discrete_model_kwargs),
                    population_options=c.PopulationOptions(
                        # intermediate_population_logfile=str(run_path / "gom_steps.csv")
                        # if info["run"] == 0
                        # else None,
                        target_continuous_to_discrete_balance=0.5
                        if "1:2" in copt
                        else 1.0,
                        forced_improvements="nfi"
                        not in copt,  # not used per default as per https://arxiv.org/pdf/1904.02050
                        enable_mixed_forced_improvements="RV" in copt
                        and "nmfi" not in copt,
                        **copt_population_kwargs,
                    ),
                    rv_options=c.RvOptions(**rv_options),
                    continuous_model=c.FullFOS(),
                    sampling_model=c.AMaLGaMSamplingModel(
                        distribution_multiplier_decrease=1.0 if "nSDR" in copt else 0.9,
                        distribution_multiplier_increase=1.0
                        if "nSDR" in copt
                        else 1.0 / 0.9,
                    ),
                    # IMS options
                    ims_options=c.IMSOptions(
                        initial_population_size=initial_population_size,
                        max_num_populations=max_num_populations,
                        subgeneration_factor=subgeneration_factor,
                        restart_stale_populations=restart_stale_populations,
                        reevaluate_solutions_after_adaption=True,
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


def status():
    import json

    total, completed = 0, 0
    for run_info, conf in tqdm(all_tasks()):
        total += 1
        run_dir = compute_run_path(LOG_DIR, run_info)

        task = extract(conf)
        task_path = run_dir / "task.yaml"

        logfile = run_dir / "stats.csv"

        done = False
        if task_path.is_file() and logfile.is_file():
            existing = load_config(task_path)

            if task == existing or json.dumps(task) == json.dumps(existing):
                with open(logfile, "r") as f:
                    *_, last_line = f.readlines()
                    status = last_line.split(",")[0]
                    if status != "Running" or status != "Aborted":
                        done = True
        if done:
            completed += 1
        else:
            print(run_dir)

    if total > 0:
        print(
            f"Status: {100.0 * (completed / total):.2f}% done ({completed} / {total})"
        )


# build introspection tools -> population diversity, rv distributions, termination & restarts (why does it effectively stop improving?)


def plot_evolution(conn, odir, run_query, max_iterations=None, p_skip=0.95):
    import bisect
    import re

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import sympy as sym
    from tqdm import tqdm

    rng = np.random.default_rng()

    creg = re.compile(r"-?\d+\.\d+")

    def remove_constant_values(expr):
        return creg.sub("c", expr)

    odir.mkdir(parents=True, exist_ok=True)

    runs = [
        r
        for r, *_ in conn.execute(
            f"SELECT DISTINCT({run_query}) AS run_id FROM gom_steps"
        ).fetchall()
    ]

    for run in tqdm(runs):
        num_constants = conn.execute(
            f"SELECT length(continuous) FROM gom_steps WHERE {run_query} = $1 LIMIT 1",
            [run],
        ).fetchone()[0]

        it = 0
        steps = []
        x = []
        best_fitness = []

        cdata = [
            dict(mean=[], q10=[], q90=[], q25=[], q75=[], num_active=[])
            for _ in range(num_constants)
        ]

        all_exprs = dict()
        all_exprs_gen = []
        frequency_counts = []

        step_types = []
        generations = []
        restarts = []

        last_best_fitness = float("inf")
        progress = tqdm(leave=False)
        num_restarts = conn.execute(
            f"SELECT MAX(current_population_restarts) FROM gom_steps WHERE {run_query} = $1",
            [run],
        ).fetchone()[0]
        for r in range(num_restarts):
            if max_iterations is not None and it > max_iterations:
                break
            if r > 0:
                restarts.append(it)
            num_generations = conn.execute(
                f"SELECT MAX(current_population_generation) FROM gom_steps WHERE {run_query} = $1 AND current_population_restarts = $2",
                [run, r],
            ).fetchone()[0]
            for g in range(num_generations):
                if max_iterations is not None and it > max_iterations:
                    break

                generations.append(it)
                num_steps = conn.execute(
                    f"SELECT MAX(gom_step) FROM gom_steps WHERE {run_query} = $1 AND current_population_restarts = $2 AND current_population_generation = $3",
                    [run, r, g],
                ).fetchone()[0]

                all_exprs_gen.append(dict())

                if num_steps is not None:
                    for s in range(num_steps):
                        if max_iterations is not None and it > max_iterations:
                            break

                        if rng.random() < p_skip:
                            it += 1
                        else:
                            x.append(it)
                            steps.append(s)
                            step_types.append(
                                conn.execute(
                                    f"SELECT step_type FROM gom_steps WHERE {run_query} = $1 AND current_population_restarts = $2 AND current_population_generation = $3 AND gom_step = $4",
                                    [run, r, g, s],
                                ).fetchone()[0]
                            )
                            it += 1

                            pop = conn.execute(
                                f"SELECT nmse_train, expressions FROM gom_steps WHERE {run_query} = $1 AND current_population_restarts = $2 AND current_population_generation = $3 AND gom_step = $4",
                                [run, r, g, s],
                            ).df()
                            best_fitness.append(pop["nmse_train"].min())
                            if best_fitness[-1] > last_best_fitness:
                                print(
                                    f"WARN best fitness got worse: {last_best_fitness} -> {best_fitness[-1]} ({step_types[-1]})"
                                )
                            last_best_fitness = best_fitness[-1]

                            # continuous per variable mean + std dev (active only & all)
                            for c in range(num_constants):
                                mean, q10, q90, q25, q75, num_active = conn.execute(
                                    f"""
                                    SELECT
                                        mean(continuous[$5]),
                                        quantile(continuous[$5], 0.1),
                                        quantile(continuous[$5], 0.9),
                                        quantile(continuous[$5], 0.25),
                                        quantile(continuous[$5], 0.75),
                                        sum(continuous_active[$5])
                                    FROM gom_steps
                                    WHERE {run_query} = $1
                                    AND current_population_restarts = $2
                                    AND current_population_generation = $3
                                    AND gom_step = $4
                                """,
                                    [run, r, g, s, c + 1],
                                ).fetchone()

                                cdata[c]["mean"].append(mean)
                                cdata[c]["q10"].append(q10)
                                cdata[c]["q90"].append(q90)
                                cdata[c]["q25"].append(q25)
                                cdata[c]["q75"].append(q75)
                                cdata[c]["num_active"].append(num_active)

                            # solution frequency (without constants)
                            pop["without_consts"] = pop["expressions"].apply(
                                remove_constant_values
                            )
                            counts = {
                                k: v
                                for k, v in pop["without_consts"].value_counts().items()
                            }
                            for k, v in counts.items():
                                all_exprs[k] = max(all_exprs.get(k, 0), v)
                                all_exprs_gen[-1][k] = max(
                                    all_exprs_gen[-1].get(k, 0), v
                                )
                            frequency_counts.append(counts)

                        progress.update()
        progress.close()

        fig, axes = plt.subplots(
            nrows=3, sharex=True, figsize=(1.25 * (len(generations) + 1), 8)
        )

        # best fitness over time
        fitness_ax = axes[0]
        fitness_ax.plot(x, best_fitness)
        fitness_ax.set_ylabel("NMSE")

        # add num discrete/continuous steps
        n_d_steps = [0 for _ in x]
        n_c_steps = [0 for _ in x]
        cg = 0
        for i in range(len(x)):
            t = step_types[i]
            n_d_steps[i] = n_d_steps[i - 1]
            n_c_steps[i] = n_c_steps[i - 1]
            while cg < len(generations) and x[i] >= generations[cg]:
                n_d_steps[i] = 0
                n_c_steps[i] = 0
                cg += 1
            n_d_steps[i] += int(t == "D")
            n_c_steps[i] += int(t == "C")
        gp_ax = fitness_ax.twinx()
        gp_ax.plot(x, n_d_steps, ls="dashed", color="black", alpha=0.5, lw=0.5)
        gp_ax.plot(x, n_c_steps, ls="dotted", color="black", alpha=0.5, lw=0.5)

        # iqr for each constant over time
        continuous_ax = axes[1]
        active_ax = continuous_ax.twinx()
        for c, color in enumerate(sns.color_palette(n_colors=num_constants)):
            continuous_ax.fill_between(
                np.array(x, dtype=np.float64),
                np.array(cdata[c]["q25"], dtype=np.float64),
                np.array(cdata[c]["q75"], dtype=np.float64),
                color=color,
                alpha=0.25,
            )
            continuous_ax.plot(x, cdata[c]["mean"], color=color)
            active_ax.plot(
                x, cdata[c]["num_active"], color=color, alpha=0.5, ls="dashed"
            )
        cylim = np.inf, -np.inf
        for ci in range(num_constants):
            q_low, q_low = conn.execute(
                f"""
                SELECT
                    quantile(continuous[$2], 0.1),
                    quantile(continuous[$2], 0.9)
                FROM gom_steps
                WHERE {run_query} = $1
                  AND continuous_active[$2]::BOOL
            """,
                [run, ci + 1],
            ).fetchone()
            cylim = min(cylim[0], q_low), max(cylim[1], q_low)
        continuous_ax.set_ylim(cylim)

        continuous_ax.set_ylabel("Constant variables")

        # expression frequency over time
        frequency_ax = axes[2]
        exprs = sorted(all_exprs.keys(), key=lambda k: all_exprs[k], reverse=True)
        expr_palette = dict()
        for expr, color in zip(exprs, sns.color_palette(n_colors=len(exprs))):
            counts = [frequency_counts[i].get(expr, 0) for i in range(len(x))]
            frequency_ax.plot(x, counts, color=color)
            expr_palette[expr] = color

        # add the expression text for the most frequenc expression each generation
        g_start, last_most_frequent = 0, ""
        for g, g_start in enumerate(generations):
            g_exprs_sorted = sorted(
                all_exprs_gen[g].keys(), key=lambda k: all_exprs_gen[g][k], reverse=True
            )
            if len(g_exprs_sorted) > 0:
                most_frequent = g_exprs_sorted[0]
                g_end = generations[g + 1] if g + 1 < len(generations) else x[-1]
                gx = (g_end + g_start) // 2
                ix = bisect.bisect_left(x, gx)

                if most_frequent != last_most_frequent:
                    last_most_frequent = most_frequent
                    gy = frequency_counts[ix].get(most_frequent, 0)
                    gy = all_exprs_gen[g][most_frequent]
                    frequency_ax.text(
                        gx,
                        gy,
                        most_frequent,
                        ha="center",
                        va="center",
                        color=expr_palette[most_frequent],
                    )
        frequency_ax.set_ylabel("Expression Frequency")

        # add indicators for each generation/restart boundary
        for ax in axes.flat:
            for g in generations:
                ax.axvline(g, color="black", ls="dotted", lw=2.0)
            for r in restarts:
                ax.axvline(r, color="black", ls="dashed", lw=4.0)

        # add second x-axis with generation index
        s = frequency_ax.secondary_xaxis(location=0)
        s.set_xticks(
            [
                (generations[i] + generations[i + 1]) // 2
                for i in range(1, len(generations) - 1)
            ],
            labels=[f"G {i}" for i in range(1, len(generations) - 1)],
        )
        s.tick_params("x", length=20, width=0.0)

        fig.savefig(odir / f"{run}.pdf", dpi=600, bbox_inches="tight", transparent=True)

        # plt.show()

        plt.close(fig)

    # problem_name, method_name, run, fold
    # exit()


def main():
    status()

    exit()

    # TODO add dry run option that only checks how many jobs would be run (per cpu)
    run_tasks(
        LOG_DIR,
        all_tasks(),
        # clean=True,
        # limit=1,
        # max_workers=1,  # server has 44 physical cores
    )

    preprocess = True
    # preprocess = False
    with load_results(
        LOG_DIR,
        file_pattern="stats",
        # enable pre-processing the .csv logs into .parquet files
        preprocess=preprocess,
        parquet_dir=PARQUET_DIR / "stats",
    ) as conn:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        # load_results(
        #     LOG_DIR,
        #     file_pattern="gom_steps",
        #     table_name="gom_steps",
        #     types=dict(gom_step="UINTEGER", step_type="VARCHAR"),
        #     # enable pre-processing the .csv logs into .parquet files
        #     preprocess=preprocess,
        #     parquet_dir=PARQUET_DIR / "gom_steps",
        #     conn=conn,
        # )
        # plot_evolution(
        #     conn,
        #     PLOT_DIR / "evolution",
        #     "format('{}.{}.{}.{}', problem_name, method_name, fold, run)",
        # )

        for split in ["train", "test"]:
            plot_convergence_so(
                PLOT_DIR / f"convergence_{split}",
                conn,
                # y_var="1.0 - objectives[1]::DOUBLE",  # transform NMSE into R2
                y_var=f"1.0 - nmse_{split}::DOUBLE",
                y_agg="MAX",  # higher R^2 is better
                y_label=f"$R^2$ {split.title()}",
                ymin="auto",
                ymax="auto",
                # method_query="format('Reeval={} BS={}', IF(contains(method_name, 'Reeval'), 'Yes', 'No') , batch_size::STRING)",
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
