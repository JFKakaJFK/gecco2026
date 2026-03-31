import pathlib

import numpy as np
import pygom
from pygom import *
from src.config import c, extract, instantiate, load_config
from src.data import prepare_problem, problem_info
from src.plots import plot_convergence_so
from src.postprocessing import load_results
from src.run import compute_run_path, run_tasks
from tqdm import tqdm

REPEATS_PER_DATASET = 30
NUM_FOLDS = 5

REPEATS_PER_FOLD = REPEATS_PER_DATASET // NUM_FOLDS

RESULT_DIR = pathlib.Path("results") / "rvterm4"
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
        "Dow Chemical",
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
                                        constant_init_lower_bound=min_y,
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
                                        archive_epsilon=0.0,  # if is_synthetic else 1e-6,
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
                c.MixedGOMEA(
                    discrete_model=c.LinkageTreeFOS(**discrete_model_kwargs),
                    population_options=c.PopulationOptions(
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
            yield (
                dict(method_name=method_name, **info),
                dict(instance=instance, method=method, budget=BUDGET, seed=seed),
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


def main():
    # status()

    # exit()

    # TODO add dry run option that only checks how many jobs would be run (per cpu)
    run_tasks(
        LOG_DIR,
        all_tasks(),
        # clean=True,
        # limit=1,
        # max_workers=1,  # server has 44 physical cores
    )

    # exit()

    with load_results(
        LOG_DIR,
        # enable pre-processing the .csv logs into .parquet files
        preprocess=True,
        parquet_dir=PARQUET_DIR,
    ) as conn:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

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
