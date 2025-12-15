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

REPEATS_PER_DATASET = 30
NUM_FOLDS = 5

REPEATS_PER_FOLD = REPEATS_PER_DATASET // NUM_FOLDS

RESULT_DIR = pathlib.Path("results") / "linkage"
DATA_DIR = RESULT_DIR / "data"
LOG_DIR = RESULT_DIR / "raw"
PARQUET_DIR = RESULT_DIR / "processed"
PLOT_DIR = RESULT_DIR / "plots"

BUDGET = c.Budget(
    # max_evaluations=int(1e4)
    max_evaluations=int(1e7)
)


def problems(rng):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for problem in [
        # UCI
        "Airfoil",
        "Bike Sharing",
        # "Breast Cancer",
        "Concrete Compressive Strength",
        "Dow Chemical",
        "Tower",
        # "Energy Cooling",
        # "Energy Heating",
        # "Yacht Hydrodynamics",
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

            for height in [5, 7]:
                template = c.Template(
                    [c.TemplateNode.full_nary(branching_factor=2, depth=height - 1)], []
                )
                for operator_set, operators in [
                    (
                        "small",
                        [c.OpAdd(), c.OpSub(), c.OpMul(), c.OpDiv(), c.OpSin()],
                    )
                ]:
                    for linear_scaling in [False, True]:
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
    initial_population_size = 64
    subgeneration_factor = 6
    max_num_populations = 25  # int(np.log2(2048 / initial_population_size)) + 1
    restart_stale_populations = True  # restart the last population if it has converged

    for similarity in [  #
        "$MI$",  # plain MI
        "$MI_{adjusted}$",  # adjusted MI as per https://arxiv.org/pdf/1904.02050
        r"$MI_{any\ active}$",  # Mask inactive and normalize, keeping only at least partialy active pairs
        r"$MI_{mask\ inactive}$",  # Mask inactive
        "Node",  # Normalized pairwise node proximity
        "Random",  # Random similiarty
    ]:
        discrete_model_kwargs = dict(
            # linkage learning parameters
            metric="random" if "Random" in similarity else "mi",
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
        )

        yield (
            f'"GP-GOMEA ({similarity})"',
            c.MixedGOMEA(
                discrete_model=c.LinkageTreeFOS(**discrete_model_kwargs),
                population_options=c.PopulationOptions(
                    forced_improvements=False,  # not used per default as per https://arxiv.org/pdf/1904.02050
                ),
                # disable evolutionary constant optimization as per https://ir.cwi.nl/pub/34425/paper_115.pdf
                rv_options=c.RvOptions(enabled=False),
                continuous_model=c.FullFOS(),
                sampling_model=c.AMaLGaMSamplingModel(),
                # IMS options
                ims_options=c.IMSOptions(
                    initial_population_size=initial_population_size,
                    max_num_populations=max_num_populations,
                    subgeneration_factor=subgeneration_factor,
                    restart_stale_populations=restart_stale_populations,
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


def main():
    # status()

    # TODO add dry run option that only checks how many jobs would be run (per cpu)
    # run_tasks(
    #     LOG_DIR,
    #     all_tasks(),
    #     # clean=True,
    #     # limit=1,
    #     # max_workers=1,
    # )

    with load_results(
        LOG_DIR,
        # enable pre-processing the .csv logs into .parquet files
        preprocess=True,
        parquet_dir=PARQUET_DIR,
    ) as conn:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
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
