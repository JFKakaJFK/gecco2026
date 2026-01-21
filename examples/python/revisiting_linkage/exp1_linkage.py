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

SEED = 42

BUDGET = c.Budget(max_evaluations=int(1e7))


def problems(rng):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    seeds = [int(rng.integers(2**32)) for _ in range(REPEATS_PER_FOLD)]

    for problem in [
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

            for height in [5, 7]:
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
                    for linear_scaling in [False, True]:
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
    initial_population_size = 64
    subgeneration_factor = 10
    max_num_populations = 25  # int(np.log2(2048 / initial_population_size)) + 1
    restart_stale_populations = True  # restart the last population if it has converged

    for similarity in [  #
        # "$MI$",  # plain MI
        # "$MI_{adjusted}$",  # adjusted MI as per https://arxiv.org/pdf/1904.02050
        # r"$MI_{mask\ inactive}$",  # Mask inactive
        # r"$MI_{any\ active}$",  # Mask inactive + only consider partially active variables/variable pairs
        # r"$MI_{all\ active}$",  # Mask inactive + only consider fully active variables
        # "Node",  # Normalized pairwise node proximity
        # "Node (static)",  # same, but first LT is kept throughout
        # r"Node * $MI_{mask\ inactive}$",
        # "Random",  # Random similiarty
        # "Node (wVIG)",
        # "Node (wVIG, static)",
        "Node (peter)",
        "Node (peter, static)",
        "Univariate"
    ]:
        custom_similarity = None
        if "Node" in similarity:
            if "wVIG" in similarity:
                # S = np.array(ctx.normalized_node_proximity().tolist())
                # depth = int(np.log2(1 + ctx.num_discrete) - 1)
                # S = ((S - 1.0) * -(1 + 2 * depth)) # proximity to distance
                # # print(depth, S, np.where(S > 0, 1.0 / S, 0.0))
                # S = np.where(S > 0, 1.0 / S, 0.0) # distance as per wVIG
                # custom_similarity=c.np.array(S.tolist())

                # print(np.array(ctx.normalized_w_vig().tolist()))

                custom_similarity=c.np.array(ctx.normalized_w_vig().tolist())
            elif "peter" in similarity:
                S = np.array(ctx.subtree_co_occurrences().tolist())
                import seaborn as sns
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                sns.heatmap(S, annot=True, ax=ax, cmap="Blues")
                fig.savefig(f"sim_peter_{ctx.num_discrete}.png", dpi=600, bbox_inches="tight")

                custom_similarity=c.np.array(ctx.subtree_co_occurrences().tolist())
            else:
                custom_similarity=c.np.array(ctx.normalized_node_proximity().tolist())
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
            f'"{similarity}"',
            c.MixedGOMEA(
                discrete_model= c.UnivariateFOS() if similarity == "Univariate" else c.LinkageTreeFOS(**discrete_model_kwargs),
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
    rng = np.random.default_rng(seed=SEED)
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
    # exit()

    # TODO add dry run option that only checks how many jobs would be run (per cpu)
    # run_tasks(
    #     LOG_DIR,
    #     all_tasks(),
    #     # clean=True,
    #     # limit=1,
    #     max_workers=44,  # server has 44 physical cores
    # )

    with load_results(
        LOG_DIR,
        # enable pre-processing the .csv logs into .parquet files
        preprocess=True,
        parquet_dir=PARQUET_DIR,
    ) as conn:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

        for split in [  #
            "train",
            "test",
        ]:
            plot_convergence_so(
                PLOT_DIR / f"convergence_{split}",
                conn,
                # y_var="1.0 - objectives[1]::DOUBLE",  # transform NMSE into R2
                y_var=f"1.0 - nmse_{split}",
                y_agg="MAX",  # higher R^2 is better
                y_label=f"$R^2$ {split.title()}",
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
