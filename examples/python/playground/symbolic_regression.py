import os
import pathlib
import re
from urllib.parse import quote

import numpy as np
import sympy as sym
from pmlb import fetch_data
from pygom import *
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

from src.config import Config, c
from src.plots import plot_convergence_so
from src.postprocessing import load_results
from src.run import run_tasks

REPEATS_TOTAL = 15
NUM_FOLDS = 5

REPEATS_PER_FOLD = REPEATS_TOTAL // NUM_FOLDS

DOMAIN = os.path.basename(__file__).split(".")[0]
OUTPUT_DIRECTORY = pathlib.Path("results") / f"{DOMAIN}_raw"
PARQUET_DIR = OUTPUT_DIRECTORY.parent / f"{DOMAIN}_processed"

PLMB_CACHE_DIR = ".pmlb/datasets"

budget = c.Budget(max_evaluations=int(5e5))


"""
# Method to the madness

## Motivation

Experiment typically is running some methods on some problems
- Can be the same method with different parameters or different methods
- Same problem can have multiple "instances" (number of parameters, input graphs for maxcut, folds or noise levels for SR etc.)
- Some parameters/settings sometimes are part of the method, but sometimes also part of the problem
    - initialization: typically a problem property in continuous optimization, but an algorithm thing for SR
    - even algorithm parameters: operator set size, template size etc
=> Flexibility is nice, so allow everything with the following assumptions
    - problems have a name (`problem_name`), either for one or multiple instances
    - methods have a name (`method_name`), can be different parameters for different problems or the same thing
    - there can be "modifiers" (e.g. number of parameters, operator set size)
        - essentially the thing that defines how to split up the resulting plots
-> pretty much all of that can be customized by adding more info to the logfile + a corresponding SQL query

## Implementation

- Tasks/jobs/single runs are specified by
    - the info to log for the given combination of experiment settings
    - and the "Config", a blueprint of what to actually instantiate and run (internally this part is cursed - the `c` object swallows any invocation in the supported scopes {at the point of writing this pygom and np}, allows writing that to a file and instantiating the corresponding wrapped objects)
        - it's almost like just using the generated bindings directly, and removing the `c.` should provide editor support
    - currently, this is somehow merged together by consuming generators yielding the different combinations (also cursed, but to an extent the sanest approach I'm aware of)
        - yes, it's JSON stored as .yaml (yaml would be a lot less verbose, but isn't working yet since there are no tuples...)
- Running at this point "just" gobbles up all tasks and runs them with a concurrent.futures ProcessPoolExecutor, skipping already present runs if the specification matches
- Postprocessing then is an optional step that collects all logs and stores them as parquet files for faster processing
- Any analysis can then query the resulting database view using SQL
    - e.g. for plotting a so far useful abstraction is having one (or several in a loop) y variable that is plotted on a grid of "problems" and "modifier" combinations; individual lines are methods and separate files are stored for different x-axis "metrics"

Note: versioning experiments is something that in my experience is always a mess
"""


def lambdify_expression(e: str | sym.Expr):
    """Converts a `sympy` compatible expression string into a function accepting a dataset `X`."""
    e = str(e)

    symbols = {x: sym.Symbol(x) for x in re.findall(r"(x\d+)", e)}
    expr = sym.sympify(e, locals=symbols)
    f = sym.lambdify(symbols.values(), expr, modules=[{"clip": np.clip}, "numpy"])

    def fn(X: np.ndarray):
        try:
            return f(*[X[:, int(s[1:])] for s in symbols.keys()])
        except Exception as e:
            print(e)
            return np.repeat(float("nan"), X.shape[0])

    return fn


def synthetic_problem(
    expr: str,
    size: int = 1000,
    lb: float = -10.0,
    ub: float = 10.0,
    noise: float = 0.01,
    seed: int | None = None,
):
    """Creates a synthetic problem by sampling a random dataset, applying the function and possibly adding noise."""
    assert ub > lb, "Invalid initialisation bounds"

    rng = np.random.Generator(np.random.Philox(seed=seed))

    num_inputs = max([int(x) + 1 for x in re.findall(r"x(\d+)", expr)])

    X = rng.random(size=(size, num_inputs)) * (ub - lb) + lb
    y = lambdify_expression(expr)(X)
    if noise > 0:
        y += rng.standard_normal(size) * noise
    return X, y


def problems(rng):
    # current run expects (problem_name, dims, config) tuples
    # - dims is a bit of a misnomer for GP, but the notion of having a "problem complexity" axis we want to explore still holds
    # - also the problems are actually "instances" and SR is ML so we want to have different folds, so different instances of the same problem are defined to mirror that

    data_dir = OUTPUT_DIRECTORY / "data"
    os.makedirs(data_dir, exist_ok=True)

    for problem in [  # any sympy compatible equation or pmlb dataset should work
        # TODO select datasets
        "sin(3.1 * x0 + 2.7)",
        "210_cloud",
        "522_pm10",
        "1027_ESL",
        "1089_USCrime",
    ]:
        eq_dir = data_dir / quote(problem, safe=" ")
        eq_dir.mkdir(parents=True, exist_ok=True)
        for num_rows in [250]:
            try:
                X, y = fetch_data(
                    problem, return_X_y=True, local_cache_dir=PLMB_CACHE_DIR
                )
                X, y = (
                    X[:num_rows],
                    y[:num_rows],
                )  # TODO stratify instead of truncating?
            except Exception as e:
                X, y = synthetic_problem(
                    problem, num_rows, noise=0.0, seed=rng.integers(2**32 - 1)
                )
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=rng.integers(2**32 - 1)
            )

            # the "task" needs to be transferrable across processess -> save the data as .csv and load it there again
            X_test_path, y_test_path = (
                eq_dir / "X_test.npy",
                eq_dir / "y_test.npy",
            )
            np.save(X_test_path, X_test, allow_pickle=False)
            np.save(y_test_path, y_test.reshape(-1, 1), allow_pickle=False)

            # the rationale here is that we want comparable test performance across folds - so the training set is split again and the validation portion is simply not used (for speed, train/test doesn't really matter though...)
            kf = KFold(
                n_splits=NUM_FOLDS, shuffle=True, random_state=rng.integers(2**32 - 1)
            )
            for fold, (train_indices, _validation_indices) in enumerate(
                kf.split(X_train)
            ):
                X_fold, y_fold = (
                    X_train[train_indices, :],
                    y_train[train_indices].reshape(-1, 1),
                )

                X_path, y_path = (
                    eq_dir / f"X{fold:03d}.npy",
                    eq_dir / f"y{fold:03d}.npy",
                )
                np.save(X_path, X_fold, allow_pickle=False)
                np.save(y_path, y_fold, allow_pickle=False)

                for height, template in [
                    (
                        5,
                        c.Template(
                            [c.TemplateNode.full_nary(branching_factor=2, depth=4)], []
                        ),
                    ),
                    # (
                    #     7,
                    #     c.Template(
                    #         [c.TemplateNode.full_nary(branching_factor=2, depth=6)], []
                    #     ),
                    # ),
                ]:
                    for operator_set, operators in [
                        (
                            "small",
                            [c.OpAdd(), c.OpSub(), c.OpMul(), c.OpDiv(), c.OpSin()],
                        )
                    ]:
                        for constant_representation in ["none", "ercs"]:
                            for linear_scaling in [False, True]:
                                for init_name, init in [("HalfHalf", c.HalfHalfInit())]:
                                    ctx = c.GPContext(
                                        num_inputs=int(X_fold.shape[1]),
                                        expression_template=template,
                                        operators=operators,
                                        constant_representation=constant_representation,
                                    )

                                    # for multiple different functions, branch here
                                    # for device in ["CPU"]:
                                    # and then in the plotting code use the method query to append e.g. device to the method name -> comparison plots are in the same plt.Axes object

                                    yield (
                                        dict(
                                            problem_name=problem,
                                            fold=fold,
                                            num_rows=num_rows,
                                            template_height=height,
                                            operator_set=operator_set,
                                            constant_representation=constant_representation,
                                            linear_scaling=linear_scaling,
                                            init_name=init_name,
                                            # this just attaches columns to the logfile (the load_results function allows specifying a type, but duckdb's defaults are usually pretty good) - so it can be abused to pass information to the plotting code (totally cursed yes, but in my experience way better than matching and expanding the logs afterwards)
                                            var_y=float(np.var(y_fold[:, 0])),
                                            var_y_test=float(np.var(y_test)),
                                        ),
                                        c.SRProblem(
                                            ctx,
                                            x_train=c.np.load(str(X_path.absolute())),
                                            y_train=c.np.load(str(y_path.absolute())),
                                            x_test=c.np.load(
                                                str(X_test_path.absolute())
                                            ),
                                            y_test=c.np.load(
                                                str(y_test_path.absolute())
                                            ),
                                            objectives="mse",
                                            linear_scaling=linear_scaling,
                                            init=init,
                                            constant_init_lower_bound=-10.0,
                                            constant_init_upper_bound=10.0,
                                            target_objectives=[
                                                1e-6
                                            ],  # for the synthetic problems we can get very good errors - no point in continuing if the error is effectively 0
                                        ),
                                    )


def methods(rng, info):
    initial_population_size = 64
    subgeneration_factor = 10
    max_num_populations = int(np.log2(2048 / initial_population_size)) + 1
    # initial_population_size = 1024
    # max_num_populations = 1
    restart_stale_populations = True  # True  # by default, we want this only for "relevant" populations that still contribute new improvements - so only the largest population in single-objective population, but due to forced improvements any population in a multi-objective settings

    # TODO:
    # - different operator sets, different heights, LS/no LS, IMS/no IMS
    # - effect of MI/NMI
    # - effect of filtering (Yes/No)
    # - effect of different init variations (HH, RC)
    # - effect of intron aware linkage learning (none, any,... vs LN)
    # - effect of different constant variations
    #   - how to represent (none, ERCs, Pool10, Pool20)
    #   - how to handle in linkage learning (merge, bins, pool + no merge/bins)
    #   - how to optimize (CM, CM + LM, RV[pool only])
    # - comparison to other methods -> SRBench 2025
    for metric in [  #
        "mi",
        # "nmi",
    ]:
        for forced_improvements in [  #
            True,
            # False,
        ]:
            for filter in [  #
                # True,
                False
            ]:
                filter_kws = (
                    dict(
                        filter_parent_threshold=1e-6,
                        filter_children_threshold=1.0 - 1e-6,
                    )
                    if filter
                    else dict()
                )
                common_desc = f"{metric.upper()}{', FI' if forced_improvements else ''}{', filtered' if filter else ''}"

                for dsp in [  #
                    0.0,
                    # 0.05,
                    # 1.0,
                ]:
                    dsp_desc = f", DSP={dsp}" if dsp > 0.0 else ""

                    for linkage_normalization in [False, True]:
                        ln_desc = ", LN" if linkage_normalization else ""
                        for intron_strategy in [
                            # "none",
                            # "any_active", # weighting tends to always be better...
                            "weighted_any_active",
                            # "all_active",  # definitely bad, as expected
                            # "mark_only",
                        ]:
                            if linkage_normalization and intron_strategy != "none":
                                continue

                            is_desc = dict(
                                none="",
                                any_active=r", $IA_{any}$",
                                weighted_any_active=r", $IA_{w,any}$",
                                all_active=r", $IA_{all}$",
                                mark_only=r", $IA_{mark}$",
                            )[intron_strategy]

                            if info["constant_representation"] != "none":
                                for opt in [None, "cmut", "lm", "rv"]:
                                    copt_model_kws = dict(
                                        merge_continuous=False,
                                        num_continuous_bins=25,
                                    )
                                    copt_population_kws = dict()
                                    copt_rv_kws = dict(enabled=False)
                                    copt_desc = ""
                                    if opt is not None:
                                        if "cmut" in opt:
                                            copt_population_kws = dict(
                                                continuous_mutation_probability=1.0,
                                                **copt_population_kws,
                                            )
                                            copt_desc += ", cmut"
                                        if "lm" in opt:
                                            copt_population_kws = dict(
                                                gradient_step_frequency=1,
                                                **copt_population_kws,
                                            )
                                            copt_desc += ", lm"
                                        if (
                                            "rv" in opt
                                        ):  # not really supposed to be used with ERCs, but whatever for now
                                            copt_model_kws = dict(
                                                merge_continuous=True,
                                                num_continuous_bins=None,
                                            )
                                            copt_rv_kws = dict(
                                                enabled=True,
                                                max_nis=25,  # + nc
                                                intron_aware=True,
                                                enable_partial_ams=False,  # full FOS does not need partial AMS...
                                                init_ams_from_population_mean=False,
                                                generations_until_full_evaluation=None,
                                            )
                                            copt_desc += ", rv"

                                    desc = (
                                        common_desc
                                        + ln_desc
                                        + dsp_desc
                                        + is_desc
                                        + copt_desc
                                    )
                                    yield (
                                        f'"Mixed ({desc})"',
                                        c.MixedGOMEA(
                                            discrete_model=c.LinkageTreeFOS(
                                                metric=metric,
                                                intron_strategy=intron_strategy,
                                                filter_root=True,
                                                normalize_initial_linkage_bias=linkage_normalization,
                                                **copt_model_kws,
                                                **filter_kws,
                                            ),
                                            rv_options=c.RvOptions(
                                                **copt_rv_kws,
                                            ),
                                            continuous_model=c.FullFOS(),
                                            sampling_model=c.AMaLGaMSamplingModel(),
                                            population_options=c.PopulationOptions(
                                                donor_search_proportion=dsp,
                                                forced_improvements=forced_improvements,
                                                **copt_population_kws,
                                            ),
                                            ims_options=c.IMSOptions(
                                                initial_population_size=initial_population_size,
                                                max_num_populations=max_num_populations,
                                                subgeneration_factor=subgeneration_factor,
                                                restart_stale_populations=restart_stale_populations,
                                            ),
                                        ),
                                    )


def all_jobs():
    rng = np.random.default_rng(seed=42)
    for info, instance in problems(rng):
        # each fold gets the same seed, but the `run_tasks` function then adds the run-number to each repetition
        # - not a great fix, but now different repetitions for the same fold produce different results...
        instance_seed = int(rng.integers(2**32 - 1))
        for method_name, method in methods(rng, info):
            yield (
                dict(method_name=method_name, **info),
                dict(
                    instance=instance, method=method, budget=budget, seed=instance_seed
                ),
            )


def main():
    # TODO set up a cli arg parser...
    preprocess = True
    # preprocess = False

    run_tasks(  # TODO add dry run option that only checks how many jobs would be run (per cpu)
        OUTPUT_DIRECTORY,
        all_jobs(),
        num_repeats=REPEATS_PER_FOLD,
        clean=True,
        # limit=1,
        # max_workers=1,
    )

    plot_dir = pathlib.Path("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    with load_results(
        OUTPUT_DIRECTORY,
        parquet_dir=PARQUET_DIR,
        preprocess=preprocess,
    ) as conn:
        # currently the convergence plot does not support only plotting a subset of variables at a time...
        # for nmi in [False, True]:
        #     methods = sorted(
        #         m
        #         for m, *_ in conn.execute(
        #             "SELECT DISTINCT(method_name) FROM results WHERE contains(method_name, 'NMI') = $1",
        #             [nmi],
        #         ).fetchall()
        #     )

        plot_convergence_so(
            # plot_dir / f"{DOMAIN}_convergence_{['mi', 'nmi'][nmi]}",
            plot_dir / f"{DOMAIN}_convergence",
            conn,
            # methods=methods, # TODO add support for plotting only some variables at a time...
            y_agg="MAX",
            ymin=None,  # "auto",  # cuts of the lower 2.5% of data points
            ymax=None,
            # SQL can be used to derive values...
            y_var="1 - objectives[1]::DOUBLE / var_y::DOUBLE",
            ylabel="$R^2$ Train",
            unit_query="format('{}.{}', fold, run)",
            # metrics=["evaluations / dims::DOUBLE", "total_time_seconds / dims::DOUBLE"],
            # metric_labels=[r"$\frac{Evaluations}{Dimensions}$", "Time/Dimensions [s]"],
            modifier_query="[num_rows::STRING, template_height::STRING,operator_set::STRING,constant_representation::STRING,IF(linear_scaling, 'Yes', 'No')::STRING,init_name::STRING]",
            modifier_labels=[
                "#Rows",
                "Height",
                "Operators",
                "Constants",
                "LS",
                "Init",
            ],
            # show_generation_boundaries=True,
            nsamples=100,
        )


if __name__ == "__main__":
    main()
