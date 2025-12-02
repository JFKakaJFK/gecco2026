import os
import pathlib
import re
from urllib.parse import quote

import numpy as np
import pandas as pd
import sympy as sym
from pmlb import fetch_data
from pygom import *
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

from src.config import Config, c
from src.plots import plot_convergence_so
from src.postprocessing import load_results
from src.run import run_tasks

REPEATS_TOTAL = 25
NUM_FOLDS = 5

REPEATS_PER_FOLD = REPEATS_TOTAL // NUM_FOLDS

DOMAIN = os.path.basename(__file__).split(".")[0]
OUTPUT_DIRECTORY = pathlib.Path("results") / f"{DOMAIN}_raw"
PARQUET_DIR = OUTPUT_DIRECTORY.parent / f"{DOMAIN}_processed"

PLMB_CACHE_DIR = ".pmlb/datasets"

budget = c.Budget(max_evaluations=int(1e6))


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
        # "-4.2 * x0 + 1.41 * x1 + 2.72 * x2",
        # "0.1 * x0 + 0.2 * x1 + 2.4 * x2",
        "sin(3.1 * x0 + 2.7)",
        "210_cloud",
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


def methods(rng):
    initial_population_size = 2
    subgeneration_factor = 4
    max_num_populations = 25
    initial_population_size = 1024
    max_num_populations = 1
    restart_stale_populations = False  # True  # by default, we want this only for "relevant" populations that still contribute new improvements - so only the largest population in single-objective population, but due to forced improvements any population in a multi-objective settings

    for metric in [  #
        "mi",
        # "nmi",
    ]:
        for forced_improvements in [  #
            True,
            # False,
        ]:
            common_desc = f"{metric.upper()}{', FI' if forced_improvements else ''}"

            for dsp in [  #
                0.0,
                # 0.05,
                # 1.0,
            ]:
                common_options = dict(
                    population_options=c.PopulationOptions(
                        donor_search_proportion=dsp,
                        forced_improvements=forced_improvements,
                    ),
                    ims_options=c.IMSOptions(
                        initial_population_size=initial_population_size,
                        max_num_populations=max_num_populations,
                        subgeneration_factor=subgeneration_factor,
                        restart_stale_populations=restart_stale_populations,
                    ),
                )
                dsp_desc = f", DSP={dsp}" if dsp > 0.0 else ""
                desc = common_desc + dsp_desc

                for intron_strategy in [
                    "none",
                    "any_active",
                    "weighted_any_active",
                    # "all_active",  # definitely bad, as expected
                    "mark_only",
                ]:
                    is_desc = dict(
                        none="",
                        any_active=r", $IA_{any}$",
                        weighted_any_active=r", $IA_{w,any}$",
                        all_active=r", $IA_{all}$",
                        mark_only=r", $IA_{mark}$",
                    )[intron_strategy]
                    yield (
                        f'"Mixed ({desc}{is_desc})"',
                        c.MixedGOMEA(
                            discrete_model=c.LinkageTreeFOS(
                                metric=metric,
                                intron_strategy=intron_strategy,
                                filter_root=True,
                            ),
                            rv_options=c.RvOptions(
                                max_nis=25,  # + nc
                                init_ams_from_population_mean=False,
                                intron_aware=False,
                                generations_until_full_evaluation=None,
                            ),
                            sampling_model=c.AMaLGaMSamplingModel(),
                            **common_options,
                        ),
                    )


def all_jobs():
    rng = np.random.default_rng(seed=42)
    for info, instance in problems(rng):
        instance_seed = int(rng.integers(2**32 - 1))
        for method_name, method in methods(rng):
            yield (
                dict(method_name=method_name, **info),
                dict(
                    instance=instance, method=method, budget=budget, seed=instance_seed
                ),
            )


def main():
    run_tasks(
        OUTPUT_DIRECTORY,
        all_jobs(),
        num_repeats=REPEATS_PER_FOLD,
        clean=True,
        # limit=1,
        # max_workers=1,
    )

    preprocess = True
    # preprocess = False

    plot_dir = pathlib.Path("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    with load_results(
        OUTPUT_DIRECTORY,
        parquet_dir=PARQUET_DIR,
        preprocess=preprocess,
    ) as conn:
        plot_convergence_so(
            plot_dir / f"{DOMAIN}_convergence",
            conn,
            y_agg="MIN",
            ymin="auto",
            ymax="auto",
            # y_var="objectives[1]::DOUBLE",
            y_label="MSE",
            unit_query="format('{}.{}', fold, run)",
            # metrics=["evaluations / dims::DOUBLE", "total_time_seconds / dims::DOUBLE"],
            # metric_labels=[r"$\frac{Evaluations}{Dimensions}$", "Time/Dimensions [s]"],
            modifier_query="[num_rows::STRING, template_height::STRING,operator_set::STRING,constant_representation::STRING,IF(linear_scaling, 'Yes', 'No')::STRING,init_name::STRING]",
            modifier_labels=["#Rows", "Height", "Operators", "CRep", "LS", "Init"],
            # show_generation_boundaries=True,
            nsamples=50,
        )


if __name__ == "__main__":
    main()
