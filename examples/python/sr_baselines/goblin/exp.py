import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pygom.gp as gp
from pmlb import fetch_data
from pygom import default_termination_callback
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

MAX_EVALUATIONS = int(5e5)  # int(1e6)
NUM_RUNS = 30
MAX_NUM_ROWS = 1000
PLMB_CACHE_DIR = "../pmlb/datasets"

RESULT_DIR = "../raw"
RESULT_DIR = "../results"

# black_box from https://arxiv.org/abs/2505.03977v1
DATASETS = [
    "1028_SWD",
    "1089_USCrime",
    "1193_BNG_lowbwt",
    "1199_BNG_echoMonths",
    "192_vineyard",
    "210_cloud",
    "522_pm10",
    "557_analcatdata_apnea1",
    "579_fri_c0_250_5",
    "606_fri_c2_1000_10",
    "650_fri_c0_500_50",
    "678_visualizing_environmental",
]


def all_runs():
    for ds in DATASETS:  # [:5]:  # TODO
        X, y = fetch_data(ds, return_X_y=True, local_cache_dir=PLMB_CACHE_DIR)
        # if X.shape[0] > MAX_NUM_ROWS:  # TODO
        #     continue

        X, y = X[:MAX_NUM_ROWS], y[:MAX_NUM_ROWS]

        assert X.shape[0] <= MAX_NUM_ROWS
        assert y.shape[0] <= MAX_NUM_ROWS

        for run in range(NUM_RUNS):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=run + 1
            )

            for ls in [False, True]:
                yield MAX_EVALUATIONS, ls, ds, run, X_train, X_test, y_train, y_test


termination_callback = default_termination_callback
termination_callback = None  # comment this to make Ctrl+C work while the algorithm is running, otherwise it takes until the C++ code under the hood returns (can take a while, the other alternative is to manually kill the python process)
# TODO currently the overhead of the python callback is considerable - check that less often
params = {
    # the default, compares to the original version
    "Mixed (MI)": lambda ls, max_evals: dict(
        linear_scaling=ls,
        budget_kwargs=dict(
            max_evaluations=max_evals, termination_callback=termination_callback
        ),
        ims_kwargs=dict(
            initial_population_size=1024,
            max_num_populations=1,
            restart_stale_populations=True,
        ),
        rv_kwargs=dict(enabled=False),
        population_kwargs=dict(),
        discrete_model_kwargs=dict(
            metric="mi",
            intron_strategy="none",
            filter_root=True,
            merge_continuous=False,
            num_continuous_bins=25,
            normalize_initial_linkage_bias=True,
        ),
    ),
    "Mixed (Node)": lambda ls, max_evals: dict(
        linear_scaling=ls,
        budget_kwargs=dict(
            max_evaluations=max_evals, termination_callback=termination_callback
        ),
        ims_kwargs=dict(
            initial_population_size=1024,
            max_num_populations=1,
            restart_stale_populations=True,
        ),
        rv_kwargs=dict(enabled=False),
        population_kwargs=dict(),
        discrete_model_kwargs=dict(
            metric="node_proximity",
        ),
    ),
    # "Mixed (RV)": lambda ls, max_evals: dict(
    #     constant_representation="pool",
    #     linear_scaling=ls,
    #     budget_kwargs=dict(
    #         max_evaluations=max_evals, termination_callback=termination_callback
    #     ),
    #     ims_kwargs=dict(
    #         initial_population_size=1024,
    #         max_num_populations=1,
    #         restart_stale_populations=True,
    #     ),
    #     rv_kwargs=dict(
    #         enabled=True,
    #         enable_partial_ams=False,
    #         p_accept=0.0,
    #         max_nis=20,
    #     ),
    #     sampling_model="AMaLGaMSamplingModel",
    #     population_kwargs=dict(),
    #     discrete_model_kwargs=dict(
    #         metric="mi",
    #         intron_strategy="none",
    #         filter_root=True,
    #         merge_continuous=False,
    #         num_continuous_bins=25,
    #         normalize_initial_linkage_bias=True,
    #     ),
    # ),
}


def run_one(
    version,
    max_evals,
    linearscaling,
    problem_name,
    run,
    X_train,
    X_test,
    y_train,
    y_test,
):
    est = gp.SymbolicRegressor(**params[version](linearscaling, max_evals))

    t_start = time.time()
    est.fit(X_train, y_train)
    t_end = time.time()

    elapsed_seconds = t_end - t_start

    expr = est.model
    try:
        r2_train = r2_score(y_train, est.predict(X_train))
    except Exception:
        r2_train = np.nan

    try:
        r2_test = r2_score(y_test, est.predict(X_test))
    except Exception:
        r2_test = np.nan

    return (
        version,
        max_evals,
        elapsed_seconds,
        problem_name,
        run,
        linearscaling,
        expr,
        r2_train,
        r2_test,
    )


def run(
    max_workers=None,
):
    os.makedirs(RESULT_DIR, exist_ok=True)
    for version in params.keys():
        print(version)
        ofile = f"{RESULT_DIR}/{version}.csv"
        with open(ofile, "w+") as f:
            f.write(
                "alg_name,max_evals,time_seconds,problem_name,run,linear_scaling,sympy_expr,r2_train,r2_test\n"
            )
        if max_workers == 1:
            runs = [(version, *r) for r in all_runs()]
            for i, run in enumerate(runs):
                (
                    alg,
                    max_evals,
                    elapsed_seconds,
                    problem,
                    run,
                    linearscaling,
                    sympy_expr,
                    r2_train,
                    r2_test,
                ) = run_one(*run)
                completed, total = i + 1, len(runs)
                print(f"{completed / total * 100: .2f}% done ({completed} of {total})")
                with open(ofile, "a+") as f:
                    cols = f'"{alg}",{max_evals},{elapsed_seconds},"{problem}",{run},{linearscaling},"{sympy_expr}",{r2_train},{r2_test}'
                    f.write(cols + "\n")
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                scheduled = [pool.submit(run_one, version, *run) for run in all_runs()]
                total, completed = len(scheduled), 0
                for future in as_completed(scheduled):
                    e = future.exception()
                    if e is not None:
                        print(e)
                    else:
                        completed += 1
                        print(
                            f"{completed / total * 100: .2f}% done ({completed} of {total})"
                        )

                    with open(ofile, "a+") as f:
                        (
                            alg,
                            max_evals,
                            elapsed_seconds,
                            problem,
                            run,
                            linearscaling,
                            sympy_expr,
                            r2_train,
                            r2_test,
                        ) = future.result()
                        cols = f'"{alg}",{max_evals},{elapsed_seconds},"{problem}",{run},{linearscaling},"{sympy_expr}",{r2_train},{r2_test}'
                        f.write(cols + "\n")


if __name__ == "__main__":
    run()
