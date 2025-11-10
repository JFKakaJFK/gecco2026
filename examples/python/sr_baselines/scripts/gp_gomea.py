import pathlib
import tempfile
import re
import time
from concurrent.futures import as_completed, ProcessPoolExecutor

import sympy as sym
from pmlb import fetch_data
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

MAX_EVALUATIONS = [500_000]
NUM_RUNS = 30
MAX_NUM_ROWS = 5000
PLMB_CACHE_DIR = "../pmlb/datasets"

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
    for ds in DATASETS:
        X, y = fetch_data(ds, return_X_y=True, local_cache_dir=PLMB_CACHE_DIR)
        X, y = X[:MAX_NUM_ROWS], y[:MAX_NUM_ROWS]

        assert X.shape[0] <= MAX_NUM_ROWS
        assert y.shape[0] <= MAX_NUM_ROWS

        for run in range(NUM_RUNS):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=run + 1
            )

            for max_evals in MAX_EVALUATIONS:
                for ls in [False, True]:
                    yield max_evals, ls, ds, run, X_train, X_test, y_train, y_test


def lambdify_expression(e):
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


ALG_NAME = "GP-GOMEA"
from pyGPGOMEA import run_gp_gomea


def run_one(
    max_evals, linearscaling, problem_name, run, X_train, X_test, y_train, y_test
):
    with tempfile.TemporaryDirectory() as odir:
        # odir = f"pmlb_results/{problem_name}/{run:03d}"
        t_start = time.time()
        run_gp_gomea(
            dict(
                time=-1,
                # value_to_reach=1e-6,
                generations=-1,
                evaluations=max_evals,
                seed=-1,
                prob="symbreg",
                functions="+_-_*_/_sin",
                erc=True,
                gomea=True,
                gomfos="LT",
                gomeanorestarts=True,
                inittype="HH",
                initmaxtreeheight=4,
                syntuniqinit=0,
                popsize=1024,
                linearscaling=linearscaling,
                results_path=odir,
                parallel=1,  # no parallelism
                silent=True,
            ),
            X_train,
            y_train,
            X_test,
            y_test,
        )
        t_end = time.time()

        elapsed_seconds = t_end - t_start

        opath = pathlib.Path(odir)
        assert (opath / "result.txt").is_file()

        df = pd.read_csv(opath / "stats.csv")

        expr = df["structure"].iloc[-1]
        if linearscaling:
            y_pred = lambdify_expression(expr)(X_train)
            X_ls = np.ones((X_train.shape[0], 2))
            X_ls[:, 0] = y_pred
            w, b = np.linalg.lstsq(X_ls, y_train, rcond=None)[0]
            expr = f"{b} + ({w} * ({expr}))"

    fn = lambdify_expression(expr)
    try:
        r2_train = r2_score(y_train, fn(X_train))
        r2_test = r2_score(y_test, fn(X_test))
    except Exception:
        r2_train = np.nan
        r2_test = np.nan

    return (
        ALG_NAME,
        max_evals,
        elapsed_seconds,
        problem_name,
        run,
        linearscaling,
        expr,
        r2_train,
        r2_test,
    )


def run(max_workers=None):
    ofile = f"{ALG_NAME}.csv"
    with open(ofile, "w+") as f:
        f.write(
            "alg_name,max_evals,time_seconds,problem_name,run,linear_scaling,sympy_expr,r2_train,r2_test\n"
        )
    if max_workers == 1:
        for run in all_runs():
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
            with open(ofile, "a+") as f:
                cols = f'"{alg}",{max_evals},{elapsed_seconds},"{problem}",{run},{linearscaling},"{sympy_expr}",{r2_train},{r2_test}'
                f.write(cols + "\n")
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            scheduled = [pool.submit(run_one, *run) for run in all_runs()]
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
