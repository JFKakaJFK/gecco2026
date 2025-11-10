import pathlib
import tempfile
import re
import time
import subprocess
from concurrent.futures import as_completed, ThreadPoolExecutor

import sympy as sym
from pmlb import fetch_data
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

MAX_EVALUATIONS = [100_000, 500_000]
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


import io


def dump_params(config: dict, file: io.TextIOBase):
    """
    Dumps the given config into a GP-GOMEA compatible parameter file.

    Flags have value `True` and the values `False` and `None` are ignored.
    """
    for k, v in config.items():
        if v is None or isinstance(v, bool):
            if v:
                file.write(f"{k}\n")
        else:
            file.write(f"{k} {v}\n")


def save_tsv(path: pathlib.Path, data: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, data, fmt="%+.17g", delimiter=" ", encoding="ascii")


RELEASE_EXE = "build/release/src/main"

ALG_NAME = "GP-RV ERCs"
# ALG_NAME = "GP-RV IA ERCs"
# ALG_NAME = "GP-RV IA"
intron_aware = False
enable_rv = False


def run_one(
    max_evals, linearscaling, problem_name, run, X_train, X_test, y_train, y_test
):
    with tempfile.TemporaryDirectory() as odir:
        # odir = f"pmlb_results/{problem_name}/{run:03d}"
        opath = pathlib.Path(odir)

        train_path = opath / "train.tsv"
        train_data = np.empty((X_train.shape[0], X_train.shape[1] + 1))
        train_data[:, :-1] = X_train
        train_data[:, -1] = y_train
        save_tsv(train_path, train_data)

        test_path = opath / "test.tsv"
        test_data = np.empty((X_test.shape[0], X_test.shape[1] + 1))
        test_data[:, :-1] = X_test
        test_data[:, -1] = y_test
        save_tsv(test_path, test_data)

        params = dict(
            seed=-1,
            gomea=True,
            gomfos="LT",
            nrtrees=1,
            parallel=1,
            gomea_no_restarts=True,
            initmaxtreeheight=4,
            functions="+_-_*_/_sin",
            popsize=1024,
            time=-1,
            generations=-1,
            evaluations=max_evals,
            gomea_forced_improvements=True,
            prob="multiobj",
            multiobj="symbreg",
            linearscaling=linearscaling,
            constant_probability=0.5,
            gomea_fos_no_mi_adjustments=intron_aware,
            gomea_fos_mask_introns=intron_aware,
            gomea_fos_normalize_introns=intron_aware,
            constant_init_lb="inf",
            constant_init_ub="inf",
            constant_init_bound_multiplier=1.0,
            erc=not enable_rv,
            results_path=odir,
            train=str(train_path),
            test=str(test_path),
        )
        params["rvg.num_constants"] = 10 if enable_rv else 0
        params["rvg.reference_unware_gp"] = True
        params["rvg.rv_to_gp_ratio"] = 1
        params["rvg.stochastic_evaluation_balance"] = True
        params["rvg.mo_skip_secondary_extreme_selection"] = True
        params["rvg.maximum_no_improvement_stretch"] = 20
        params["rvg.intron_variation"] = "SampleOnly"
        params["rvg.na_variation"] = "SampleOnly"
        params["rvg.semantic_crossover_probability"] = 0.0
        params["rvg.no_constant_renaming"] = True

        params_file = str(opath / "params.conf")

        with open(params_file, "w+") as f:
            dump_params(params, f)

        t_start = time.time()
        subprocess.run(
            [RELEASE_EXE, "--file", params_file],
            check=True,
            # stdout=stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            # cwd=os.getcwd(),
        )
        t_end = time.time()

        elapsed_seconds = t_end - t_start
        df = pd.read_csv(opath / "mo_archive.csv")

        expr = df["expressions"].iloc[-1][1:-1]
        # if linearscaling:
        #     y_pred = lambdify_expression(expr)(X_train)
        #     X_ls = np.ones((X_train.shape[0], 2))
        #     X_ls[:, 0] = y_pred
        #     w, b = np.linalg.lstsq(X_ls, y_train, rcond=None)[0]
        #     expr = f"{b} + ({w} * ({expr}))"

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
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
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
    run(20)
