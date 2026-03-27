import pathlib
import re

import duckdb
import numpy as np
import sympy as sym
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer


def lambdify_expression(e):
    """Converts a sympy-compatible expression string into a function accepting a dataset X."""
    e = str(e)
    symbols = {x: sym.Symbol(x) for x in re.findall(r"(x\d+)", e)}
    expr = sym.sympify(e, locals=symbols)
    f = sym.lambdify(symbols.values(), expr, modules=[{"clip": np.clip}, "numpy"])

    def fn(X: np.ndarray):
        try:
            return f(*[X[:, int(s[1:])] for s in symbols.keys()])
        except Exception as err:
            print(err)
            return np.repeat(float("nan"), X.shape[0])

    return fn


def evaluate_gpu_results(
    db_path: pathlib.Path,
    data_dir: pathlib.Path,
    out_path: pathlib.Path,
) -> None:
    src = duckdb.connect(str(db_path), read_only=True)

    rows = src.execute("""
        SELECT fold, run, evaluations, template_depth, population_size, seed, old_mse, expressions
        FROM results
        ORDER BY fold, run, evaluations
    """).fetchall()
    src.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = duckdb.connect(str(out_path))
    out.execute("""
        CREATE TABLE IF NOT EXISTS results (
            fold            INTEGER,
            run             INTEGER,
            evaluations     UBIGINT,
            template_depth  INTEGER,
            population_size INTEGER,
            old_mse         DOUBLE,
            eval_mse        DOUBLE,
            eval_r2         DOUBLE,
            expression      TEXT,
        )
    """)

    imputed_cache: dict[tuple, np.ndarray] = {}

    for fold, run, evaluations, template_depth, population_size, seed, old_mse, expr in rows:
        cache_key = (fold, seed)
        if cache_key not in imputed_cache:
            X_raw = np.load(data_dir / f"X{fold:03d}.npy")
            if np.isnan(X_raw).any():
                imputer = IterativeImputer(max_iter=10, random_state=seed, sample_posterior=True)
                imputed_cache[cache_key] = imputer.fit_transform(X_raw)
            else:
                imputed_cache[cache_key] = X_raw

        X = imputed_cache[cache_key]
        y = np.load(data_dir / f"y{fold:03d}.npy").ravel()

        try:
            y_pred = lambdify_expression(expr)(X)
            eval_mse = float(np.mean((y - y_pred) ** 2))
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            eval_r2 = float(1.0 - ss_res / ss_tot)
        except Exception as e:
            print(f"fold {fold}: ERROR: {e}")
            continue

        out.execute(
            "INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [fold, run, evaluations, template_depth, population_size, old_mse, eval_mse, eval_r2, expr],  # noqa: E501
        )

    out.close()
    print(f"Saved to {out_path}")


def fill_mse_gpu_results(
    db_path: pathlib.Path,
    data_dir: pathlib.Path,
) -> None:
    conn = duckdb.connect(str(db_path))

    rows = conn.execute("""
        SELECT rowid, fold, seed, expressions
        FROM results
        WHERE mse IS NULL OR isnan(mse::DOUBLE)
        ORDER BY fold, rowid
    """).fetchall()

    print(f"Filling {len(rows)} NaN mse values...")

    imputed_cache: dict[tuple, np.ndarray] = {}

    for rowid, fold, seed, expr in rows:
        cache_key = (fold, seed)
        if cache_key not in imputed_cache:
            X_raw = np.load(data_dir / f"X{fold:03d}.npy")
            if np.isnan(X_raw).any():
                imputer = IterativeImputer(max_iter=10, random_state=seed, sample_posterior=True)
                imputed_cache[cache_key] = imputer.fit_transform(X_raw)
            else:
                imputed_cache[cache_key] = X_raw

        X = imputed_cache[cache_key]
        y = np.load(data_dir / f"y{fold:03d}.npy").ravel()

        y_pred = lambdify_expression(expr)(X)
        mse = float(np.mean((y - y_pred) ** 2))

        conn.execute("UPDATE results SET mse = ? WHERE rowid = ?", [mse, rowid])

    conn.close()
    print("Done.")


if __name__ == "__main__":
    evaluate_gpu_results(
        db_path=pathlib.Path("results/experiment_2/gpu_results/auto_mpg.duckdb"),
        data_dir=pathlib.Path("results/auto_mpg_data/data"),
        out_path=pathlib.Path("results/temp/auto_mpg.duckdb"),
    )
