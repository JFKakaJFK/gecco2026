import pathlib
import re
from urllib.parse import quote

import numpy as np
import pandas as pd
import sympy as sym
from pmlb import fetch_data
from sklearn.model_selection import KFold, train_test_split


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


def load_local(problem: str, data_dir):
    data_dir = pathlib.Path(data_dir)
    # unfortunately, https://github.com/uci-ml-repo/ucimlrepo doesn't handle all of the following datasets correctly...
    if problem == "Bike Sharing":
        # https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip -> daily.csv as bike_sharing_daily.csv
        # https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
        # ! this dataset has some categorical columns, that is ignored here
        # columns instant, casual and registered need to be removed
        df = pd.read_csv(data_dir / "bike_sharing_daily.csv")
        features = [
            c for c in df.columns if c not in ["instant", "casual", "registered", "cnt"]
        ]
        df["dteday"] = pd.to_datetime(
            df["dteday"]
        ).dt.dayofyear  # map dates to [1, 366]
        return df[features].to_numpy(dtype=np.float64), df["cnt"].to_numpy(
            dtype=np.float64
        )
    elif problem == "Breast Cancer":
        # https://archive.ics.uci.edu/static/public/17/data.csv -> breast_cancer.csv
        # https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
        # ! this dataset is a binary classification task, the 'B'enign / 'M'alignant labels are replaced with 0/1 respectively
        df = pd.read_csv(data_dir / "breast_cancer.csv")
        features = [c for c in df.columns if c not in ["ID", "Diagnosis"]]
        return df[features].to_numpy(dtype=np.float64), (
            df["Diagnosis"] == "M"
        ).to_numpy(dtype=np.float64).astype(np.float64)
    elif problem == "Concrete Compressive Strength":
        # https://archive.ics.uci.edu/static/public/165/data.csv
        # https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
        df = pd.read_csv(data_dir / "concrete.csv")
        features = [c for c in df.columns if c not in ["Concrete compressive strength"]]
        return df[features].to_numpy(dtype=np.float64), df[
            "Concrete compressive strength"
        ].to_numpy(dtype=np.float64)
    elif problem == "Yacht Hydrodynamics":
        # https://archive.ics.uci.edu/dataset/243/yacht+hydrodynamics
        # ! the dataset contains some occurrences of "  " (these need to be replaced with a single space to make delimiter=" " work corrrectly)
        data = np.loadtxt(data_dir / "yacht_hydrodynamics.data", delimiter=" ")
        return data[:, :-1], data[:, -1]
    elif problem == "Airfoil":
        # https://archive.ics.uci.edu/dataset/291/airfoil+self+noise
        # https://archive.ics.uci.edu/static/public/291/data.csv -> airfoil.csv
        df = pd.read_csv(data_dir / "airfoil.csv")
        features = [c for c in df.columns if c not in ["scaled-sound-pressure"]]
        return df[features].to_numpy(dtype=np.float64), df[
            "scaled-sound-pressure"
        ].to_numpy(dtype=np.float64)
    elif problem == "Energy Heating":
        # https://archive.ics.uci.edu/dataset/242/energy+efficiency
        # https://archive.ics.uci.edu/static/public/242/data.csv -> energy.csv
        df = pd.read_csv(data_dir / "energy.csv")
        features = [c for c in df.columns if c not in ["Y1", "Y2"]]
        return df[features].to_numpy(dtype=np.float64), df["Y1"].to_numpy(
            dtype=np.float64
        )
    elif problem == "Energy Cooling":
        # https://archive.ics.uci.edu/dataset/242/energy+efficiency
        # https://archive.ics.uci.edu/static/public/242/data.csv -> energy.csv
        df = pd.read_csv(data_dir / "energy.csv")
        features = [c for c in df.columns if c not in ["Y1", "Y2"]]
        return df[features].to_numpy(dtype=np.float64), df["Y2"].to_numpy(
            dtype=np.float64
        )
    elif problem == "Dow Chemical":
        # Dow Chemical from the 2010 Evo* competition
        # https://web.archive.org/web/20120628140646/http%3A//casnew.iti.upv.es/index.php/evocompetitions/105-symregcompetition (download link should still work)
        # Both the train and test set from the original competition are combined here
        df = pd.concat(
            [
                pd.read_csv(data_dir / "dow_chemical.csv"),
                pd.read_csv(data_dir / "dow_chemical_test.csv"),
            ],
            ignore_index=True,
        )
        features = [c for c in df.columns if c not in ["y"]]
        return df[features].to_numpy(dtype=np.float64), df["y"].to_numpy(
            dtype=np.float64
        )
    elif problem == "Tower":
        # Tower dataset referenced in https://ieeexplore.ieee.org/document/4632147
        # Obtained via https://drive.google.com/drive/folders/1cUU7f23z_lBPQCOX7h1unZWKZH3YO_EB (referenced in http://arxiv.org/abs/1904.02050)
        Xy = pd.read_csv(data_dir / "tower.csv").to_numpy(dtype=np.float64)
        return Xy[:, :-1], Xy[:, -1]

    raise ValueError(f"Unknown Local dataset: '{problem}'")


def load_problem(
    problem: str, data_dir="datasets", pmlb_cache_dir=".pmlb", **kwargs
) -> tuple[tuple[np.ndarray, np.ndarray], bool]:
    try:
        return load_local(problem, data_dir), False
    except Exception as e:
        # print(problem, e)
        pass
    try:
        return fetch_data(
            problem, return_X_y=True, local_cache_dir=pmlb_cache_dir
        ), False
    except Exception as _:
        return synthetic_problem(problem, **kwargs), True


def prepare_problem(
    rng,
    problem,
    num_folds,
    data_dir,
    synthetic_problem_kwargs=dict(
        size=500,
        noise=0.0,
    ),
):
    # 1. get the data
    (X, y), is_synthetic = load_problem(
        problem,
        # for synthetic problems only
        seed=rng.integers(2**32),
        **synthetic_problem_kwargs,
    )

    # 2. train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=rng.integers(2**32)
    )

    pdir = data_dir / quote(problem, safe=' (){}$_+-"')
    pdir.mkdir(parents=True, exist_ok=True)

    X_test_path, y_test_path = (
        pdir / "X_test.npy",
        pdir / "y_test.npy",
    )
    np.save(X_test_path, X_test, allow_pickle=False)
    np.save(y_test_path, y_test.reshape(-1, 1), allow_pickle=False)

    # 3. split the training portion into folds
    # the rationale here is that we want comparable test performance across folds - so the training set is split again and the validation portion is simply not used (for speed, train/test doesn't really matter though...)
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=rng.integers(2**32 - 1))
    for fold, (train_indices, _validation_indices) in enumerate(kf.split(X_train)):
        X_fold, y_fold = (
            X_train[train_indices, :],
            y_train[train_indices].reshape(-1, 1),
        )

        X_path, y_path = (
            pdir / f"X{fold:03d}.npy",
            pdir / f"y{fold:03d}.npy",
        )
        np.save(X_path, X_fold, allow_pickle=False)
        np.save(y_path, y_fold, allow_pickle=False)

        yield (
            fold,
            X_path,
            y_path,
            X_test_path,
            y_test_path,
            X_fold,
            y_fold,
            is_synthetic,
        )


def problem_info(name: str, X: np.ndarray, y: np.ndarray):
    y_mean = float(np.mean(y))
    y_var = float(np.var(y))
    # LS
    X_ls = np.ones((X.shape[0], X.shape[1] + 1))
    X_ls[:, :-1] = X
    b = np.linalg.lstsq(X_ls, y)[0]
    y_ls = X_ls @ b
    r2_ls = 1 - np.mean((y_ls - y) ** 2) / y_var

    print(
        f"{name}:\n - #rows: {X.shape[0]}\n - #features: {X.shape[1]}\n - mean_y: {y_mean}\n - var_y: {y_var}\n - Linear regression R2: {r2_ls}"
    )
