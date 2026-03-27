import os
import re
from collections.abc import Callable, Generator
from pathlib import Path
from typing import TypedDict

import numpy as np
import pmlb
import sympy as sym
from pygom import KernelVersion
from sklearn.model_selection import KFold, train_test_split

from src.dataset_config import DatasetConfig
from src.experiment_config import ExperimentConfig
from src.problem import Problem

PMLB_CACHE_DIRECTORY = "pmlb_cache"


class Task(TypedDict):
    dataset: str
    accelerated: bool
    kernel: KernelVersion | None
    population_size: int | None
    num_observations: int
    num_features: int
    branching_factor: int
    depth: int
    operator_set: str
    target_objectives: list[float] | None
    fold: int
    iteration: int
    X_path: Path
    y_path: Path
    X_test_path: Path | None
    y_test_path: Path | None
    seed: int
    max_duration: int


TaskGenerator = Generator[Task]
TaskFactory = Callable[[], TaskGenerator]
TaskTransform = Callable[[TaskGenerator, dict[str, int]], TaskGenerator]


def lambdify_expression(e: str | sym.Expr):
    """
    Converts a `sympy` compatible expression string into a function
    accepting a dataset `X`.
    """
    e = str(e)

    symbols = {x: sym.Symbol(x) for x in re.findall(r"(x\d+)", e)}
    expr = sym.sympify(e, locals=symbols)
    f = sym.lambdify(symbols.values(), expr, modules=[{"clip": np.clip}, "numpy"])

    def fn(X: np.ndarray):
        try:
            return f(*[X[:, int(s[1:])] for s in symbols])
        except Exception as e:
            print(e)
            return np.repeat(float("nan"), X.shape[0])

    return fn


def synthetic_problem(
    expr: str,
    size: int = 1000,
    lb: float = 1.0,
    ub: float = 5.0,
    noise: float = 0.01,
    seed: int | None = None,
):
    """
    Creates a synthetic problem by sampling a random dataset,
    applying the function and possibly adding noise.
    """
    assert ub > lb, "Invalid initialization bounds"

    rng = np.random.Generator(np.random.Philox(seed=seed))

    num_inputs = max([int(x) + 1 for x in re.findall(r"x(\d+)", expr)])

    X = rng.random(size=(size, num_inputs)) * (ub - lb) + lb
    y = lambdify_expression(expr)(X)

    if noise > 0:
        y += rng.standard_normal(size) * noise

    return X, y


def create_tasks(
    problem: Problem,
    config: ExperimentConfig,
    rng: np.random.Generator,
    output_directory: Path,
    dry_run: bool = False,
) -> TaskGenerator:
    data_directory = output_directory / problem["dataset"].name / "data"

    if not dry_run:
        os.makedirs(data_directory, exist_ok=True)

    # Retrieve dataset data
    dataset: DatasetConfig = problem["dataset"]

    match dataset.dataset_type:
        case "pmlb":
            X, y = pmlb.fetch_data(
                dataset.name,
                return_X_y=True,
                local_cache_dir=PMLB_CACHE_DIRECTORY,
            )
        case "synthetic":
            assert dataset.equation is not None
            X, y = synthetic_problem(
                dataset.equation,
                dataset.observations,
                noise=0.0,
                seed=42,
            )

    if config.test_size != 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=rng.integers(2**32 - 1)
        )
    else:
        X_train = X
        X_test = None
        y_train = y
        y_test = None

    if X_test is not None and y_test is not None:
        X_test_path, y_test_path = (
            data_directory / "X_test.npy",
            data_directory / "y_test.npy",
        )

        if not dry_run:
            np.save(X_test_path, X_test, allow_pickle=False)
            np.save(y_test_path, y_test.reshape(-1, 1), allow_pickle=False)
    else:
        X_test_path = None
        y_test_path = None

    target_objectives = (
        config.get_target_objectives(problem["dataset"]) if config.use_target else None
    )

    kf = KFold(
        n_splits=config.num_folds,
        shuffle=True,
        random_state=rng.integers(2**32 - 1),
    )
    for fold, (train_indices, _) in enumerate(kf.split(X_train)):
        X_fold, y_fold = (
            X_train[train_indices, :],
            y_train[train_indices].reshape(-1, 1),
        )

        X_path, y_path = (
            data_directory / f"X{fold:03d}.npy",
            data_directory / f"y{fold:03d}.npy",
        )

        if not dry_run:
            np.save(X_path, X_fold, allow_pickle=False)
            np.save(y_path, y_fold, allow_pickle=False)

        seed = int(rng.integers(2**32 - 1))

        for iteration in range(config.num_iterations):
            task: Task = {
                **problem,
                "dataset": dataset.name,
                "accelerated": False,
                "kernel": None,
                "branching_factor": problem["template"].branching_factor,
                "depth": problem["template"].depth,
                "target_objectives": target_objectives,
                "fold": fold,
                "iteration": iteration,
                "X_path": X_path,
                "y_path": y_path,
                "X_test_path": X_test_path,
                "y_test_path": y_test_path,
                "seed": seed,
                "max_duration": problem["max_duration"],
            }

            yield task


def task_factory(
    problem: Problem,
    config: ExperimentConfig,
    output_directory: Path,
    seed: int = 42,
    dry_run: bool = False,
) -> TaskFactory:
    def _factory() -> TaskGenerator:
        rng = np.random.default_rng(seed)

        return create_tasks(problem, config, rng, output_directory, dry_run)

    return _factory


def override_tasks(overrides: dict[str, int]) -> TaskGenerator:
    def _transform(tasks: TaskGenerator) -> TaskGenerator:
        for task in tasks:
            yield {**task, **overrides}

    return _transform


def gpu_transform(kernel: KernelVersion) -> TaskTransform:
    def _transform(tasks: TaskGenerator) -> TaskGenerator:
        override = override_tasks({"accelerated": True, "kernel": kernel})

        yield from override(tasks)

    return _transform


def cpu_transform() -> TaskTransform:
    def _transform(tasks: TaskGenerator) -> TaskGenerator:
        override = override_tasks({"accelerated": False})

        yield from override(tasks)

    return _transform
