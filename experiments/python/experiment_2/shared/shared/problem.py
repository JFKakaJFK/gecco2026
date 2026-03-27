from collections.abc import Generator
from typing import TypedDict

<<<<<<<< HEAD:experiments/python/3_gravity/src/problem.py
from src.dataset_config import DatasetConfig
from src.experiment_config import ExperimentConfig, TemplateConfig
========
from shared.dataset_config import DatasetConfig
from shared.experiment_config import ExperimentConfig, TemplateConfig
>>>>>>>> dev:experiments/python/experiment_2/shared/shared/problem.py


class Problem(TypedDict):
    dataset: DatasetConfig
    population_size: int | None
    num_observations: int
    num_features: int
    template: TemplateConfig
    operator_set: str
    max_duration: int | None


def format_problem(problem: Problem) -> str:
    pop = (
        str(problem["population_size"])
        if problem["population_size"] is not None
        else "auto"
    )

    return (
        "Problem(\n"
        f"  dataset           = {problem['dataset'].name}\n"
        f"  population_size   = {pop}\n"
        f"  num_observations  = {problem['num_observations']}\n"
        f"  num_features      = {problem['num_features']}\n"
        f"  template          = {problem['template']}\n"
        f"  operator_set      = {problem['operator_set']}\n"
        ")"
    )


ProblemGenerator = Generator[Problem]


def generate_problems(config: ExperimentConfig) -> ProblemGenerator:
    population_sizes = (
        [None] if config.population_sizes is None else config.population_sizes
    )

    num_features = [None] if config.num_features is None else config.num_features

    for data in config.datasets:
        for pop in population_sizes:
            for obs in config.determine_observations(data.observations):
                for feat in num_features:
                    # If unspecified, use the number of features available in the dataset
                    if feat is None:
                        feat = data.features

                    for tmpl in config.templates:
                        for op_set in config.operator_sets:
                            problem: Problem = {
                                "dataset": data,
                                "population_size": pop,
                                "num_observations": obs,
                                "num_features": feat,
                                "template": tmpl,
                                "operator_set": op_set,
                                "max_duration": config.max_duration,
                            }

                            yield problem
