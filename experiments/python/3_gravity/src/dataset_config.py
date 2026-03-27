from dataclasses import dataclass
from typing import Literal

DatasetType = Literal["pmlb", "synthetic"]


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    dataset_type: DatasetType
    observations: int
    features: int
    equation: str | None = None
    target: float | list[float] | None = None

    def target_objectives(self) -> list[float] | None:
        if isinstance(self.target, float):
            return [self.target]

        return self.target


@dataclass(frozen=True, init=False)
class FeynmanDatasetConfig(DatasetConfig):
    def __init__(self, name: str, features: int):
        super().__init__(
            name=name,
            dataset_type="pmlb",
            observations=100_000,
            features=features,
            target=1e-6,
        )


DATASETS: dict[str, DatasetConfig] = {
    "1_addition": DatasetConfig(
        name="1_addition",
        dataset_type="synthetic",
        observations=1_000_000,
        features=9,
        equation="x0+x1+x2+x3+x4+x5+x6+x7+x8",
    ),
    "2_division": DatasetConfig(
        name="2_division",
        dataset_type="synthetic",
        observations=1_000_000,
        features=9,
        equation="(x0+x1+x2)/(x3+x4+x5+x6+x7+x8)",
    ),
    "3_subtraction": DatasetConfig(
        name="3_subtraction",
        dataset_type="synthetic",
        observations=1_000_000,
        features=9,
        equation="(x0+x1+x2)/(x3-x4+x5-x6+x7-x8)",
    ),
    "4_multiplication": DatasetConfig(
        name="4_multiplication",
        dataset_type="synthetic",
        observations=1_000_000,
        features=9,
        equation="(x0*x1*x2)/(x3-x4+x5-x6+x7-x8)",
    ),
    "5_square": DatasetConfig(
        name="5_square",
        dataset_type="synthetic",
        observations=1_000_000,
        features=9,
        equation="(x0*x1*x2)/((x3-x4)**2+(x5-x6)**2+(x7-x8)**2)",
    ),
    "feynman_million": DatasetConfig(
        name="feynman_million",
        dataset_type="synthetic",
        observations=1_000_000,
        features=9,
        equation="(x0*x1*x2)/((x4-x3)**2+(x6-x5)**2+(x8-x7)**2)",
    ),
    ############################################################
    ### Feynman Equations (https://arxiv.org/pdf/1905.11481) ###
    ############################################################
    "feynman_I_9_18": FeynmanDatasetConfig(name="feynman_I_9_18", features=9),
}
