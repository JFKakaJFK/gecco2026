from dataclasses import dataclass
from typing import Literal

DatasetType = Literal["pmlb", "sklearn", "synthetic", "uci"]


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
    "sin": DatasetConfig(
        name="sin(3.1 * x0 + 2.7)",
        dataset_type="synthetic",
        observations=1_000_000,
        features=1,
        target=1e-6,
    ),
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
    "201": DatasetConfig(
        name="201_pol",
        dataset_type="pmlb",
        observations=15_000,
        features=48,
    ),
    "1191": DatasetConfig(
        name="1191_BNG_pbc",
        dataset_type="pmlb",
        observations=1_000_000,
        features=18,
    ),
    ############################################################
    ### Feynman Equations (https://arxiv.org/pdf/1905.11481) ###
    ############################################################
    "feynman_I_9_18": FeynmanDatasetConfig(name="feynman_I_9_18", features=9),
    ############################
    ### Arithmetic Equations ###
    ############################
    # A = x1y1 + x2y2 + x3y3
    "feynman_I_11_19": FeynmanDatasetConfig(name="feynman_I_11_19", features=6),
    # F = mu * N_n
    "feynman_I_12_1": FeynmanDatasetConfig(name="feynman_I_12_1", features=2),
    # U = G * m1 * m2 * (1 / r2 - 1 / r1)
    "feynman_I_13_12": FeynmanDatasetConfig(name="feynman_I_13_12", features=5),
    # r = (m1r1 + m2r2) / (m1 + m2)
    "feynman_I_18_4": FeynmanDatasetConfig(name="feynman_I_18_4", features=4),
    ###############################
    ### Trigonometric Equations ###
    ###############################
    # F = q * (Ef + B * v * sin(theta))
    "feynman_I_12_11": FeynmanDatasetConfig(name="feynman_I_12_11", features=5),
    # E = -mu_M * B * cos(theta)
    "feynman_II_15_4": FeynmanDatasetConfig(name="feynman_II_15_4", features=3),
    # E = 2 * U * (1 - cos(k * d))
    "feynman_III_15_12": FeynmanDatasetConfig(name="feynman_III_15_12", features=3),
    # f = beta * (1 + alpha * cos(theta))
    "feynman_III_17_37": FeynmanDatasetConfig(name="feynman_III_17_37", features=3),
    #############################
    ### Square/Sqrt Equations ###
    #############################
    # sqrt((x2-x1)^2+(y2-y1)^2)
    "feynman_I_8_14": FeynmanDatasetConfig(name="feynman_I_8_14", features=4),
    # k = sqrt(omega**2 / c**2 - pi**2 / d**2)
    "feynman_II_24_17": FeynmanDatasetConfig(name="feynman_II_24_17", features=3),
    #############################
    ### Exponential Equations ###
    #############################
    # E = n * k_b * T * ln(V2/V1)
    "feynman_I_44_4": FeynmanDatasetConfig(name="feynman_I_44_4", features=5),
    ##############################################################################
    ### UCI Machine Learning Repository (https://archive.ics.uci.edu/datasets) ###
    ##############################################################################
    "daily_demand": DatasetConfig(
        name="409",
        dataset_type="uci",
        observations=60,
        features=12,
    ),
    "auto_mpg": DatasetConfig(
        name="9",
        dataset_type="uci",
        observations=398,
        features=7,
    ),
    ################################################################################
    ### SciKit-Learn (https://scikit-learn.org/stable/api/sklearn.datasets.html) ###
    ################################################################################
    "california": DatasetConfig(
        name="california_housing",
        dataset_type="sklearn",
        observations=20640,
        features=8,
    ),
}
