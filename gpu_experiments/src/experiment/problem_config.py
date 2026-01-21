from dataclasses import dataclass
from typing import Literal

ProblemType = Literal["synthetic", "pmlb"]


@dataclass(frozen=True)
class ProblemConfig:
    name: str
    problem_type: ProblemType
    observations: int
    features: int
    target: float | list[float] | None = None

    def target_objectives(self) -> list[float] | None:
        if isinstance(self.target, float):
            return [self.target]

        return self.target


@dataclass(frozen=True, init=False)
class FeynmanProblemConfig(ProblemConfig):
    def __init__(self, name: str, features: int):
        super().__init__(
            name=name,
            problem_type="pmlb",
            observations=100_000,
            features=features,
            target=1e-6,
        )


PROBLEMS: dict[str, ProblemConfig] = {
    "sin": ProblemConfig(
        name="sin(3.1 * x0 + 2.7)",
        problem_type="synthetic",
        observations=1_000_000,
        features=1,
        target=1e-6,
    ),
    "201": ProblemConfig(
        name="201_pol",
        problem_type="pmlb",
        observations=15_000,
        features=48,
    ),
    "1191": ProblemConfig(
        name="1191_BNG_pbc",
        problem_type="pmlb",
        observations=1_000_000,
        features=18,
    ),
    ############################################################
    ### Feynman Equations (https://arxiv.org/pdf/1905.11481) ###
    ############################################################
    ### Arithmetic Equations ###
    # A = x1y1 + x2y2 + x3y3
    "feynman_I_11_19": FeynmanProblemConfig(
        name="feynman_I_11_19",
        features=6,
    ),
    # F = mu * N_n
    "feynman_I_12_1": FeynmanProblemConfig(
        name="feynman_I_12_1",
        features=2,
    ),
    # r = (m1r1 + m2r2) / (m1 + m2)
    "feynman_I_18_4": FeynmanProblemConfig(
        name="feynman_I_18_4",
        features=4,
    ),
    # P = (kappa * (T2 - T1) * A) / d
    "feynman_II_2_42": FeynmanProblemConfig(
        name="feynman_II_2_42",
        features=5,
    ),
    ### Trigonometric Equations ###
    # tau = r * F * sin(theta)
    "feynman_I_18_12": FeynmanProblemConfig(
        name="feynman_I_18_12",
        features=3,
    ),
    # E = -mu_M * B * cos(theta)
    "feynman_II_15_4": FeynmanProblemConfig(
        name="feynman_II_15_4",
        features=3,
    ),
    # E = 2 * U * (1 - cos(k * d))
    "feynman_III_15_12": FeynmanProblemConfig(
        name="feynman_III_15_12",
        features=3,
    ),
    # f = beta * (1 + alpha * cos(theta))
    "feynman_III_17_37": FeynmanProblemConfig(
        name="feynman_III_17_37",
        features=3,
    ),
    ### Square/Sqrt Equations ###
    # sqrt((x2-x1)^2+(y2-y1)^2)
    "feynman_I_8_14": FeynmanProblemConfig(
        name="feynman_I_8_14",
        features=4,
    ),
    # x = x1 (cos(omega * t) + alpha * cos(omega * t)^2)
    "feynman_I_50_26": FeynmanProblemConfig(
        name="feynman_I_50_26",
        features=4,
    ),
    # V_e = (1 / (4 * pi * epsilon)) * ((p_d * cos(theta)) / r^2)
    "feynman_II_6_11": FeynmanProblemConfig(
        name="feynman_I_8_14",
        features=4,
    ),
    "feynman_III_10_19": FeynmanProblemConfig(
        name="feynman_III_10_19",
        features=4,
    ),
    ### Exponential Equations ###
    "feynman_I_44_4": FeynmanProblemConfig(
        name="feynman_I_44_4",
        features=5,
    ),
}
