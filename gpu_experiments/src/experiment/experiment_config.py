from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal, TypeVar

from pygom import KernelVersion as KV

from src.experiment.problem_config import PROBLEMS

T = TypeVar("T")

ProblemType = Literal["synthetic", "pmlb"]

KERNEL_VERSIONS: tuple[KV, ...] = (
    KV.baseline,
    KV.restrict,
    KV.shared_memory,
    KV.block_reduce,
    KV.single_kernel,
    KV.single_kernel_fmaf,
    KV.single_kernel_inplace,
)


@dataclass(frozen=True)
class ProblemConfig:
    name: str
    type: ProblemType
    observations: int
    features: int
    target: float | list[float] | None = None

    def target_objectives(self) -> list[float] | None:
        if isinstance(self.target, float):
            return [self.target]

        return self.target


@dataclass(frozen=True)
class TemplateConfig:
    branching_factor: int = 2
    depth: int = 4


@dataclass(frozen=True)
class OperatorSet:
    name: str
    operators: str


@dataclass(frozen=True)
class GPUConfig:
    enabled: bool = True
    kernels: tuple[KV, ...] = (KV.single_kernel,)


@dataclass(frozen=True)
class CPUConfig:
    enabled: bool = True


OPERATOR_SETS: dict[str, str] = {
    "arith": "+,-,*,/",
    "trig": "+,-,*,/,sin,cos",
    "square": "+,-,*,/,sin,cos,square,sqrt",
    "exp": "+,-,*,/,sin,cos,square,sqrt,exp,log",
    "all": "+,-,*,/,sin,cos,square,sqrt,exp,log,pow,abs,min,max",
}


def ensure_iter(x: T | list[T]) -> list[T]:
    if isinstance(x, list):
        return x
    else:
        return [x]


@dataclass
class ExperimentConfig:
    name: str

    # Problem space
    problems: list[ProblemConfig] | ProblemConfig
    population_sizes: list[int] | int
    num_observations: list[int] | int
    num_features: list[int] | int | None
    templates: list[TemplateConfig] | TemplateConfig = field(
        default_factory=TemplateConfig
    )
    operator_sets: list[str] | str = "all"

    use_target: bool = False

    num_folds: int = 5
    num_iterations: int = 3
    test_size: float = 0.25

    # Execution
    cpu: CPUConfig = field(default_factory=CPUConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)

    def determine_observations(self, observations):
        return [
            int(obs * (1 - self.test_size) * (1 - 1 / self.num_folds))
            for obs in observations
        ]

    def iter_problems(self) -> Iterable[ProblemConfig]:
        return ensure_iter(self.problems)

    def iter_population(self) -> Iterable[int]:
        return ensure_iter(self.population_sizes)

    def iter_observations(self) -> Iterable[int]:
        return self.determine_observations(ensure_iter(self.num_observations))

    def iter_features(self) -> Iterable[int]:
        return ensure_iter(self.num_features)

    def iter_templates(self) -> Iterable[TemplateConfig]:
        return ensure_iter(self.templates)

    def iter_operators(self) -> Iterable[str]:
        return ensure_iter(self.operator_sets)

    def get_target_objectives(self, problem: ProblemConfig) -> list[float] | None:
        if problem not in self.iter_problems():
            raise ValueError("Problem not found in config")

        return ensure_iter(problem.target_objectives())


TEST_CONFIG = ExperimentConfig(
    name="test",
    problems=PROBLEMS["sin"],
    population_sizes=[1024, 2048],
    num_observations=[10_000, 100_000],
    num_features=1,
    # templates=[TemplateConfig(2, 3), TemplateConfig(2, 4)],
    # operator_sets=["small", "all"],
    gpu=GPUConfig(enabled=True, kernels=(KERNEL_VERSIONS)),
    use_target=True,
)

ARITH_FEYNMAN_CONFIG = ExperimentConfig(
    name="arith_feynman",
    problems=[
        PROBLEMS["feynman_I_11_19"],
        PROBLEMS["feynman_I_12_1"],
        PROBLEMS["feynman_I_18_4"],
        PROBLEMS["feynman_II_2_42"],
    ],
    population_sizes=2048,
    num_observations=100_000,
    num_features=None,
    operator_sets="arith",
    use_target=True,
)

TRIG_FEYNMAN_CONFIG = ExperimentConfig(
    name="trig_feynman",
    problems=[
        PROBLEMS["feynman_I_18_12"],
        PROBLEMS["feynman_II_15_4"],
        PROBLEMS["feynman_III_15_12"],
        PROBLEMS["feynman_III_17_37"],
    ],
    population_sizes=2048,
    num_observations=100_000,
    num_features=None,
    operator_sets="trig",
    gpu=GPUConfig(enabled=True, kernels=(KERNEL_VERSIONS)),
    use_target=True,
)

SQRT_FEYNMAN_CONFIG = ExperimentConfig(
    name="sqrt_feynman",
    problems=[
        PROBLEMS["feynman_I_8_14"],
        PROBLEMS["feynman_I_50_26"],
        PROBLEMS["feynman_II_6_11"],
        PROBLEMS["feynman_III_10_19"],
    ],
    population_sizes=2048,
    num_observations=100_000,
    num_features=None,
    operator_sets="square",
    gpu=GPUConfig(enabled=True, kernels=(KERNEL_VERSIONS)),
    use_target=True,
)

EXP_FEYNMAN_CONFIG = ExperimentConfig(
    name="exp_feynman",
    problems=PROBLEMS["feynman_I_44_4"],
    population_sizes=2048,
    num_observations=100_000,
    num_features=None,
    operator_sets="exp",
    gpu=GPUConfig(enabled=True, kernels=(KERNEL_VERSIONS)),
    use_target=True,
)


class Configs:
    # Test experiment
    TEST = TEST_CONFIG

    # Feynman experiments
    SIMPLE_FEYNMAN = ARITH_FEYNMAN_CONFIG
    TRIG_FEYNMAN = TRIG_FEYNMAN_CONFIG
    SQRT_FEYNMAN = SQRT_FEYNMAN_CONFIG
    EXP_FEYNMAN = EXP_FEYNMAN_CONFIG


cfg = Configs()
