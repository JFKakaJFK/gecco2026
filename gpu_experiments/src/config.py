from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal, TypeVar

from pygom import KernelVersion as KV

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


DATASETS: dict[str, ProblemConfig] = {
    "sin": ProblemConfig(
        name="sin(3.1 * x0 + 2.7)",
        type="synthetic",
        observations=1_000_000,
        features=1,
    ),
    "201": ProblemConfig(
        name="201_pol",
        type="pmlb",
        observations=15_000,
        features=48,
    ),
    "1191": ProblemConfig(
        name="1191_BNG_pbc",
        type="pmlb",
        observations=1_000_000,
        features=18,
    ),
    "feynman_9": ProblemConfig(
        name="feynman_test_9",
        type="pmlb",
        observations=100_000,
        features=5,
        target=2e7,
    ),
}

OPERATOR_SETS: dict[str, str] = {
    "small": "+,-,*,/",
    "exp": "+,-,*,/,sin,cos,exp,log",
    "all": "+,-,*,/,sin,cos,exp,log,square,sqrt,pow,abs,min,max",
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
    num_features: list[int] | int
    templates: list[TemplateConfig] | TemplateConfig = field(
        default_factory=TemplateConfig
    )
    operator_sets: list[str] | str = "all"

    use_target: bool = False

    num_folds: int = 5
    num_iterations: int = 3

    # Execution
    cpu: CPUConfig = field(default_factory=CPUConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)

    def iter_problems(self) -> Iterable[ProblemConfig]:
        return ensure_iter(self.problems)

    def iter_population(self) -> Iterable[int]:
        return ensure_iter(self.population_sizes)

    def iter_observations(self) -> Iterable[int]:
        return ensure_iter(self.num_observations)

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
    problems=DATASETS["sin"],
    population_sizes=[256, 512],
    num_observations=[10, 100],
    num_features=1,
    templates=[TemplateConfig(2, 3), TemplateConfig(2, 4)],
    operator_sets=["small", "all"],
    gpu=GPUConfig(enabled=True, kernels=(KV.baseline, KV.single_kernel)),
)

# Used for comparing the fastest kernel version to the CPU version
SCALABILITY_POPULATION_CONFIG = ExperimentConfig(
    name="scalability_population",
    problems=DATASETS["feynman_9"],
    population_sizes=[2**i for i in range(8, 16)],  # 256 - 32768
    num_observations=60_000,
    num_features=DATASETS["feynman_9"].features,
    use_target=True,
)

SCALABILITY_OBSERVATION_CONFIG = ExperimentConfig(
    name="scalability_observation",
    problems=DATASETS["feynman_9"],
    population_sizes=512,
    num_observations=[int(10**i * 0.75 * 0.8) for i in range(2, 6)],  # 60 - 60_000
    num_features=DATASETS["feynman_9"].features,
    use_target=True,
)


# Used for comparing all kernel versions to the CPU version
CPU_KERNEL_POPULATION_CONFIG = ExperimentConfig(
    name="cpu_kernel_population",
    problems=DATASETS["feynman_9"],
    population_sizes=[2**i for i in range(8, 16)],  # 256 - 32768
    num_observations=60_000,
    num_features=DATASETS["feynman_9"].features,
    use_target=True,
    gpu=GPUConfig(kernels=KERNEL_VERSIONS),
)

CPU_KERNEL_OBSERVATION_CONFIG = ExperimentConfig(
    name="cpu_kernel_observation",
    problems=DATASETS["feynman_9"],
    population_sizes=512,
    num_observations=[int(10**i * 0.75 * 0.8) for i in range(2, 6)],  # 60 - 60_000
    num_features=DATASETS["feynman_9"].features,
    use_target=True,
    cpu=CPUConfig(enabled=False),
    gpu=GPUConfig(kernels=KERNEL_VERSIONS),
)

# Used for comparing different kernel versions
KERNEL_SWEEP_POPULATION_CONFIG = ExperimentConfig(
    name="kernel_sweep_population",
    problems=DATASETS["feynman_9"],
    population_sizes=[2**i for i in range(8, 16)],  # 256 - 32768
    num_observations=60_000,
    num_features=DATASETS["feynman_9"].features,
    use_target=True,
    cpu=CPUConfig(enabled=False),
    gpu=GPUConfig(kernels=KERNEL_VERSIONS),
)

KERNEL_SWEEP_OBSERVATION_CONFIG = ExperimentConfig(
    name="kernel_sweep_observation",
    problems=DATASETS["feynman_9"],
    population_sizes=512,
    num_observations=[int(10**i * 0.75 * 0.8) for i in range(2, 6)],  # 60 - 60_000
    num_features=DATASETS["feynman_9"].features,
    use_target=True,
    cpu=CPUConfig(enabled=False),
    gpu=GPUConfig(kernels=KERNEL_VERSIONS),
)


class Configs:
    # Test experiment
    TEST = TEST_CONFIG

    # Scalability experiments
    SCALABILITY_POPULATION = SCALABILITY_POPULATION_CONFIG
    SCALABILITY_OBSERVATION = SCALABILITY_OBSERVATION_CONFIG

    # CPU vs kernels
    CPU_KERNEL_POPULATION = CPU_KERNEL_POPULATION_CONFIG
    CPU_KERNEL_OBSERVATION = CPU_KERNEL_OBSERVATION_CONFIG

    # Kernel sweep experiments
    KERNEL_SWEEP_POPULATION = KERNEL_SWEEP_POPULATION_CONFIG
    KERNEL_SWEEP_OBSERVATION = KERNEL_SWEEP_OBSERVATION_CONFIG


cfg = Configs()
