from dataclasses import dataclass, field, replace
from typing import Literal

from pygom import KernelVersion as KV

ProblemType = Literal["synthetic", "pmlb"]

KERNEL_VERSIONS: tuple[KV] = (
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


@dataclass
class ExperimentConfig:
    name: str

    # Problem space
    problems: list[ProblemConfig]
    population_sizes: list[int]
    num_observations: list[int]
    num_features: list[int]
    templates: list[list[tuple[int, int]]]
    operator_sets: list[OperatorSet]

    # Execution
    cpu: CPUConfig = field(default_factory=CPUConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)


DATASETS: dict[str, ProblemConfig] = {
    "sin": ProblemConfig(
        name="sin(3.1 * x0 + 2.7)",
        type="synthetic",
        observations=1_000_000,
        features=10,
    ),
    "4544": ProblemConfig(
        name="4544_GeographicalOriginalofMusic",
        type="pmlb",
        observations=1059,
        features=117,
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
}

OPERATOR_SETS: dict[str, OperatorSet] = {
    "small": OperatorSet("small", "+,-,*,/"),
    "exp": OperatorSet("exp", "+,-,*,/,sin,cos,exp,log"),
    "all": OperatorSet(
        "all",
        "+,-,*,/,sin,cos,exp,log,square,sqrt,pow,abs,min,max",
    ),
}

KERNELS_CONFIG = ExperimentConfig(
    name="kernel_sweep",
    problems=[DATASETS["1191"]],
    population_sizes=[2**i for i in range(9, 16, 2)],  # 512, 2048, 8192, 32768
    num_observations=[10**i for i in range(4, 7)],  # 10_000, 100_000, 1_000_000
    num_features=[DATASETS["1191"].features // 2, DATASETS["1191"].features],  # 9, 18
    templates=[[(2, 4)]],
    operator_sets=[OPERATOR_SETS["small"], OPERATOR_SETS["all"]],
    cpu=CPUConfig(enabled=False),
    gpu=GPUConfig(kernels=KERNEL_VERSIONS),
)

CPU_GPU_CONFIG = ExperimentConfig(
    name="cpu_vs_gpu",
    problems=list(DATASETS.values()),
    population_sizes=[2**i for i in range(8, 16)],  # 256 - 32768
    num_observations=[10**i for i in range(1, 7)],  # 10 - 1_000_000
    num_features=[2**i for i in range(0, 8)],  # 1 - 128
    templates=[[(2, 3)], [(2, 4)], [(2, 5)]],
    operator_sets=list(v for v in OPERATOR_SETS.values()),
    cpu=CPUConfig(enabled=True),
    gpu=GPUConfig(enabled=True),
)

TEST_CONFIG = ExperimentConfig(
    name="test",
    problems=[DATASETS["sin"]],
    population_sizes=[256, 512],
    num_observations=[10, 100],
    num_features=[1],
    templates=[[(2, 3)], [(2, 4)]],
    operator_sets=[OPERATOR_SETS["small"], OPERATOR_SETS["all"]],
    cpu=CPUConfig(enabled=True),
    gpu=GPUConfig(enabled=True, kernels=(KV.baseline, KV.single_kernel)),
)
