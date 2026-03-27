from dataclasses import dataclass, field
from typing import Literal, TypeVar

from pygom import KernelVersion as KV

from src.dataset_config import DATASETS, DatasetConfig

T = TypeVar("T")

ProblemType = Literal["synthetic", "pmlb"]

ALL_KV: tuple[KV, ...] = (
    KV.baseline,
    KV.restrict,
    KV.shared_memory,
    KV.block_reduce,
    KV.single_kernel,
    KV.single_kernel_fmaf,
    KV.single_kernel_inplace,
)

MAIN_KV: tuple[KV, ...] = (KV.shared_memory, KV.block_reduce, KV.single_kernel_inplace)
BASELINE_KV: tuple[KV, ...] = (KV.baseline, KV.restrict, KV.shared_memory)
SINGLE_KV: tuple[KV, ...] = (
    KV.single_kernel,
    KV.single_kernel_fmaf,
    KV.single_kernel_inplace,
)


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
    kernels: tuple[KV, ...] = (KV.single_kernel_inplace,)


@dataclass(frozen=True)
class CPUConfig:
    enabled: bool = True


OPERATOR_SETS: dict[str, str] = {
    "arith": "+,-,*,/",
    "squaresub": "+,-,*,/,squaresub",
    "trig": "+,-,*,/,sin,cos",
    "square": "+,-,*,/,square",
    "exp": "+,-,*,/,exp,log",
    "paper": "+,-,*,/,sin,cos,exp,log,square,sqrt,pow",
    "all": "+,-,*,/,sin,cos,square,sqrt,exp,log,pow,abs,min,max",
}


@dataclass
class ExperimentConfig:
    """
    Attributes:
        population_sizes (list[int] | None): Sets the population sizes that
        will be included in the experiment. When equal to 'None', it is assumed
        that it is in the search space
        num_features (list[int] | None): Determines the number of features to be used.
        If equal to 'None', the amount of features available in the dataset will be used.
    """

    name: str

    # Problem Space
    datasets: list[DatasetConfig]
    population_sizes: list[int] | None  # None when in search space
    num_observations: list[int]
    num_features: list[int] | None  # Will take the num_features from DatasetConfig
    templates: list[TemplateConfig]
    operator_sets: list[str]

    # Execution
    use_target: bool
    num_folds: int
    num_iterations: int
    test_size: float
    cpu: CPUConfig
    gpu: GPUConfig
    max_duration: int | None

    # Search Space
    search_space: dict[str, list[int]] = field(default_factory=dict)
    required_rate: float | None = None

    def determine_observations(self):
        return [
            int(obs * (1 - self.test_size) * (1 - 1 / self.num_folds))
            for obs in self.num_observations
        ]

    def get_target_objectives(self, dataset: DatasetConfig) -> list[float] | None:
        if dataset not in self.datasets:
            print(dataset)
            raise ValueError("Dataset not found in config")

        return dataset.target_objectives()


#####################
### Paper Configs ###
#####################


FEYNMAN_EXACT_CONFIG = ExperimentConfig(
    # Problem Space
    name="feynman",
    datasets=[DATASETS["feynman_I_9_18"]],
    population_sizes=[2**i for i in range(14, 17)],  # 128 - 1048576
    num_observations=[100_000],
    num_features=None,
    templates=[TemplateConfig(2, 6)],
    operator_sets=["square"],
    # Execution Parameters
    use_target=False,
    num_folds=9,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(enabled=False),
    gpu=GPUConfig(enabled=True, kernels=(KV.single_kernel_inplace,)),
    max_duration=60,
)

ADDITION_CONFIG = ExperimentConfig(
    name="1_addition",
    datasets=[DATASETS["1_addition"]],
    population_sizes=[65536],
    num_observations=[1_000_000],
    num_features=None,
    templates=[TemplateConfig(2, 6), TemplateConfig(2, 8)],
    operator_sets=["square"],
    # Execution parameters
    use_target=False,
    num_folds=9,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(enabled=False),
    gpu=GPUConfig(enabled=True, kernels=(KV.single_kernel_inplace,)),
    max_duration=240,
)

DIVISION_CONFIG = ExperimentConfig(
    name="2_division",
    datasets=[DATASETS["2_division"]],
    population_sizes=[65536],
    num_observations=[1_000_000],
    num_features=None,
    templates=[TemplateConfig(2, 6), TemplateConfig(2, 8)],
    operator_sets=["square"],
    # Execution parameters
    use_target=False,
    num_folds=9,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(enabled=False),
    gpu=GPUConfig(enabled=True, kernels=(KV.single_kernel_inplace,)),
    max_duration=240,
)

SUBTRACTION_CONFIG = ExperimentConfig(
    name="3_subtraction",
    datasets=[DATASETS["3_subtraction"]],
    population_sizes=[65536],
    num_observations=[1_000_000],
    num_features=None,
    templates=[TemplateConfig(2, 6), TemplateConfig(2, 8)],
    operator_sets=["square"],
    # Execution parameters
    use_target=False,
    num_folds=9,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(enabled=False),
    gpu=GPUConfig(enabled=True, kernels=(KV.single_kernel_inplace,)),
    max_duration=240,
)

MULTIPLICATION_CONFIG = ExperimentConfig(
    name="4_multiplication",
    datasets=[DATASETS["4_multiplication"]],
    population_sizes=[65536],
    num_observations=[1_000_000],
    num_features=None,
    templates=[TemplateConfig(2, 6), TemplateConfig(2, 8)],
    operator_sets=["square"],
    # Execution parameters
    use_target=False,
    num_folds=9,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(enabled=False),
    gpu=GPUConfig(enabled=True, kernels=(KV.single_kernel_inplace,)),
    max_duration=240,
)

SQUARE_CONFIG = ExperimentConfig(
    name="5_square",
    datasets=[DATASETS["5_square"]],
    population_sizes=[65536],
    num_observations=[1_000_000],
    num_features=None,
    templates=[TemplateConfig(2, 6), TemplateConfig(2, 8)],
    operator_sets=["square"],
    # Execution parameters
    use_target=False,
    num_folds=9,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(enabled=False),
    gpu=GPUConfig(enabled=True, kernels=(KV.single_kernel_inplace,)),
    max_duration=240,
)

FEYNMAN_MILLION_CONFIG = ExperimentConfig(
    name="feynman_million",
    datasets=[DATASETS["feynman_million"]],
    population_sizes=[65536],
    num_observations=[1_000_000],
    num_features=None,
    templates=[TemplateConfig(2, 6), TemplateConfig(2, 8)],
    operator_sets=["square"],
    # Execution Parameters
    use_target=False,
    num_folds=30,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(enabled=False),
    gpu=GPUConfig(enabled=True, kernels=(KV.single_kernel_inplace,)),
    max_duration=240,
)


class Configs:
    # Paper experiments
    FEYNMAN_MILLION = FEYNMAN_MILLION_CONFIG

    ADDITION = ADDITION_CONFIG
    DIVISION = DIVISION_CONFIG
    SUBTRACTION = SUBTRACTION_CONFIG
    MULTIPLICATION = MULTIPLICATION_CONFIG
    SQUARE = SQUARE_CONFIG


cfg = Configs()
