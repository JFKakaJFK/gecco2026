from dataclasses import dataclass, field
from typing import Literal, TypeVar

from pygom import KernelVersion as KV

from src.experiment.dataset_config import DATASETS, DatasetConfig

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

MAIN_KV: tuple[KV, ...] = (
    KV.shared_memory,
    KV.block_reduce,
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

    # Search Space
    search_space: dict[str, list[int]] = field(default_factory=dict)
    required_rate: float | None = None

    def determine_observations(self, observations):
        return [
            int(obs * (1 - self.test_size) * (1 - 1 / self.num_folds))
            for obs in observations
        ]

    def get_target_objectives(self, dataset: DatasetConfig) -> list[float] | None:
        if dataset not in self.datasets:
            print(dataset)
            raise ValueError("Dataset not found in config")

        return dataset.target_objectives()


TEST_EXECUTION_CONFIG = ExperimentConfig(
    name="test_execution",
    # Problem Definition
    datasets=[DATASETS["feynman_I_11_19"], DATASETS["feynman_I_12_1"]],
    population_sizes=[64, 128],
    num_observations=[100, 200],
    num_features=None,
    templates=[TemplateConfig(2, 3), TemplateConfig(2, 4)],
    operator_sets=["arith"],
    # Execution Parameters
    use_target=True,
    num_folds=3,
    num_iterations=2,
    test_size=0.25,
    cpu=CPUConfig(enabled=True),
    gpu=GPUConfig(enabled=True, kernels=(KV.block_reduce, KV.single_kernel)),
)

TEST_SEARCH_CONFIG = ExperimentConfig(
    name="test_search",
    # Problem Definition
    datasets=[DATASETS["feynman_I_11_19"], DATASETS["feynman_I_12_1"]],
    population_sizes=None,
    num_observations=[100, 1_000],
    num_features=None,
    templates=[TemplateConfig(2, 4)],
    operator_sets=["arith"],
    # Execution Parameters
    use_target=True,
    num_folds=5,
    num_iterations=1,
    test_size=0.25,
    cpu=CPUConfig(enabled=True),
    gpu=GPUConfig(enabled=True, kernels=(KV.block_reduce, KV.single_kernel)),
    # Search Space
    search_space={"population_size": [8, 16, 32, 64, 128, 256, 512, 1024, 2048]},
    required_rate=0.8,
)

SEARCH_ARITH_FEYNMAN_CONFIG = ExperimentConfig(
    # Problem Space
    name="search_arith_feynman",
    datasets=[
        DATASETS["feynman_I_11_19"],
        DATASETS["feynman_I_12_1"],
        DATASETS["feynman_I_13_12"],
        DATASETS["feynman_I_18_4"],
    ],
    population_sizes=None,
    num_observations=[10**i for i in range(2, 6)],  # 100 - 100_000
    num_features=None,
    templates=[TemplateConfig(2, 4)],
    operator_sets=["arith"],
    # Execution Parameters
    use_target=True,
    num_folds=50,
    num_iterations=1,
    test_size=0.25,
    cpu=CPUConfig(enabled=False),
    gpu=GPUConfig(enabled=True, kernels=ALL_KV),
    # Search Space
    search_space={"population_size": [2**i for i in range(3, 17)]},  # 8 - 65536
    required_rate=49 / 50,
)

SEARCH_TRIG_FEYNMAN_CONFIG = ExperimentConfig(
    # Problem Space
    name="search_trig_feynman",
    datasets=[
        DATASETS["feynman_I_12_11"],
        DATASETS["feynman_II_15_4"],
        DATASETS["feynman_III_15_12"],
        DATASETS["feynman_III_17_37"],
    ],
    population_sizes=None,
    num_observations=[10**i for i in range(2, 6)],  # 100 - 100_000
    num_features=None,
    templates=[TemplateConfig(2, 4)],
    operator_sets=["trig"],
    # Execution Parameters
    use_target=True,
    num_folds=50,
    num_iterations=1,
    test_size=0.25,
    cpu=CPUConfig(enabled=False),
    gpu=GPUConfig(enabled=True, kernels=ALL_KV),
    # Search Space
    search_space={"population_size": [2**i for i in range(3, 17)]},  # 8 - 65536
    required_rate=49 / 50,
)

SEARCH_SQUARE_FEYNMAN_CONFIG = ExperimentConfig(
    # Problem Space
    name="search_square_feynman",
    datasets=[
        DATASETS["feynman_I_8_14"],
        DATASETS["feynman_II_24_17"],
    ],
    population_sizes=None,
    num_observations=[10**i for i in range(2, 6)],  # 100 - 100_000
    num_features=None,
    templates=[TemplateConfig(2, 4)],
    operator_sets=["square"],
    # Execution Parameters
    use_target=True,
    num_folds=50,
    num_iterations=1,
    test_size=0.25,
    cpu=CPUConfig(enabled=False),
    gpu=GPUConfig(enabled=True, kernels=ALL_KV),
    # Search Space
    search_space={"population_size": [2**i for i in range(3, 17)]},  # 8 - 65536
    required_rate=49 / 50,
)

SEARCH_EXP_FEYNMAN_CONFIG = ExperimentConfig(
    # Problem Space
    name="search_exp_feynman",
    datasets=[
        DATASETS["feynman_I_44_4"],
    ],
    population_sizes=None,
    num_observations=[10**i for i in range(2, 6)],  # 100 - 100_000
    num_features=None,
    templates=[TemplateConfig(2, 4)],
    operator_sets=["exp"],
    # Execution Parameters
    use_target=True,
    num_folds=50,
    num_iterations=1,
    test_size=0.25,
    cpu=CPUConfig(enabled=False),
    gpu=GPUConfig(enabled=True, kernels=ALL_KV),
    # Search Space
    search_space={"population_size": [2**i for i in range(3, 17)]},  # 8 - 65536
    required_rate=49 / 50,
)


class Configs:
    # Test experiment
    TEST_EXECUTION = TEST_EXECUTION_CONFIG
    TEST_SEARCH = TEST_SEARCH_CONFIG

    # Feynman experiments
    # ARITH_FEYNMAN = ARITH_FEYNMAN_CONFIG
    # TRIG_FEYNMAN = TRIG_FEYNMAN_CONFIG
    # SQRT_FEYNMAN = SQRT_FEYNMAN_CONFIG
    # EXP_FEYNMAN = EXP_FEYNMAN_CONFIG


cfg = Configs()
