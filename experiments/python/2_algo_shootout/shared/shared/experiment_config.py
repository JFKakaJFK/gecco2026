from dataclasses import dataclass, field
from typing import Literal

from shared.dataset_config import DATASETS, DatasetConfig

# Only used for GPU GP-GOMEA
KernelType = Literal[
    "baseline",
    "restrict",
    "shared_memory",
    "block_reduce",
    "single_kernel",
    "single_kernel_fmaf",
    "single_kernel_inplace",
    "hybrid",
]

OPERATOR_SETS: dict[str, str] = {
    "arith": "+,-,*,/",
    "trig": "+,-,*,/,sin,cos",
    "square": "+,-,*,/,square,sqrt",
    "exp": "+,-,*,/,exp,log",
    "paper": "+,-,*,/,sin,cos,square,sqrt,exp,log,pow",
    "all": "+,-,*,/,sin,cos,square,sqrt,exp,log,pow,abs,min,max",
}


@dataclass(frozen=True)
class GPUConfig:
    enabled: bool = True
    kernels: tuple[KernelType] = ("single_kernel_inplace",)


@dataclass(frozen=True)
class CPUConfig:
    enabled: bool = True


@dataclass(frozen=True)
class TemplateConfig:
    branching_factor: int = 2
    depth: int = 4
    max_tree_len: int | None = None


@dataclass(frozen=True)
class OperatorSet:
    name: str
    operators: str


@dataclass(frozen=True)
class ExperimentConfig:
    name: str

    # Problem Space
    datasets: list[DatasetConfig]
    population_sizes: list[int] | None  # None when in search space
    num_observations: list[int] | None
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
    max_duration: int | None  # Assumed to be minutes

    # Search Space
    search_space: dict[str, list[int]] = field(default_factory=dict)
    required_rate: float | None = None

    def determine_observations(self, num_observations=None):
        temp = (
            [num_observations]
            if self.num_observations is None
            else self.num_observations
        )

        return [
            int(obs * (1 - self.test_size) * (1 - 1 / self.num_folds)) for obs in temp
        ]

    def get_target_objectives(self, dataset: DatasetConfig) -> list[float] | None:
        if dataset not in self.datasets:
            print(dataset)
            raise ValueError("Dataset not found in config")

        return dataset.target_objectives()


@dataclass(frozen=True, init=False)
class PaperExperimentConfig(ExperimentConfig):
    def __init__(self, name: str, gpu_enabled: bool):
        super().__init__(
            # Problem space
            name=name,
            datasets=[DATASETS[name]],
            population_sizes=[2**i for i in range(7, 17)],  # 128 - 65536
            num_observations=None,
            num_features=None,
            templates=[TemplateConfig(2, 4, 31), TemplateConfig(2, 6, 127)],
            operator_sets=["paper"],
            # Execution Parameters
            use_target=False,
            num_folds=30,
            num_iterations=1,
            test_size=0,
            cpu=CPUConfig(enabled=not gpu_enabled),
            gpu=GPUConfig(enabled=gpu_enabled),
            max_duration=10 if gpu_enabled else 60,
        )


####################
### TEST Configs ###
####################

TEST_CPU_CONFIG = ExperimentConfig(
    name="test_cpu",
    datasets=[DATASETS["daily_demand"]],
    population_sizes=[128, 256],
    num_observations=None,
    num_features=None,
    templates=[TemplateConfig(2, 4)],
    operator_sets=["arith"],
    # Execution Parameters
    use_target=False,
    num_folds=3,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(enabled=True),
    gpu=GPUConfig(enabled=False),
    max_duration=1,
)

TEST_GPU_CONFIG = ExperimentConfig(
    name="test_gpu",
    datasets=[DATASETS["daily_demand"]],
    population_sizes=[128, 256],
    num_observations=None,
    num_features=None,
    templates=[TemplateConfig(2, 4)],
    operator_sets=["arith"],
    # Execution Parameters
    use_target=False,
    num_folds=3,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(enabled=False),
    gpu=GPUConfig(enabled=True),
    max_duration=1,
)

#####################
### Paper Configs ###
#####################

### CPU Configs ###
DAILY_DEMAND_CPU_CONFIG = PaperExperimentConfig("daily_demand", False)
AUTO_MPG_CPU_CONFIG = PaperExperimentConfig("auto_mpg", False)
CALIFORNIA_CPU_CONFIG = PaperExperimentConfig("california_housing", False)
FEYNMAN_CPU_CONFIG = PaperExperimentConfig("feynman_I_9_18", False)

### GPU Configs ###
DAILY_DEMAND_GPU_CONFIG = PaperExperimentConfig("daily_demand", True)
AUTO_MPG_GPU_CONFIG = PaperExperimentConfig("auto_mpg", True)
CALIFORNIA_GPU_CONFIG = PaperExperimentConfig("california_housing", True)
FEYNMAN_GPU_CONFIG = PaperExperimentConfig("feynman_I_9_18", True)


class Configs:
    TEST_CPU = TEST_CPU_CONFIG
    TEST_GPU = TEST_GPU_CONFIG

    # Paper experiments
    DAILY_DEMAND_CPU = DAILY_DEMAND_CPU_CONFIG
    AUTO_MPG_CPU = AUTO_MPG_CPU_CONFIG
    CALIFORNIA_CPU = CALIFORNIA_CPU_CONFIG
    FEYNMAN_CPU = FEYNMAN_CPU_CONFIG

    DAILY_DEMAND_GPU = DAILY_DEMAND_GPU_CONFIG
    AUTO_MPG_GPU = AUTO_MPG_GPU_CONFIG
    CALIFORNIA_GPU = CALIFORNIA_GPU_CONFIG
    FEYNMAN_GPU = FEYNMAN_GPU_CONFIG


cfg = Configs()
