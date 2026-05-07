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
    "modular": "+,-,*,/,sin,cos,log,sqrt,abs",
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
    subtrees: tuple[tuple[int, int], ...] = ()


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
### Test Configs ###
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

##################################
### Experiment 1: CPU Shootout ###
##################################

CPU_SHOOTOUT_CONFIG = ExperimentConfig(
    # Problem Space
    name="feynman",
    datasets=[DATASETS["feynman_I_9_18"]],
    population_sizes=[128],
    num_observations=[100_000],
    num_features=None,
    templates=[TemplateConfig(2, 6)],
    operator_sets=["paper"],
    # Execution Parameters
    use_target=False,
    num_folds=30,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(enabled=True),
    gpu=GPUConfig(enabled=False),
    max_duration=60,
)

########################################
### Experiment 2: Algorithm Shootout ###
########################################

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

####################################
### Experiment 3: Exact Solution ###
####################################

ADDITION_CONFIG = ExperimentConfig(
    name="1_addition",
    datasets=[DATASETS["1_addition"]],
    population_sizes=[65536],
    num_observations=[100_000],
    num_features=None,
    templates=[TemplateConfig(2, 6), TemplateConfig(2, 8)],
    operator_sets=["square"],
    # Execution parameters
    use_target=False,
    num_folds=9,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(enabled=False),
    gpu=GPUConfig(enabled=True, kernels=("single_kernel_inplace",)),
    max_duration=240,
)

DIVISION_CONFIG = ExperimentConfig(
    name="2_division",
    datasets=[DATASETS["2_division"]],
    population_sizes=[65536],
    num_observations=[100_000],
    num_features=None,
    templates=[TemplateConfig(2, 6), TemplateConfig(2, 8)],
    operator_sets=["square"],
    # Execution parameters
    use_target=False,
    num_folds=9,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(enabled=False),
    gpu=GPUConfig(enabled=True, kernels=("single_kernel_inplace",)),
    max_duration=240,
)

SUBTRACTION_CONFIG = ExperimentConfig(
    name="3_subtraction",
    datasets=[DATASETS["3_subtraction"]],
    population_sizes=[65536],
    num_observations=[100_000],
    num_features=None,
    templates=[TemplateConfig(2, 6), TemplateConfig(2, 8)],
    operator_sets=["square"],
    # Execution parameters
    use_target=False,
    num_folds=9,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(enabled=False),
    gpu=GPUConfig(enabled=True, kernels=("single_kernel_inplace",)),
    max_duration=240,
)

MULTIPLICATION_CONFIG = ExperimentConfig(
    name="4_multiplication",
    datasets=[DATASETS["4_multiplication"]],
    population_sizes=[65536],
    num_observations=[100_000],
    num_features=None,
    templates=[TemplateConfig(2, 6), TemplateConfig(2, 8)],
    operator_sets=["square"],
    # Execution parameters
    use_target=False,
    num_folds=9,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(enabled=False),
    gpu=GPUConfig(enabled=True, kernels=("single_kernel_inplace",)),
    max_duration=240,
)

SQUARE_CONFIG = ExperimentConfig(
    name="5_square",
    datasets=[DATASETS["5_square"]],
    population_sizes=[65536],
    num_observations=[100_000],
    num_features=None,
    templates=[TemplateConfig(2, 6), TemplateConfig(2, 8)],
    operator_sets=["square"],
    # Execution parameters
    use_target=False,
    num_folds=9,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(enabled=False),
    gpu=GPUConfig(enabled=True, kernels=("single_kernel_inplace",)),
    max_duration=240,
)

######################################
### Experiment 4: Modular GP-GOMEA ###
######################################

MODULAR_DOT_PRODUCT_CPU = ExperimentConfig(
    name="dot_product",
    datasets=[DATASETS["feynman_I_11_19"]],
    population_sizes=[2**i for i in range(7, 17)],
    num_observations=None,
    num_features=None,
    templates=[TemplateConfig(2, 6), TemplateConfig(2, 4, subtrees=((2, 3), (2, 3)))],
    operator_sets=["square"],
    # Execution parameters
    use_target=False,
    num_folds=9,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(enabled=True),
    gpu=GPUConfig(enabled=False),
    max_duration=60,
)

MODULAR_DISTANCE_CPU = ExperimentConfig(
    name="distance",
    datasets=[DATASETS["feynman_I_8_14"]],
    population_sizes=[2**i for i in range(7, 17)],
    num_observations=None,
    num_features=None,
    templates=[TemplateConfig(2, 6), TemplateConfig(2, 4, subtrees=((2, 3), (2, 3)))],
    operator_sets=["square"],
    # Execution parameters
    use_target=False,
    num_folds=9,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(enabled=True),
    gpu=GPUConfig(enabled=False),
    max_duration=60,
)

MODULAR_GRAVITY_CPU = ExperimentConfig(
    name="gravity",
    datasets=[DATASETS["feynman_I_9_18"]],
    population_sizes=[2**i for i in range(7, 17)],
    num_observations=None,
    num_features=None,
    templates=[TemplateConfig(2, 6), TemplateConfig(2, 4, subtrees=((2, 3), (2, 3)))],
    operator_sets=["square"],
    # Execution parameters
    use_target=False,
    num_folds=9,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(enabled=True),
    gpu=GPUConfig(enabled=False),
    max_duration=60,
)

MODULAR_1_CPU_CONFIG = ExperimentConfig(
    name="modular_1",
    datasets=[DATASETS["modular_1"]],
    population_sizes=[2**i for i in range(10, 16)],
    num_observations=None,
    num_features=None,
    templates=[
        TemplateConfig(2, 7),
        TemplateConfig(2, 4, subtrees=((2, 4), (2, 4), (2, 4))),
    ],
    operator_sets=["modular"],
    # Execution parameters
    use_target=False,
    num_folds=9,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(enabled=True),
    gpu=GPUConfig(enabled=False),
    max_duration=240,
)

MODULAR_2_CPU_CONFIG = ExperimentConfig(
    name="modular_2",
    datasets=[DATASETS["modular_2"]],
    population_sizes=[2**i for i in range(10, 16)],
    num_observations=None,
    num_features=None,
    templates=[
        TemplateConfig(2, 7),
        TemplateConfig(2, 4, subtrees=((2, 4), (2, 4), (2, 4))),
    ],
    operator_sets=["modular"],
    # Execution parameters
    use_target=False,
    num_folds=9,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(enabled=True),
    gpu=GPUConfig(enabled=False),
    max_duration=240,
)

MODULAR_3_CPU_CONFIG = ExperimentConfig(
    name="modular_3",
    datasets=[DATASETS["modular_3"]],
    population_sizes=[2**i for i in range(10, 16)],
    num_observations=None,
    num_features=None,
    templates=[
        TemplateConfig(2, 7),
        TemplateConfig(2, 4, subtrees=((2, 4), (2, 4), (2, 4))),
    ],
    operator_sets=["modular"],
    # Execution parameters
    use_target=False,
    num_folds=9,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(enabled=True),
    gpu=GPUConfig(enabled=False),
    max_duration=240,
)

MODULAR_4_CPU_CONFIG = ExperimentConfig(
    name="modular_4",
    datasets=[DATASETS["modular_4"]],
    population_sizes=[2**i for i in range(10, 16)],
    num_observations=None,
    num_features=None,
    templates=[
        TemplateConfig(2, 7),
        TemplateConfig(2, 4, subtrees=((2, 4), (2, 4), (2, 4))),
    ],
    operator_sets=["modular"],
    # Execution parameters
    use_target=False,
    num_folds=9,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(enabled=True),
    gpu=GPUConfig(enabled=False),
    max_duration=240,
)

MODULAR_5_CPU_CONFIG = ExperimentConfig(
    name="modular_5",
    datasets=[DATASETS["modular_5"]],
    population_sizes=[2**i for i in range(10, 16)],
    num_observations=None,
    num_features=None,
    templates=[
        TemplateConfig(2, 7),
        TemplateConfig(2, 4, subtrees=((2, 4), (2, 4), (2, 4))),
    ],
    operator_sets=["modular"],
    # Execution parameters
    use_target=False,
    num_folds=9,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(enabled=True),
    gpu=GPUConfig(enabled=False),
    max_duration=240,
)

MODULAR_DOT_PRODUCT_GPU = ExperimentConfig(
    name="dot_product",
    datasets=[DATASETS["feynman_I_11_19"]],
    population_sizes=[2**i for i in range(7, 17)],
    num_observations=None,
    num_features=None,
    templates=[TemplateConfig(2, 6), TemplateConfig(2, 4, subtrees=((2, 3), (2, 3)))],
    operator_sets=["square"],
    # Execution parameters
    use_target=False,
    num_folds=9,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(enabled=False),
    gpu=GPUConfig(enabled=True, kernels=("single_kernel_inplace",)),
    max_duration=10,
)

MODULAR_DISTANCE_GPU = ExperimentConfig(
    name="distance",
    datasets=[DATASETS["feynman_I_8_14"]],
    population_sizes=[2**i for i in range(7, 17)],
    num_observations=None,
    num_features=None,
    templates=[TemplateConfig(2, 6), TemplateConfig(2, 4, subtrees=((2, 3), (2, 3)))],
    operator_sets=["square"],
    # Execution parameters
    use_target=False,
    num_folds=9,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(enabled=False),
    gpu=GPUConfig(enabled=True, kernels=("single_kernel_inplace",)),
    max_duration=10,
)

MODULAR_GRAVITY_GPU = ExperimentConfig(
    name="gravity",
    datasets=[DATASETS["feynman_I_9_18"]],
    population_sizes=[2**i for i in range(7, 17)],
    num_observations=None,
    num_features=None,
    templates=[TemplateConfig(2, 6), TemplateConfig(2, 4, subtrees=((2, 3), (2, 3)))],
    operator_sets=["square"],
    # Execution parameters
    use_target=False,
    num_folds=9,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(enabled=False),
    gpu=GPUConfig(enabled=True, kernels=("single_kernel_inplace",)),
    max_duration=10,
)

MODULAR_1_GPU_CONFIG = ExperimentConfig(
    name="modular_1",
    datasets=[DATASETS["modular_1"]],
    population_sizes=[2**i for i in range(10, 16)],
    num_observations=None,
    num_features=None,
    templates=[
        TemplateConfig(2, 7),
        TemplateConfig(2, 4, subtrees=((2, 4), (2, 4), (2, 4))),
    ],
    operator_sets=["modular"],
    # Execution parameters
    use_target=False,
    num_folds=9,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(enabled=False),
    gpu=GPUConfig(enabled=True, kernels=("single_kernel_inplace",)),
    max_duration=30,
)

MODULAR_2_GPU_CONFIG = ExperimentConfig(
    name="modular_2",
    datasets=[DATASETS["modular_2"]],
    population_sizes=[2**i for i in range(10, 16)],
    num_observations=None,
    num_features=None,
    templates=[
        TemplateConfig(2, 7),
        TemplateConfig(2, 4, subtrees=((2, 4), (2, 4), (2, 4))),
    ],
    operator_sets=["modular"],
    # Execution parameters
    use_target=False,
    num_folds=9,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(enabled=False),
    gpu=GPUConfig(enabled=True, kernels=("single_kernel_inplace",)),
    max_duration=30,
)

MODULAR_3_GPU_CONFIG = ExperimentConfig(
    name="modular_3",
    datasets=[DATASETS["modular_3"]],
    population_sizes=[2**i for i in range(10, 16)],
    num_observations=None,
    num_features=None,
    templates=[
        TemplateConfig(2, 7),
        TemplateConfig(2, 4, subtrees=((2, 4), (2, 4), (2, 4))),
    ],
    operator_sets=["modular"],
    # Execution parameters
    use_target=False,
    num_folds=9,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(enabled=False),
    gpu=GPUConfig(enabled=True, kernels=("single_kernel_inplace",)),
    max_duration=30,
)

MODULAR_4_GPU_CONFIG = ExperimentConfig(
    name="modular_4",
    datasets=[DATASETS["modular_4"]],
    population_sizes=[2**i for i in range(10, 16)],
    num_observations=None,
    num_features=None,
    templates=[
        TemplateConfig(2, 7),
        TemplateConfig(2, 4, subtrees=((2, 4), (2, 4), (2, 4))),
    ],
    operator_sets=["modular"],
    # Execution parameters
    use_target=False,
    num_folds=9,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(enabled=False),
    gpu=GPUConfig(enabled=True, kernels=("single_kernel_inplace",)),
    max_duration=30,
)

MODULAR_5_GPU_CONFIG = ExperimentConfig(
    name="modular_5",
    datasets=[DATASETS["modular_5"]],
    population_sizes=[2**i for i in range(10, 16)],
    num_observations=None,
    num_features=None,
    templates=[
        TemplateConfig(2, 7),
        TemplateConfig(2, 4, subtrees=((2, 4), (2, 4), (2, 4))),
    ],
    operator_sets=["modular"],
    # Execution parameters
    use_target=False,
    num_folds=9,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(enabled=False),
    gpu=GPUConfig(enabled=True, kernels=("single_kernel_inplace",)),
    max_duration=30,
)

######################################
### Experiment 5: Hybrid Kernel ###
######################################

HYBRID_DAILY_DEMAND_CONFIG = ExperimentConfig(
    name="daily_demand",
    datasets=[DATASETS["daily_demand"]],
    population_sizes=[2**i for i in range(5, 11)],  # 32 - 1024
    num_observations=None,
    num_features=None,
    templates=[TemplateConfig(2, 4), TemplateConfig(2, 6)],
    operator_sets=["paper"],
    # Execution Parameters
    use_target=False,
    num_folds=9,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(False),
    gpu=GPUConfig(enabled=True, kernels=("single_kernel_inplace", "hybrid")),
    max_duration=10,
)

HYBRID_AUTO_MPG_CONFIG = ExperimentConfig(
    name="auto_mpg",
    datasets=[DATASETS["auto_mpg"]],
    population_sizes=[2**i for i in range(5, 11)],  # 32 - 1024
    num_observations=None,
    num_features=None,
    templates=[TemplateConfig(2, 4), TemplateConfig(2, 6)],
    operator_sets=["paper"],
    # Execution Parameters
    use_target=False,
    num_folds=9,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(False),
    gpu=GPUConfig(enabled=True, kernels=("single_kernel_inplace", "hybrid")),
    max_duration=10,
)

HYBRID_CALIFORNIA_CONFIG = ExperimentConfig(
    name="california_housing",
    datasets=[DATASETS["california_housing"]],
    population_sizes=[2**i for i in range(5, 11)],  # 32 - 1024
    num_observations=None,
    num_features=None,
    templates=[TemplateConfig(2, 4), TemplateConfig(2, 6)],
    operator_sets=["paper"],
    # Execution Parameters
    use_target=False,
    num_folds=9,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(False),
    gpu=GPUConfig(enabled=True, kernels=("single_kernel_inplace", "hybrid")),
    max_duration=10,
)

HYBRID_FEYNMAN_CONFIG = ExperimentConfig(
    name="feynman_I_9_18",
    datasets=[DATASETS["feynman_I_9_18"]],
    population_sizes=[2**i for i in range(5, 11)],  # 32 - 1024
    num_observations=None,
    num_features=None,
    templates=[TemplateConfig(2, 4), TemplateConfig(2, 6)],
    operator_sets=["paper"],
    # Execution Parameters
    use_target=False,
    num_folds=9,
    num_iterations=1,
    test_size=0,
    cpu=CPUConfig(False),
    gpu=GPUConfig(enabled=True, kernels=("single_kernel_inplace", "hybrid")),
    max_duration=10,
)


class Configs:
    TEST_CPU = TEST_CPU_CONFIG
    TEST_GPU = TEST_GPU_CONFIG

    # Experiment 1
    CPU_SHOOTOUT = CPU_SHOOTOUT_CONFIG

    # Experiment 2
    DAILY_DEMAND_CPU = DAILY_DEMAND_CPU_CONFIG
    AUTO_MPG_CPU = AUTO_MPG_CPU_CONFIG
    CALIFORNIA_CPU = CALIFORNIA_CPU_CONFIG
    FEYNMAN_CPU = FEYNMAN_CPU_CONFIG

    DAILY_DEMAND_GPU = DAILY_DEMAND_GPU_CONFIG
    AUTO_MPG_GPU = AUTO_MPG_GPU_CONFIG
    CALIFORNIA_GPU = CALIFORNIA_GPU_CONFIG
    FEYNMAN_GPU = FEYNMAN_GPU_CONFIG

    # Experiment 3
    ADDITION = ADDITION_CONFIG
    DIVISION = DIVISION_CONFIG
    SUBTRACTION = SUBTRACTION_CONFIG
    MULTIPLICATION = MULTIPLICATION_CONFIG
    SQUARE = SQUARE_CONFIG

    # Experiment 4
    DOT_PRODUCT_CPU = MODULAR_DOT_PRODUCT_CPU
    DISTANCE_CPU = MODULAR_DISTANCE_CPU
    GRAVITY_CPU = MODULAR_GRAVITY_CPU

    MODULAR_1_CPU = MODULAR_1_CPU_CONFIG
    MODULAR_2_CPU = MODULAR_2_CPU_CONFIG
    MODULAR_3_CPU = MODULAR_3_CPU_CONFIG
    MODULAR_4_CPU = MODULAR_4_CPU_CONFIG
    MODULAR_5_CPU = MODULAR_5_CPU_CONFIG

    DOT_PRODUCT_GPU = MODULAR_DOT_PRODUCT_GPU
    DISTANCE_GPU = MODULAR_DISTANCE_GPU
    GRAVITY_GPU = MODULAR_GRAVITY_GPU

    # Experiment 5
    HYBRID_DAILY_DEMAND = HYBRID_DAILY_DEMAND_CONFIG
    HYBRID_AUTO_MPG = HYBRID_AUTO_MPG_CONFIG
    HYBRID_CALIFORNIA = HYBRID_CALIFORNIA_CONFIG
    HYBRID_FEYNMAN = HYBRID_FEYNMAN_CONFIG

    MODULAR_1_GPU = MODULAR_1_GPU_CONFIG
    MODULAR_2_GPU = MODULAR_2_GPU_CONFIG
    MODULAR_3_GPU = MODULAR_3_GPU_CONFIG
    MODULAR_4_GPU = MODULAR_4_GPU_CONFIG
    MODULAR_5_GPU = MODULAR_5_GPU_CONFIG


cfg = Configs()
