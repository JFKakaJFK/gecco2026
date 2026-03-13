from dataclasses import dataclass, field

from src.dataset_config import DATASETS, DatasetConfig


@dataclass(frozen=True)
class TemplateConfig:
    branching_factor: int = 2
    depth: int = 4
    max_nodes: int = 31


@dataclass(frozen=True)
class OperatorSet:
    name: str
    operators: str


OPERATOR_SETS: dict[str, str] = {
    "arith": "+,-,*,/",
    "trig": "+,-,*,/,sin,cos",
    "square": "+,-,*,/,square,sqrt",
    "exp": "+,-,*,/,exp,log",
    "paper": "+,-,*,/,sin,cos,square,sqrt,exp,log,pow",
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
    max_duration: int

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

DAILY_DEMAND_CONFIG = ExperimentConfig(
    # Problem Space
    name="daily_demand",
    datasets=[DATASETS["daily_demand"]],
    population_sizes=[2**i for i in range(7, 15)],  # 128 - 16384
    num_observations=[60],
    num_features=None,
    templates=[TemplateConfig(2, 4, 31)],
    operator_sets=["paper"],
    # Execution Parameters
    use_target=False,
    num_folds=30,
    num_iterations=1,
    test_size=0,
    max_duration=600,
)

AUTO_MPG_CONFIG = ExperimentConfig(
    # Problem Space
    name="auto_mpg",
    datasets=[DATASETS["auto_mpg"]],
    population_sizes=[2**i for i in range(7, 17)],  # 128 - 65536
    num_observations=[398],
    num_features=None,
    templates=[TemplateConfig(2, 4, 31)],
    operator_sets=["paper"],
    # Execution Parameters
    use_target=False,
    num_folds=30,
    num_iterations=1,
    test_size=0,
    max_duration=600,
)

CALIFORNIA_CONFIG = ExperimentConfig(
    # Problem Space
    name="california_housing",
    datasets=[DATASETS["california"]],
    population_sizes=[2**i for i in range(7, 15)],  # 128 - 16384
    num_observations=[20640],
    num_features=None,
    templates=[TemplateConfig(2, 4, 31)],
    operator_sets=["paper"],
    # Execution Parameters
    use_target=False,
    num_folds=30,
    num_iterations=1,
    test_size=0,
    max_duration=600,
)

FEYNMAN_CONFIG = ExperimentConfig(
    # Problem Space
    name="feynman",
    datasets=[DATASETS["feynman_I_9_18"]],
    population_sizes=[2**i for i in range(7, 15)],  # 128 - 16384
    num_observations=[100_000],
    num_features=None,
    templates=[TemplateConfig(2, 4, 31)],
    operator_sets=["paper"],
    # Execution Parameters
    use_target=False,
    num_folds=30,
    num_iterations=1,
    test_size=0,
    max_duration=600,
)


class Configs:
    # Paper experiments
    DAILY_DEMAND = DAILY_DEMAND_CONFIG
    AUTO_MPG = AUTO_MPG_CONFIG
    CALIFORNIA = CALIFORNIA_CONFIG
    FEYNMAN = FEYNMAN_CONFIG


cfg = Configs()
