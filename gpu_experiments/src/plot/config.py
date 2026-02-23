from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal


@dataclass(frozen=True)
class ConvergencePlotConfig:
    name: str
    kernels: list[str] | None = None

    row_modifier_query: str | None = None
    row_modifier_labels: list[str] | None = None
    row_sort_key: Callable[[tuple], Any] | None = None
    col_modifier_query: str | None = None
    col_modifier_labels: list[str] | None = None
    col_sort_key: Callable[[tuple], Any] | None = None

    unit_query: str = "run"

    y_var: str = "objectives[1]"
    y_agg: Literal["MIN", "MAX"] = "MIN"
    y_label: str = "Objective Value"

    metrics: Sequence[str] = ("evaluations", "total_time_seconds", "generation")
    metric_labels: Sequence[str] = ("Evaluations", "Time [s]", "Generations")

    ylog: bool = False
    nsamples: int = 25

    ymin: float | str | None = None
    ymax: float | str | None = None


R2_ALL = ConvergencePlotConfig(
    name="r2_convergence",
    row_modifier_query="[template_depth::STRING,operator_set::STRING]",
    row_modifier_labels=["Depth", "Operators"],
    col_modifier_query="[num_observations::STRING, population_size::STRING]",
    col_modifier_labels=["#Observations", "#Individiuals"],
    col_sort_key=lambda m: (int(m[0]), int(m[1])),
    unit_query="format('{}.{}', fold, run)",
    y_var="1 - objectives[1]::DOUBLE / var_y::DOUBLE",
    y_agg="MAX",
    y_label="$R^2$ Train",
    nsamples=100,
    ymin="auto",  # cuts of the lower 2.5% of data points
)

# CPU vs Baseline
R2_CPU_BASELINE = replace(
    R2_ALL, name="r2_convergence_cpu_baseline", kernels=["cpu", "baseline"]
)

# CPU vs Baseline vs Baseline Improvements
R2_CPU_BASELINE_IMPROV = replace(
    R2_ALL,
    name="r2_convergence_cpu_baseline_improvements",
    kernels=["cpu", "baseline", "restrict", "shared_memory"],
)

# Baseline vs Baseline Improvements
R2_BASELINE_IMPROV = replace(
    R2_ALL,
    name="r2_convergence_baseline_improvements",
    kernels=["baseline", "restrict", "shared_memory"],
)

# Baseline vs Block Reduce
R2_BASELINE_REDUCE = replace(
    R2_ALL,
    name="r2_convergence_baseline_reduce",
    kernels=["shared_memory", "block_reduce"],
)

# Baseline vs Block Reduce vs Single Kernel
R2_BASELINE_REDUCE_SINGLE = replace(
    R2_ALL,
    name="r2_convergence_baseline_reduce_single",
    kernels=["shared_memory", "block_reduce", "single_kernel"],
)

# Single Kernel vs Single Kernel Improvements
R2_SINGLE_IMPROV = replace(
    R2_ALL,
    name="r2_convergence_single_improvements",
    kernels=["single_kernel", "single_kernel_fmaf", "single_kernel_inplace"],
)

SPEED_ALL = ConvergencePlotConfig(
    name="speed_convergence",
    row_modifier_query="[template_depth::STRING,operator_set::STRING]",
    row_modifier_labels=["Depth", "Operators"],
    col_modifier_query="[num_observations::STRING, population_size::STRING]",
    col_modifier_labels=["#Observations", "#Individiuals"],
    col_sort_key=lambda m: (int(m[0]), int(m[1])),
    unit_query="format('{}.{}', fold, run)",
    y_var="MAX(evaluations)::DOUBLE / MAX(eval_time_seconds)::DOUBLE",
    y_agg="",
    y_label="Evaluations / Second",
    ylog=False,
    nsamples=100,
)

# CPU vs Baseline
SPEED_CPU_BASELINE = replace(
    SPEED_ALL, name="speed_convergence_cpu_baseline", kernels=["cpu", "baseline"]
)

# CPU vs Baseline vs Baseline Improvements
SPEED_CPU_BASELINE_IMPROV = replace(
    SPEED_ALL,
    name="speed_convergence_cpu_baseline_improvements",
    kernels=["cpu", "baseline", "restrict", "shared_memory"],
)

# Baseline vs Baseline Improvements
SPEED_BASELINE_IMPROV = replace(
    SPEED_ALL,
    name="speed_convergence_baseline_improvements",
    kernels=["baseline", "restrict", "shared_memory"],
)

# Baseline vs Block Reduce
SPEED_BASELINE_REDUCE = replace(
    SPEED_ALL,
    name="speed_convergence_baseline_reduce",
    kernels=["shared_memory", "block_reduce"],
)

# Baseline vs Block Reduce vs Single Kernel
SPEED_BASELINE_REDUCE_SINGLE = replace(
    SPEED_ALL,
    name="speed_convergence_baseline_reduce_single",
    kernels=["shared_memory", "block_reduce", "single_kernel"],
)

# Single Kernel vs Single Kernel Improvements
SPEED_SINGLE_IMPROV = replace(
    SPEED_ALL,
    name="speed_convergence_single_improvements",
    kernels=["single_kernel", "single_kernel_fmaf", "single_kernel_inplace"],
)


CONFIGS: dict[str, ConvergencePlotConfig] = {
    # R2 Convergence Configs
    "r2_all": R2_ALL,
    "r2_cpu_baseline": R2_CPU_BASELINE,
    "r2_cpu_baseline_improv": R2_CPU_BASELINE_IMPROV,
    "r2_baseline_improv": R2_BASELINE_IMPROV,
    "r2_baseline_reduce": R2_BASELINE_REDUCE,
    "r2_baseline_reduce_single": R2_BASELINE_REDUCE_SINGLE,
    "r2_single_improv": R2_SINGLE_IMPROV,
    # Evaluation Speed Convergence Configs
    "speed_all": SPEED_ALL,
    "speed_cpu_baseline": SPEED_CPU_BASELINE,
    "speed_cpu_baseline_improv": SPEED_CPU_BASELINE_IMPROV,
    "speed_baseline_improv": SPEED_BASELINE_IMPROV,
    "speed_baseline_reduce": SPEED_BASELINE_REDUCE,
    "speed_baseline_reduce_single": SPEED_BASELINE_REDUCE_SINGLE,
    "speed_single_improv": SPEED_SINGLE_IMPROV,
}
