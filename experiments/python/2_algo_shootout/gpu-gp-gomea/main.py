import csv
import os
from collections.abc import Callable
from datetime import datetime
from itertools import chain, product
from pathlib import Path

from pygom import KernelVersion
from shared.experiment_config import ExperimentConfig, cfg
from shared.problem import Problem, format_problem, generate_problems
from shared.task import (
    TaskTransform,
    cpu_transform,
    gpu_transform,
    override_tasks,
    task_factory,
)

from src.db import create_db
from src.run import run_cpu_tasks, run_gpu_tasks, run_tasks

MIN_POPULATION = 8
MAX_POPULATION = 65536


def hardware_helper(hardware: str | KernelVersion):
    if hardware == "cpu":
        return run_cpu_tasks, cpu_transform()
    else:
        return run_gpu_tasks, gpu_transform(hardware)


def grid_search_helper(
    problem: Problem,
    config: ExperimentConfig,
    runner: Callable[..., float | None],
    hardware_transform: TaskTransform,
    writer: csv.DictWriter[str],
    output_directory: Path,
    required_rate: float,
    kernel_str: str,
):
    search_space: dict[str, list[int]] = config.search_space

    factory = task_factory(problem, config, output_directory)

    for values in product(*search_space.values()):
        tasks = hardware_transform(factory())

        overrides: dict[str, int] = dict(zip(search_space.keys(), values, strict=True))

        rate: float = run_tasks(
            tasks,
            override_tasks(overrides),
            runner,
            output_directory,
            required_rate=required_rate,
        )

        if rate is not None and rate >= required_rate:
            writer.writerow(
                {**problem, **overrides, "kernel": kernel_str, "rate": rate}
            )

            return


def grid_search(config: ExperimentConfig, directory: Path):
    print(f"Starting with {config.name}")

    output_directory = directory / config.name

    os.makedirs(output_directory, exist_ok=True)

    search_space: dict[str : list[int]] = config.search_space
    required_rate: float | None = config.required_rate

    hardware: tuple[KernelVersion | str, ...] = (
        config.gpu.kernels if config.gpu.enabled else ()
    ) + (("cpu",) if config.cpu.enabled else ())

    # Check if search space is not empty
    if not search_space:
        raise ValueError("Search space cannot be empty for grid search")

    # Check if required_rate is defined
    if required_rate is None:
        raise ValueError("Required rate cannot be None for grid search")

    results_path = output_directory / "grid_search_results.csv"
    fieldnames: list[str] = list(
        dict.fromkeys(
            chain(
                Problem.__annotations__,
                search_space,
                ["kernel", "rate"],
            )
        )
    )

    with open(results_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Loop over problems
        for problem in generate_problems(config):
            print(format_problem(problem))

            # Loop over hardware
            for h in hardware:
                runner, transform = hardware_helper(h)

                kernel_str: str = (
                    str(h).replace("KernelVersion.", "") if h != "cpu" else "cpu"
                )

                # Perform grid search for problem
                grid_search_helper(
                    problem,
                    config,
                    runner,
                    transform,
                    writer,
                    output_directory,
                    required_rate,
                    kernel_str,
                )


def grid_execution(config: ExperimentConfig, directory: Path, dry_run: bool = False):
    output_directory = directory / config.name

    if not dry_run:
        os.makedirs(output_directory, exist_ok=True)

    cpu_batches = []
    gpu_batches = []

    # Loop over problems
    for problem in generate_problems(config):
        factory = task_factory(problem, config, output_directory, dry_run=dry_run)

        if config.cpu.enabled:
            cpu_batches.append(cpu_transform()(factory()))

        if config.gpu.enabled:
            for kernel in config.gpu.kernels:
                gpu_batches.append(gpu_transform(kernel)(factory()))

    run_tasks(
        chain.from_iterable(cpu_batches),
        None,
        run_cpu_tasks,
        output_directory,
        dry_run=dry_run,
    )

    run_tasks(
        chain.from_iterable(gpu_batches),
        None,
        run_gpu_tasks,
        output_directory,
        dry_run=dry_run,
    )

    if not dry_run:
        create_db(output_directory)


def main():
    run_date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    # run_date = "2026-02-01_17:02:09"
    output_directory = Path("results") / run_date

    # GPU Experiments
    grid_execution(cfg.DAILY_DEMAND_GPU, output_directory, dry_run=True)
    grid_execution(cfg.AUTO_MPG_GPU, output_directory, dry_run=True)
    grid_execution(cfg.CALIFORNIA_GPU, output_directory, dry_run=True)
    grid_execution(cfg.FEYNMAN_GPU, output_directory, dry_run=True)


if __name__ == "__main__":
    main()
