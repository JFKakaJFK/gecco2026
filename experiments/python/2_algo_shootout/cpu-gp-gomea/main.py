import os
from datetime import datetime
from itertools import chain
from pathlib import Path

from shared.experiment_config import ExperimentConfig, cfg
from shared.problem import generate_problems
from shared.task import cpu_transform, task_factory

from src.db import create_db
from src.run import run_cpu_tasks, run_tasks


def grid_execution(config: ExperimentConfig, directory: Path, dry_run: bool = False):
    output_directory = directory / config.name

    if not dry_run:
        os.makedirs(output_directory, exist_ok=True)

    cpu_batches = []

    # Loop over problems
    for problem in generate_problems(config):
        factory = task_factory(problem, config, output_directory, dry_run=dry_run)

        if config.cpu.enabled:
            cpu_batches.append(cpu_transform()(factory()))

    run_tasks(
        chain.from_iterable(cpu_batches),
        None,
        run_cpu_tasks,
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
    grid_execution(cfg.DAILY_DEMAND_CPU, output_directory, dry_run=True)
    grid_execution(cfg.AUTO_MPG_CPU, output_directory, dry_run=True)
    grid_execution(cfg.CALIFORNIA_CPU, output_directory, dry_run=True)
    grid_execution(cfg.FEYNMAN_CPU, output_directory, dry_run=True)


if __name__ == "__main__":
    main()
