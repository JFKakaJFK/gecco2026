import os
from datetime import datetime
from itertools import chain
from pathlib import Path

from shared.experiment_config import ExperimentConfig, cfg
from shared.problem import generate_problems
from shared.task import task_factory

from src.run import run_cpu_tasks


def grid_execution(config: ExperimentConfig, directory: Path, dry_run: bool = False):
    output_directory = directory / config.name

    if not dry_run:
        os.makedirs(output_directory, exist_ok=True)

    batches = []

    # Loop over problems
    for problem in generate_problems(config):
        factory = task_factory(problem, config, output_directory, dry_run=dry_run)

        batches.append(factory())

    run_cpu_tasks(
        chain.from_iterable(batches),
        output_directory,
        dry_run=dry_run,
    )


def main():
    run_date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    # run_date = "2026-02-01_17:02:09"
    output_directory = Path("results") / run_date

    # CPU Experiments
    grid_execution(cfg.TEST_CPU, output_directory)

    # grid_execution(cfg.DAILY_DEMAND_CPU, output_directory, dry_run=True)
    # grid_execution(cfg.AUTO_MPG, output_directory, dry_run=False)
    # grid_execution(cfg.CALIFORNIA, output_directory, dry_run=False)
    # grid_execution(cfg.FEYNMAN, output_directory, dry_run=False)


if __name__ == "__main__":
    main()
