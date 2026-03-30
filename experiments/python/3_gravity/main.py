import os
from datetime import datetime
from itertools import chain
from pathlib import Path

from shared.experiment_config import ExperimentConfig, cfg
from shared.problem import generate_problems
from shared.task import (
    gpu_transform,
    task_factory,
)

from src.db import create_db
from src.run import run_gpu_tasks, run_tasks


def grid_execution(config: ExperimentConfig, directory: Path, dry_run: bool = False):
    output_directory = directory / config.name

    if not dry_run:
        os.makedirs(output_directory, exist_ok=True)

    batches = []

    # Loop over problems
    for problem in generate_problems(config):
        factory = task_factory(problem, config, output_directory, dry_run=dry_run)

        for kernel in config.gpu.kernels:
            batches.append(gpu_transform(kernel)(factory()))

    run_tasks(
        chain.from_iterable(batches),
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

    # grid_execution(cfg.FEYNMAN_MILLION, output_directory, dry_run=True)

    grid_execution(cfg.ADDITION, output_directory, dry_run=True)
    grid_execution(cfg.DIVISION, output_directory, dry_run=True)
    grid_execution(cfg.SUBTRACTION, output_directory, dry_run=True)
    grid_execution(cfg.MULTIPLICATION, output_directory, dry_run=True)
    grid_execution(cfg.SQUARE, output_directory, dry_run=True)


if __name__ == "__main__":
    main()
