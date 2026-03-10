import multiprocessing as mp
import os
from datetime import datetime
from itertools import chain
from pathlib import Path

from src.experiment_config import ExperimentConfig, cfg
from src.problem import generate_problems
from src.run import run_gpu_tasks
from src.task import task_factory


def grid_execution(config: ExperimentConfig, directory: Path, dry_run: bool = False):
    output_directory = directory / config.name

    if not dry_run:
        os.makedirs(output_directory, exist_ok=True)

    batches = []

    # Loop over problems
    for problem in generate_problems(config):
        factory = task_factory(problem, config, output_directory, dry_run=dry_run)

        batches.append(factory())

    run_gpu_tasks(
        chain.from_iterable(batches), output_directory, dry_run=dry_run, max_workers=1
    )


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    run_date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    # run_date = "2026-02-01_17:02:09"
    output_directory = Path("results") / run_date

    # CPU Experiments
    grid_execution(cfg.DAILY_DEMAND, output_directory, dry_run=True)
    grid_execution(cfg.AUTO_MPG, output_directory, dry_run=True)
    grid_execution(cfg.CALIFORNIA, output_directory, dry_run=True)
    grid_execution(cfg.FEYNMAN, output_directory, dry_run=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    main()
