from collections.abc import Generator
from datetime import datetime
from pathlib import Path

import numpy as np
from pygom import KernelVersion

from src.config import ExperimentConfig, cfg
from src.db import create_db
from src.plots import plot
from src.run import run_cpu_tasks, run_gpu_tasks
from src.task import Task, problems

LINE_UP = "\033[1A"
LINE_CLEAR = "\x1b[2K"


def print_status(msg: str, dry_run: bool, n: int = 0):
    if dry_run:
        return

    if n > 0:
        for _ in range(n):
            print(LINE_UP, end=LINE_CLEAR)

    print(msg)


def cpu_jobs(
    problems: Generator[Task],
) -> Generator[Task]:
    yield from problems


def gpu_jobs(
    problems: Generator[Task], kernels: tuple[KernelVersion, ...]
) -> Generator[Task]:
    for task in problems:
        for kernel in kernels:
            new_task: Task = dict(task)
            new_task["accelerated"] = True
            new_task["kernel"] = kernel

            yield new_task


def all_jobs(output_dir: Path, cfg: ExperimentConfig, dry_run: bool = False):
    # Run CPU jobs
    if cfg.cpu.enabled:
        print_status("Starting CPU tasks...", dry_run)

        run_cpu_tasks(
            output_dir,
            cpu_jobs(
                problems(np.random.default_rng(seed=42), output_dir, cfg, dry_run)
            ),
            dry_run=dry_run,
        )

        print_status("Finished with CPU tasks", dry_run, n=1)

    print_status("Starting GPU tasks...", dry_run)

    # Run GPU jobs
    run_gpu_tasks(
        output_dir,
        gpu_jobs(
            problems(
                np.random.default_rng(seed=42),
                output_dir,
                cfg,
                dry_run,
            ),
            cfg.gpu.kernels,
        ),
        dry_run=dry_run,
    )

    print_status("Finished with GPU tasks", dry_run, n=2)


def print_experiment_header(name: str) -> None:
    title = f"### {name.replace('_', ' ').title()} Experiment ###"
    line = "#" * len(title)

    print(line)
    print(title)
    print(line)


def run_experiment(dir: Path, config: ExperimentConfig, dry_run: bool = False):
    # if not dry_run:
    print_experiment_header(config.name)

    output_directory = dir / config.name

    all_jobs(output_directory, config, dry_run)

    if not dry_run:
        print_status("Starting database creation...", dry_run)
        create_db(output_directory)
        print_status("Finished with database creation", dry_run, n=1)

        print_status("Starting plot creation...", dry_run)
        plot(output_directory)
        print_status("Finished with plot creation", dry_run, n=1)


def main():
    run_date = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    # run_date = "2026-01-13_11:38:49"
    output_directory = Path("results") / run_date

    # run_experiment(output_directory, cfg.TEST, dry_run=False)
    # run_experiment(output_directory, cfg., dry_run=True)
    run_experiment(output_directory, cfg.KERNEL_SWEEP_POPULATION, dry_run=False)


if __name__ == "__main__":
    main()
