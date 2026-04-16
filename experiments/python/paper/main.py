import pathlib

from src.db import (
    create_db_experiment_2,
    create_db_experiment_3,
    create_db_experiment_6,
)
from src.plot import plot_experiment_2, plot_experiment_3, plot_experiment_6

if __name__ == "__main__":
    # Experiment 2
    create_db_experiment_2(pathlib.Path("results/2_algo_shootout"))
    plot_experiment_2(pathlib.Path("results/2_algo_shootout"), var="mse")
    plot_experiment_2(pathlib.Path("results/2_algo_shootout"), var="evaluations")

    # Experiment 3
    create_db_experiment_3(pathlib.Path("results/3_gravity"))
    plot_experiment_3(pathlib.Path("results/3_gravity"))

    # Experiment 6
    create_db_experiment_6(pathlib.Path("results/6_restart/cpu_results"))
    plot_experiment_6(pathlib.Path("results/6_restart/cpu_results"), var="mse")
    plot_experiment_6(pathlib.Path("results/6_restart/cpu_results"), var="evaluations")

    create_db_experiment_6(pathlib.Path("results/6_restart/gpu_results"))
    plot_experiment_6(pathlib.Path("results/6_restart/gpu_results"), var="mse")
    plot_experiment_6(pathlib.Path("results/6_restart/gpu_results"), var="evaluations")
