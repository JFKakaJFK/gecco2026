import pathlib

from src.db import (
    create_db_experiment_2,
    create_db_experiment_3,
    create_db_experiment_4,
    create_db_experiment_5,
    create_db_experiment_6,
)
from src.plot import (
    plot_experiment_2,
    plot_experiment_2_gomea,
    plot_experiment_3,
    plot_experiment_4_feynman,
    plot_experiment_4_modular,
    plot_experiment_5,
    plot_experiment_6,
)
from src.validate import (
    validate_experiment_2,
    validate_experiment_3,
    validate_experiment_4,
    validate_experiment_5,
)

if __name__ == "__main__":
    # Experiment 2
    # validate_experiment_2(pathlib.Path("results/2_algo_shootout"))
    # create_db_experiment_2(pathlib.Path("results/2_algo_shootout"))
    # for var in ("mse", "mse_val", "evaluations"):
    #     plot_experiment_2(pathlib.Path("results/2_algo_shootout"), var=var)
    #     plot_experiment_2(
    #         pathlib.Path("results/2_algo_shootout"), var=var, exclude=["Operon (CPU)"]
    #     )
    #     plot_experiment_2_gomea(pathlib.Path("results/2_algo_shootout"), var=var)

    # Experiment 3
    # validate_experiment_3(pathlib.Path("results/3_gravity"))
    # create_db_experiment_3(pathlib.Path("results/3_gravity"))
    # plot_experiment_3(pathlib.Path("results/3_gravity"))
    # plot_experiment_3(pathlib.Path("results/3_gravity"), nmse_col="nmse_val")

    # Experiment 4
    # validate_experiment_4(pathlib.Path("results/4_modular"))
    # create_db_experiment_4(pathlib.Path("results/4_modular"))
    for var in ("nmse", "nmse_val", "evaluations"):
        # plot_experiment_4_feynman(pathlib.Path("results/4_modular"), var=var)
        plot_experiment_4_modular(pathlib.Path("results/4_modular"), var=var)

    # Experiment 5
    # validate_experiment_5(pathlib.Path("results/5_hybrid_block"))
    # create_db_experiment_5(pathlib.Path("results/5_hybrid_block"))
    # plot_experiment_5(pathlib.Path("results/5_hybrid_block"), var="mse")
    # plot_experiment_5(pathlib.Path("results/5_hybrid_block"), var="mse_val")
    # plot_experiment_5(pathlib.Path("results/5_hybrid_block"), var="evaluations")

    # # Experiment 6
    # create_db_experiment_6(pathlib.Path("results/6_restart/cpu_results"))
    # plot_experiment_6(pathlib.Path("results/6_restart/cpu_results"), var="mse")
    # plot_experiment_6(pathlib.Path("results/6_restart/cpu_results"), var="evaluations")

    # create_db_experiment_6(pathlib.Path("results/6_restart/gpu_results"))
    # plot_experiment_6(pathlib.Path("results/6_restart/gpu_results"), var="mse")
    # plot_experiment_6(pathlib.Path("results/6_restart/gpu_results"), var="evaluations")
