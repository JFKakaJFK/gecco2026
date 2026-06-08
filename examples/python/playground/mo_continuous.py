import pathlib

from pygom import *
from src.config import Config, c
from src.plots import plot_scalability
from src.postprocessing import load_results
from src.run import run_all

REPEATS = 30


budget = c.Budget(max_evaluations=int(1e6))


def problems():
    # display name, instance info, actual instance
    yield (
        "ZDT1",
        30,
        c.BenchmarkInstance(
            c.ZDT1(),
            init=c.RandomInit(),
            continuous_init_lower_bound=0.0,
            continuous_init_upper_bound=1.0,
            continuous_lower_bound=0.0,
            continuous_upper_bound=1.0,
        ),
    )


def methods():
    # display name, actual method
    yield (
        "MO-RV-GOMEA",
        c.MoRvGOMEA(),
    )
    yield (
        "Mixed",
        c.MixedGOMEA(
            ims_options=c.IMSOptions(
                initial_population_size=40, subgeneration_factor=8
            ),
            population_options=c.PopulationOptions(forced_improvements=False),
        ),
    )


if __name__ == "__main__":
    domain = "mo_continuous"
    output_directory = f"results/{domain}_raw"
    parquet_dir = f"results/{domain}_processed"

    run_all(
        output_directory=output_directory,
        problems=problems,
        methods=methods,
        budget=budget,
        num_repeats=REPEATS,
        # clean=True,
        # limit=1,
        # max_workers=1,
    )

    preprocess = True
    # preprocess = False

    # TODO - how to establish success??

    # plot_dir = pathlib.Path("plots")
    # plot_dir.mkdir(parents=True, exist_ok=True)

    # with load_results(
    #     output_directory,
    #     parquet_dir=parquet_dir,
    #     preprocess=preprocess,
    # ) as conn:
    #     fig = plot_scalability(conn)  # , log_x=False, log_y=False)
    #     for fmt in ["pdf", "png"]:
    #         fig.savefig(
    #             plot_dir / f"{domain}_scalability.{fmt}",
    #             dpi=600,
    #             # transparent=True,
    #             bbox_inches="tight",
    #         )
