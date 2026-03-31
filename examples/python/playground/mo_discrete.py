import pathlib

from pygom import *

from src.config import Config, c
from src.plots import plot_scalability
from src.postprocessing import load_results
from src.run import run_all

REPEATS = 30
DIMS = [
    10,
    20,
    40,
    80,
    # 160,
    # , 320
]

budget = c.Budget(max_evaluations=int(1e6))


def problems():
    for d in DIMS:
        # display name, instance info, actual instance
        yield (
            "OneMax/ZeroMax",
            d,
            c.BenchmarkInstance(
                [c.OneMax(d), c.ZeroMax(d)],
                target_archive_size=d + 1,
                init=c.RandomInit(),
            ),
        )
        yield (
            "DTrap5/Inverse",
            d,
            c.BenchmarkInstance(
                [
                    c.Repeat(c.DeceptiveTrap(5), d // 5),
                    c.Inverted(c.Repeat(c.DeceptiveTrap(5), d // 5)),
                ],
                target_archive_size=d // 5
                + 1,  # num_blocks + 1 solutions are non-dominated...
                init=c.RandomInit(),
            ),
        )
        yield (
            "LOTZ",
            d,
            c.BenchmarkInstance(
                [c.LeadingOnes(d), c.TrailingZeros(d)],
                target_archive_size=d + 1,
                init=c.RandomInit(),
            ),
        )


def methods():
    # display name, actual method
    yield '"MO Binary"', c.MOBinaryGOMEA(target_archive_size=2 * DIMS[-1])
    yield (
        '"MO Binary (weak)"',
        c.MOBinaryGOMEA(mutation="weak", target_archive_size=2 * DIMS[-1]),
    )
    yield (
        '"MO Binary (strong)"',
        c.MOBinaryGOMEA(mutation="strong", target_archive_size=2 * DIMS[-1]),
    )

    for metric in ["mi", "nmi"]:
        for mutation in ["", "weak", "strong"]:
            discrete_mutation_probability = 0.0
            if mutation == "weak":
                #     discrete_mutation_probability = 1.0 / problem.num_discrete()
                continue
            elif mutation == "strong":
                discrete_mutation_probability = None
            yield (
                f'"{f"{metric.upper()} LT {mutation}".strip()}"',
                c.MixedGOMEA(
                    ims_options=c.IMSOptions(
                        initial_population_size=8, subgeneration_factor=2
                    ),
                    population_options=c.PopulationOptions(
                        discrete_mutation_probability=discrete_mutation_probability
                    ),
                    discrete_model=c.LinkageTreeFOS(metric=metric),
                ),
            )


if __name__ == "__main__":
    domain = "mo_binary"
    output_directory = f"results/{domain}_raw"
    parquet_dir = f"results/{domain}_processed"

    run_all(
        output_directory=output_directory,
        problems=problems,
        methods=methods,
        budget=budget,
        num_repeats=REPEATS,
        clean=True,
        # limit=1,
        # max_workers=1,
    )

    preprocess = True
    # preprocess = False

    plot_dir = pathlib.Path("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    with load_results(
        output_directory,
        parquet_dir=parquet_dir,
        preprocess=preprocess,
    ) as conn:
        fig = plot_scalability(conn)  # , log_x=False, log_y=False)
        for fmt in ["pdf", "png"]:
            fig.savefig(
                plot_dir / f"{domain}_scalability.{fmt}",
                dpi=600,
                # transparent=True,
                bbox_inches="tight",
            )
