import pathlib

from pygom import *
from src.config import Config, c
from src.plots import plot_scalability
from src.postprocessing import load_results
from src.run import run_all

REPEATS = 30

budget = c.Budget(max_evaluations=int(1e6))


def problems():
    for d in [
        10,
        20,
        40,
        80,
        # 160,
        # , 320
    ]:
        # display name, instance info, actual instance
        yield (
            "OneMax",
            d,
            c.BenchmarkInstance(
                c.OneMax(d),
                target=[float(d)],
                init=c.RandomInit(),
            ),
        )
        yield (
            "DTrap5",
            d,
            c.BenchmarkInstance(
                c.Repeat(c.DeceptiveTrap(5), d // 5),
                target=[float(d)],
                init=c.RandomInit(),
            ),
        )
        yield (
            "LeadingOnes",
            d,
            c.BenchmarkInstance(
                c.Masked(c.LeadingOnes(d)),
                target=[float(d)],
                init=c.RandomInit(),
            ),
        )
        yield (
            "LeadingOnes IA",
            d,
            c.BenchmarkInstance(
                c.LeadingOnes(d),
                target=[float(d)],
                init=c.RandomInit(),
            ),
        )


def methods():
    # display name, actual method
    for repr in ["aos", "soa_cols", "soa_rows"]:
        for intron_strategy in ["none", "any_active"]:
            yield (
                f"{repr} LT {'IA' if intron_strategy != 'none' else ''}".strip(),
                c.MixedGOMEA(
                    discrete_model=c.LinkageTreeFOS(intron_strategy=intron_strategy),
                    repr=repr,
                ),
            )


if __name__ == "__main__":
    domain = "repr"
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
        print(conn.sql("SHOW TABLE results"))

        fig = plot_scalability(conn)
        for fmt in ["pdf", "png"]:
            fig.savefig(
                plot_dir / f"{domain}_scalability.{fmt}",
                dpi=600,
                # transparent=True,
                bbox_inches="tight",
            )
