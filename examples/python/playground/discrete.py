import pathlib

from pygom import *

from src.config import Config, c
from src.plots import plot_scalability
from src.postprocessing import load_results
from src.run import run_all

REPEATS = 50

budget = c.Budget(max_evaluations=int(1e5))


def problems():
    for d in [
        5,
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
    initial_population_size = 2
    max_num_populations = 25
    restart_stale_populations = True  # by default, we want this only for "relevant" populations that still contribute new improvements - so only the largest population in single-objective population, but due to forced improvements any population in a multi-objective setting.
    # display name, actual method
    yield (
        '"Library (MI, LT)"',
        c.DiscreteGOMEA(
            base_population_size=initial_population_size,
            max_number_of_populations=max_num_populations,
        ),
    )
    # yield (
    #     '"Library (MI, U)"',
    #     c.DiscreteGOMEA(
    #         base_population_size=initial_population_size,
    #         max_number_of_populations=max_num_populations,
    #         linkage_model="Univariate",
    #     ),
    # )
    yield (
        '"Library (NMI, LT)"',
        c.DiscreteGOMEA(
            similarity_metric="NMI",
            base_population_size=initial_population_size,
            max_number_of_populations=max_num_populations,
        ),
    )
    # yield (
    #     '"Library (GIGA, U)"',
    #     c.DiscreteGOMEA(linkage_model="Univariate", gene_invariant=True),
    # )

    for metric in ["mi", "nmi"]:
        for fos in [
            "LT",  #  "U"
        ]:
            if fos == "U":
                yield (
                    f"{metric.upper()} (U)",
                    c.MixedGOMEA(
                        discrete_model=c.UnivariateFOS(),
                        ims_options=c.IMSOptions(
                            initial_population_size=initial_population_size,
                            max_num_populations=max_num_populations,
                            restart_stale_populations=restart_stale_populations,
                        ),
                    ),
                )

            else:
                for intron_strategy in ["none", "any_active"]:
                    yield (
                        f"{metric.upper()} LT {'IA' if intron_strategy != 'none' else ''}".strip(),
                        c.MixedGOMEA(
                            discrete_model=c.LinkageTreeFOS(
                                metric=metric,
                                intron_strategy=intron_strategy,
                                filter_root=True,
                            ),
                            ims_options=c.IMSOptions(
                                initial_population_size=initial_population_size,
                                max_num_populations=max_num_populations,
                                restart_stale_populations=restart_stale_populations,
                            ),
                        ),
                    )


if __name__ == "__main__":
    domain = "discrete"
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
        fig = plot_scalability(
            conn,
            # methods=["Library (MI, LT)", "Library (MI, U)", "MI LT", "MI (U)"],
        )
        for fmt in ["pdf", "png"]:
            fig.savefig(
                plot_dir / f"{domain}_scalability.{fmt}",
                dpi=600,
                # transparent=True,
                bbox_inches="tight",
            )
