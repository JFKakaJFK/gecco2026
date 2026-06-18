import pathlib

from pygom import *
from src.config import Config, c
from src.plots import plot_scalability
from src.postprocessing import load_results
from src.run import run_all

REPEATS = 30

budget = c.Budget(max_evaluations=int(1e6))

VTR = 1e-10

# http://hydra.nat.uni-magdeburg.de/packing/csq/csq.html#Download
# let tbody = document.querySelector(".results > table:nth-child(1) > tbody:nth-child(1)")
# JSON.stringify(Array.from(tbody.children).map((tr) => tr.children[2].innerHTML).slice(2, 50).map((v) => parseFloat(v)))
cias_optima = [
    None,  # 0 circles
    None,  # 1 circle
    1.4142135623730951,
    1.035276180410083,
    1,
    0.7071067811865476,
    0.6009252125773316,
    0.5358983848622454,
    0.5176380902050415,
    0.5,
    0.42127954398390344,
    0.3982073102368442,
    0.38873012632302,
    0.3660960076964251,
    0.3489152603740189,
    0.34108137740210887,
    0.3333333333333333,
    0.30615398530033294,
    0.3004626062886658,
    0.28954199199498165,
    0.28661165235168157,
    0.2718122553593071,
    0.26795840155071676,
    0.25881904510252074,
    0.2543330950302498,
    0.25,
    0.23873475724121662,
    0.2358495283014151,
    0.23053549364266748,
    0.22688290074420908,
    0.22450296453108828,
    0.21754729161912445,
    0.21317456258987652,
    0.21132838414326313,
    0.20560464675956822,
    0.20276360086322706,
    0.2,
    0.196429184629569,
    0.1953423041269059,
    0.19436506316151,
    0.18817552201831544,
    0.18609951184812457,
    0.18427707211709893,
    0.18019113545742596,
    0.1786392456712009,
    0.17571631417558983,
    0.1744593608724132,
    0.17127056382298722,
    0.1694054293702881,
    0.1673860768683254,
]


def problems():
    init_lb, init_ub = 100, 125

    for d in [  #
        2,
        5,
        10,
        20,
        # 40,
        # 80,
    ]:
        # display name, instance info, actual instance
        yield (
            "Sphere",
            d,
            c.BenchmarkInstance(
                c.Sphere(d),
                target_objectives=[VTR],
                continuous_init_lower_bound=init_lb,
                continuous_init_upper_bound=init_ub,
            ),
        )
        if d // 5 >= 2:
            yield (
                "SoREB",
                d,
                c.BenchmarkInstance(
                    c.Repeat(c.Rotated(c.Ellipsoid(5), -45.0), d // 5),
                    # c.Repeat(c.Rotated(c.Ellipsoid(2), -45.0), d // 2),
                    target_objectives=[VTR],
                    continuous_init_lower_bound=init_lb,
                    continuous_init_upper_bound=init_ub,
                ),
            )

    for d in [2, 4, 8]:
        yield (
            "Ellipsoid",
            d,
            c.BenchmarkInstance(
                c.Ellipsoid(d),
                target_objectives=[VTR],
                continuous_init_lower_bound=init_lb,
                continuous_init_upper_bound=init_ub,
            ),
        )
        yield (
            "Rot. Ellipsoid",
            d,
            c.BenchmarkInstance(
                c.Rotated(c.Ellipsoid(d), -45.0),
                target_objectives=[VTR],
                continuous_init_lower_bound=init_lb,
                continuous_init_upper_bound=init_ub,
            ),
        )
        # yield (
        #     "Rosenbrock",
        #     d,
        #     c.BenchmarkInstance(
        #         c.Rosenbrock(d),
        #         target=[VTR],
        #         continuous_init_lower_bound=init_lb,
        #         continuous_init_upper_bound=init_ub,
        #     ),
        # )

    for d in [
        4,
        10,
        20,
        30,
        # , 40
    ]:
        if d <= 30:
            yield (
                "CirclesInASquare",
                d,
                c.BenchmarkInstance(
                    c.CirclesInASquare(d),
                    target_objectives=[  # d *
                        VTR - cias_optima[d]
                    ],
                    continuous_init_lower_bound=0.0,
                    continuous_init_upper_bound=1.0,
                ),
            )
    # need LT/univariate model...
    # for d in [
    #     2,
    #     4,
    #     8,
    #     # 16
    # ]:
    #     # yield (
    #     #     "Griewank",
    #     #     d,
    #     #     c.BenchmarkInstance(
    #     #         c.Griewank(d),
    #     #         target_objectives=[VTR],
    #     #         continuous_init_lower_bound=4.0,
    #     #         continuous_init_upper_bound=5.0,
    #     #     ),
    #     # )
    #     yield (
    #         "Rastrigin",
    #         d,
    #         c.BenchmarkInstance(
    #             c.Rastrigin(d),
    #             target_objectives=[VTR],
    #             continuous_init_lower_bound=-5.12,
    #             continuous_init_upper_bound=5.12,
    #         ),
    #     )


def methods():
    initial_population_size = 10  # 100
    max_num_populations = 25
    restart_stale_populations = True
    # restart_stale_populations = False

    max_nis = 100  # 00
    # display name, actual method
    yield (
        "AMaLGaM",
        c.AMaLGaM(
            base_population_size=initial_population_size,
            max_num_populations=max_num_populations,
        ),
    )
    yield (
        '"RV-GOMEA (Full)"',
        c.RvGOMEA(
            base_population_size=initial_population_size,
            max_number_of_populations=max_num_populations,
            max_nis=max_nis,
            selection_during_gom=False,
            update_elitist_during_gom=False,
        ),
    )
    # yield (
    #     '"RV-GOMEA (LT)"',
    #     c.RvGOMEA(
    #         linkage_model="LinkageTree",
    #         base_population_size=initial_population_size,
    #         max_number_of_populations=max_num_populations,
    #         max_nis=max_nis,
    #         selection_during_gom=False,
    #         update_elitist_during_gom=False,
    #     ),
    # )
    # yield (
    #     '"RV-GOMEA (U)"',
    #     c.RvGOMEA(
    #         linkage_model="Univariate",
    #         base_population_size=initial_population_size,
    #         max_number_of_populations=max_num_populations,
    #         max_nis=max_nis,
    #         selection_during_gom=False,
    #         update_elitist_during_gom=False,
    #     ),
    # )
    yield (
        '"Mixed (Full)"',
        c.MixedGOMEA(
            rv_options=c.RvOptions(
                max_nis=max_nis,
                init_ams_from_population_mean=False,
                intron_aware=False,
                generations_until_full_evaluation=50,
            ),
            ims_options=c.IMSOptions(
                initial_population_size=initial_population_size,
                max_num_populations=max_num_populations,
                subgeneration_factor=8,
                restart_stale_populations=restart_stale_populations,
            ),
            sampling_model=c.AMaLGaMSamplingModel(),
        ),
    )

    for v in [  #
        "single",
        "multiple",
        "full",
        # "directed",
    ]:
        yield (
            f'"ES({v}, 1:7)"',
            c.classic.ES(strategy=v),
        )

    yield (
        '"DE"',
        c.classic.DE(
            population_size=25,
            strategy=c.classic.Rand1Bin(base="best", scale="dither"),
        ),
    )

    yield (
        '"PSO"',
        c.classic.PSO(),
    )

    return
    # yield (
    #     '"Mixed (Full, MD)"',
    #     c.MixedGOMEA(
    #         rv_options=c.RvOptions(
    #             max_nis=max_nis,
    #             init_ams_from_population_mean=False,
    #             intron_aware=False,
    #             generations_until_full_evaluation=50,
    #         ),
    #         ims_options=c.IMSOptions(
    #             initial_population_size=initial_population_size,
    #             max_num_populations=max_num_populations,
    #             subgeneration_factor=8,
    #             restart_stale_populations=restart_stale_populations,
    #         ),
    #         sampling_model=c.AMaLGaMSamplingModel(
    #             use_mahalanobis_distance_for_sdr=True
    #         ),
    #     ),
    # )
    # yield (
    #     '"Mixed (Full, RAMS)"',
    #     c.MixedGOMEA(
    #         rv_options=c.RvOptions(
    #             max_nis=max_nis,
    #             init_ams_from_population_mean=False,
    #             intron_aware=False,
    #             randomize_ams_indices=True,
    #             generations_until_full_evaluation=50,
    #         ),
    #         ims_options=c.IMSOptions(
    #             initial_population_size=initial_population_size,
    #             max_num_populations=max_num_populations,
    #             subgeneration_factor=8,
    #             restart_stale_populations=restart_stale_populations,
    #         ),
    #         sampling_model=c.AMaLGaMSamplingModel(),
    #     ),
    # )
    yield (
        '"Mixed (Full, IA)"',
        c.MixedGOMEA(
            rv_options=c.RvOptions(
                max_nis=max_nis,
                init_ams_from_population_mean=False,
                generations_until_full_evaluation=50,
                intron_aware=True,
            ),
            ims_options=c.IMSOptions(
                initial_population_size=initial_population_size,
                max_num_populations=max_num_populations,
                subgeneration_factor=8,
                restart_stale_populations=restart_stale_populations,
            ),
            sampling_model=c.AMaLGaMSamplingModel(),
            discrete_model=c.LinkageTreeFOS(
                intron_strategy="any_active",
                # filter_root=True,
            ),
        ),
    )
    yield (
        '"Mixed (LT)"',
        c.MixedGOMEA(
            rv_options=c.RvOptions(
                max_nis=max_nis,
                init_ams_from_population_mean=False,
                intron_aware=True,
                randomize_ams_indices=True,
                generations_until_full_evaluation=50,
            ),
            ims_options=c.IMSOptions(
                initial_population_size=initial_population_size,
                max_num_populations=max_num_populations,
                subgeneration_factor=8,
                restart_stale_populations=restart_stale_populations,
            ),
            sampling_model=c.AMaLGaMSamplingModel(),
            continuous_model=c.LinkageTreeFOS(),
        ),
    )
    yield (
        '"Mixed (U)"',
        c.MixedGOMEA(
            rv_options=c.RvOptions(
                max_nis=max_nis,
                init_ams_from_population_mean=False,
                intron_aware=False,
                generations_until_full_evaluation=50,
            ),
            ims_options=c.IMSOptions(
                initial_population_size=initial_population_size,
                max_num_populations=max_num_populations,
                subgeneration_factor=8,
                restart_stale_populations=restart_stale_populations,
            ),
            sampling_model=c.AMaLGaMSamplingModel(),
            continuous_model=c.UnivariateFOS(),
        ),
    )
    # yield (
    #     '"Mixed (U, IA)"',
    #     c.MixedGOMEA(
    #         rv_options=c.RvOptions(
    #             max_nis=max_nis,
    #             init_ams_from_population_mean=False,
    #             intron_aware=True,
    #             randomize_ams_indices=True,
    #             generations_until_full_evaluation=50,
    #         ),
    #         ims_options=c.IMSOptions(
    #             initial_population_size=initial_population_size,
    #             max_num_populations=max_num_populations,
    #             subgeneration_factor=8,
    #             restart_stale_populations=restart_stale_populations,
    #         ),
    #         sampling_model=c.AMaLGaMSamplingModel(
    #             use_mahalanobis_distance_for_sdr=use_mahalanobis_distance_for_sdr,
    #         ),
    #         continuous_model=c.UnivariateFOS(),
    #     ),
    # )


if __name__ == "__main__":
    domain = "continuous"
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
        max_workers=44,
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
        fig = plot_scalability(conn)
        for fmt in ["pdf", "png"]:
            fig.savefig(
                plot_dir / f"{domain}_scalability.{fmt}",
                dpi=600,
                # transparent=True,
                bbox_inches="tight",
            )
