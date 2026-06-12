import os
import pathlib

import pandas as pd
import pygom
from pygom import *
from src.config import Config, c
from src.plots import plot_convergence_so, plot_scalability
from src.postprocessing import load_results
from src.run import run_all, run_one
from tqdm import tqdm

REPEATS = 30
REPEATS = 100

budget = c.Budget(max_evaluations=int(1e6))


def problems():
    for d in [  #
        5,
        10,
        20,
        40,
        80,
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

    # for init in ["random", "complete"]:
    #     for d in [
    #         10,
    #         25,
    #         50,
    #         100,  # 200, 400
    #     ]:
    #         actual_init = c.RandomInit() if init == "random" else c.CompleteInit()
    #         yield (
    #             f"LeadingOnes IA ({init})",
    #             d,
    #             c.BenchmarkInstance(
    #                 c.LeadingOnes(d), target=[float(d)], init=actual_init
    #             ),
    #         )
    #         branching_factor = 2
    #         yield (
    #             f"HLeadingOnes IA ({init})",
    #             d,
    #             c.BenchmarkInstance(
    #                 c.HLeadingOnes(d, branching_factor),
    #                 discrete_domain=branching_factor
    #                 + 1,  # each node can be in [0, branching_factor]
    #                 target=[float(d)],
    #                 init=actual_init,
    #             ),
    #         )


def methods():
    initial_population_size = 2
    subgeneration_factor = 4
    max_num_populations = 25
    # initial_population_size = 100
    # initial_population_size = 1024
    # max_num_populations = 1
    restart_stale_populations = False  # True  # by default, we want this only for "relevant" populations that still contribute new improvements - so only the largest population in single-objective population, but due to forced improvements any population in a multi-objective setting.

    # display name, actual method
    # yield (
    #     '"Library (GIGA, U)"',
    #     c.DiscreteGOMEA(linkage_model="Univariate", gene_invariant=True),
    # )
    # yield (
    #     '"Mixed (GIGA)"',
    #     c.MixedGOMEA(
    #         discrete_model=c.LinkageTreeFOS(
    #             metric="random",
    #             filter_root=True,
    #         ),
    #         population_options=c.PopulationOptions(gene_invariant=True),
    #     ),
    # )

    for sn, selection in [
        # ("Truncation", c.classic.TruncationSelection()),
        ("T4", c.classic.TournamentSelection(4)),
    ]:
        for cx, crossover in [
            ("UX", c.classic.UniformCrossover(0.5)),
            # ("1PT", c.classic.NPointCrossover(1)),
            # ("2PT", c.classic.NPointCrossover(2)),
        ]:
            yield (
                f'"SimpleGA(SS, S={sn},C={cx})"',
                c.classic.SimpleGA(
                    population_size=100,
                    crossover=crossover,
                    selection=selection,
                ),
            )
            # if sn != "Truncation":
            #     yield (
            #         f'"SimpleGA(G, S={sn},C={cx})"',
            #         c.classic.SimpleGA(
            #             population_size=100,
            #             crossover_strategy=crossover,
            #             selection_strategy=selection,
            #             steady_state=False,
            #         ),
            #     )

    for metric in [  #
        "mi",
        # "nmi",
    ]:
        for linkage_model in [  #
            # "Univariate",
            "LinkageTree",
        ]:
            for forced_improvements in [  #
                True,
                # False,
            ]:
                lm = "".join(c for c in linkage_model if c == c.upper())

                common_desc = (
                    f"{metric.upper()}, {lm}{', FI' if forced_improvements else ''}"
                )

                for fos_order in [  #
                    "default",  # "parallel", "fixed"
                ]:
                    desc = common_desc + (
                        f", {fos_order}" if fos_order != "default" else ""
                    )
                    yield (
                        f'"Library ({desc})"',
                        c.DiscreteGOMEA(
                            linkage_model=linkage_model,
                            forced_improvements=forced_improvements,
                            similarity_metric=metric.upper(),
                            base_population_size=initial_population_size,
                            subgeneration_factor=subgeneration_factor,
                            max_number_of_populations=max_num_populations,
                            fos_order=fos_order,
                        ),
                    )

                for dsp in [  #
                    0.0,
                    # 0.05,
                    # 1.0,
                ]:
                    for strict_elite_acceptance in [  #
                        False,
                        # True
                    ]:
                        for sequential_gom in [  #
                            False,
                            # True,
                        ]:
                            common_options = dict(
                                population_options=c.PopulationOptions(
                                    donor_search_proportion=dsp,
                                    forced_improvements=forced_improvements,
                                    strict_elite_acceptance=strict_elite_acceptance,
                                    sequential_gom=sequential_gom,
                                ),
                                ims_options=c.IMSOptions(
                                    initial_population_size=initial_population_size,
                                    max_num_populations=max_num_populations,
                                    subgeneration_factor=subgeneration_factor,
                                    restart_stale_populations=restart_stale_populations,
                                ),
                            )
                            dsp_desc = f", DSP={dsp}" if dsp > 0.0 else ""
                            sea_desc = ", strict" if strict_elite_acceptance else ""
                            seq_desc = ", seq" if sequential_gom else ""
                            desc = common_desc + dsp_desc + sea_desc + seq_desc

                            if lm == "U":
                                yield (
                                    f'"Mixed ({desc})"',
                                    c.MixedGOMEA(
                                        discrete_model=c.UnivariateFOS(),
                                        **common_options,
                                    ),
                                )
                            elif lm == "LT":
                                for intron_strategy in [
                                    "none",
                                    # "any_active",
                                    # "weighted_any_active",
                                    # "all_active",  # definitely bad, as expected
                                    # "mark_only",
                                ]:
                                    is_desc = dict(
                                        none="",
                                        any_active=r", $IA_{any}$",
                                        weighted_any_active=r", $IA_{w,any}$",
                                        all_active=r", $IA_{all}$",
                                        mark_only=r", $IA_{mark}$",
                                    )[intron_strategy]
                                    yield (
                                        f'"Mixed ({desc}{is_desc})"',
                                        c.MixedGOMEA(
                                            discrete_model=c.LinkageTreeFOS(
                                                metric=metric,
                                                intron_strategy=intron_strategy,
                                                filter_root=True,
                                            ),
                                            **common_options,
                                        ),
                                    )


def add_reference_results(odir: pathlib.Path | str, all_rows: bool = False):
    """Just for debugging, loads results from a different version that"""
    # copy over results from yet another reference to make the format match...
    path = pathlib.Path(
        "/Users/johannes/Documents/github/pygomea/results/pygomea"
    )  # all methods
    path = pathlib.Path(
        "/Users/johannes/Documents/github/pygomea/results/pygomea100"
    )  # pop size 100
    path = pathlib.Path(
        "/Users/johannes/Documents/github/pygomea/results/pygomea2"
    )  # IMS up to 200D, IA only

    if not path.exists():
        print("No reference results found, skipping ...")
        return

    opath = pathlib.Path(odir) / "pygom.csv"

    problem = "LeadingOnes IA"

    problems = [d.name for d in opath.parent.glob(f"{problem}*")]

    if len(problems) == 0:  # not (opath.parent / problem).is_dir():
        print(f"Skipping since {problem} results were not found...")
        if opath.exists():
            opath.unlink()
            return
    # elif not opath.exists():
    else:
        with open(opath, "w+") as f:
            f.write(
                "status,evaluations,generation,total_time_seconds,alg_time_seconds,eval_time_seconds,problem_name,method_name,dims,run,seed,discrete,discrete_active,continuous,continuous_active,values,objectives,constraint_value\n"
            )
            for p in problems:
                for method_dir in tqdm(path.iterdir()):
                    method = dict(
                        BBO_ignore="PyGOMEA BBO",
                        IA_ignore="PyGOMEA GP",
                        IA_any_active="PyGOMEA $IA_{any}$",
                    )[method_dir.name]
                    for dim_dir in tqdm(method_dir.iterdir()):
                        dims = int(dim_dir.name[:-1])
                        for run_file in dim_dir.iterdir():
                            run = int(run_file.name.split(".")[0])
                            # status,evaluations,total_time_seconds,alg_time_seconds,eval_time_seconds,values,fitness
                            # ->
                            # status,evaluations,generation,total_time_seconds,alg_time_seconds,eval_time_seconds,problem_name,method_name,dims,run,seed,discrete,discrete_active,continuous,continuous_active,values,objectives,constraint_value

                            df = pd.read_csv(run_file)

                            for row_idx in range(len(df)) if all_rows else [-1]:
                                row = df.iloc[row_idx]

                                status = dict(
                                    SUCCESS="TargetReached",
                                    CONVERGED="Converged",
                                    RUNNING="Running",
                                    MAX_EVALS="EvaluationLimitReached",
                                    TIME_EVALS="TimeLimitReached",
                                )[row["status"]]
                                values = row["values"].replace("█", "0")
                                f.write(
                                    f"""{status},{row["evaluations"]},0,{row["total_time_seconds"]},{row["alg_time_seconds"]},{row["eval_time_seconds"]},{p},"{method}",{dims},{run},{run},"[{values}]","[{values}]","[]","[]","([{row["values"]}],[])","[{row["fitness"][2:-1].split(",")[0]}]",0\n"""
                                )


def debug_one():
    """A helper to re-run outliers (with a profiler attached, e.g. `samply record uv run discrete.py`)"""
    problem = "LeadingOnes IA"
    method = '"Mixed (MI, LT, FI, IA)"'
    dims = 150
    run = 58

    task_path = f"results/discrete_raw/{problem}/{dims}/{method}/task.yaml"
    logfile = pathlib.Path(
        f"results/debug/discrete/{problem}/{dims}/{method}/{run:03d}.csv"
    )
    seed = 15329365302178898957

    if logfile.exists():
        logfile.unlink()

    os.makedirs(pathlib.Path(logfile).parent, exist_ok=True)

    loginfo = list(
        dict(
            problem_name=problem,
            method_name=method,
            dims=str(dims),
            run=str(run),
        ).items()
    )
    run_one(task_path, logfile, loginfo, seed=seed)


def main():
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

    # add_reference_results(output_directory, all_rows=True)

    preprocess = True
    # preprocess = False

    plot_dir = pathlib.Path("plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    with load_results(
        output_directory,
        parquet_dir=parquet_dir,
        preprocess=preprocess,
    ) as conn:
        for log_y in [True, False]:
            fig = plot_scalability(
                conn,
                # methods=["Library (MI, LT)", "Library (MI, U)", "MI LT", "MI (U)"],
                log_y=log_y,
            )
            for fmt in ["pdf", "png"]:
                fig.savefig(
                    plot_dir / f"{domain}_scalability{['', '_logy'][log_y]}.{fmt}",
                    dpi=600,
                    # transparent=True,
                    bbox_inches="tight",
                )

        # plot_convergence_so(
        #     plot_dir / f"{domain}_convergence",
        #     conn,
        #     y_agg="MAX",
        #     ymin="0",
        #     # y_var="objectives[1]::DOUBLE / dims::DOUBLE",
        #     # metrics=["evaluations / dims::DOUBLE", "total_time_seconds / dims::DOUBLE"],
        #     # metric_labels=[r"$\frac{Evaluations}{Dimensions}$", "Time/Dimensions [s]"],
        #     modifier_query="[dims]",
        #     modifier_labels=["Dimensions"],
        #     # show_generation_boundaries=True,
        #     nsamples=100,
        # )


if __name__ == "__main__":
    main()
    # debug_one()
