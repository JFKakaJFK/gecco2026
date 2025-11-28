import os
import pathlib

import pandas as pd
from pygom import *

from src.config import Config, c
from src.plots import plot_scalability
from src.postprocessing import load_results
from src.run import run_all, run_one

REPEATS = 30
REPEATS = 500

budget = c.Budget(max_evaluations=int(1e6))


def problems():
    for d in [5, 10, 20, 40, 80, 160]:
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
        # yield (
        #     "DTrap5",
        #     d,
        #     c.BenchmarkInstance(
        #         c.Repeat(c.DeceptiveTrap(5), d // 5),
        #         target=[float(d)],
        #         init=c.RandomInit(),
        #     ),
        # )
    #     yield (
    #         "LeadingOnes",
    #         d,
    #         c.BenchmarkInstance(
    #             c.Masked(c.LeadingOnes(d)),
    #             target=[float(d)],
    #             init=c.RandomInit(),
    #         ),
    #     )

    init = c.CompleteInit()
    # init = c.RandomInit()
    # for d in [10, 25, 50, 100, 200]:
    #     yield (
    #         "LeadingOnes IA",
    #         d,
    #         c.BenchmarkInstance(c.LeadingOnes(d), target=[float(d)], init=init),
    #     )
    #     branching_factor = 2
    #     yield (
    #         "HLeadingOnes IA",
    #         d,
    #         c.BenchmarkInstance(
    #             c.HLeadingOnes(d, branching_factor),
    #             discrete_domain=branching_factor
    #             + 1,  # each node can be in [0, branching_factor]
    #             target=[float(d)],
    #             init=init,
    #         ),
    #     )


def methods():
    initial_population_size = 2
    max_num_populations = 25
    initial_population_size = 100
    max_num_populations = 1
    restart_stale_populations = False  # True  # by default, we want this only for "relevant" populations that still contribute new improvements - so only the largest population in single-objective population, but due to forced improvements any population in a multi-objective setting.

    # display name, actual method
    # yield (
    #     '"Library (GIGA, U)"',
    #     c.DiscreteGOMEA(linkage_model="Univariate", gene_invariant=True),
    # )

    for metric in [  #
        "mi",
        # "nmi"
    ]:
        for linkage_model in [  #
            "Univariate",
            # "LinkageTree",
        ]:
            for forced_improvements in [  #
                # True,
                False
            ]:
                lm = "".join(c for c in linkage_model if c == c.upper())

                common_desc = (
                    f"{metric.upper()}, {lm}{', FI' if forced_improvements else ''}"
                )

                for fos_order in ["default", "parallel", "fixed"]:
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
                            max_number_of_populations=max_num_populations,
                            fos_order=fos_order,
                        ),
                    )

                for dsp in [
                    0.0,
                    # 0.05,
                    # 1.0,
                ]:
                    for strict_elite_acceptance in [False, True]:
                        common_options = dict(
                            population_options=c.PopulationOptions(
                                donor_search_proportion=dsp,
                                forced_improvements=forced_improvements,
                                strict_elite_acceptance=strict_elite_acceptance,
                            ),
                            ims_options=c.IMSOptions(
                                initial_population_size=initial_population_size,
                                max_num_populations=max_num_populations,
                                restart_stale_populations=restart_stale_populations,
                            ),
                        )
                        dsp_desc = f", DSP={dsp}" if dsp > 0.0 else ""
                        sea_desc = ", strict" if strict_elite_acceptance else ""
                        desc = common_desc + dsp_desc + sea_desc

                        if lm == "U":
                            yield (
                                f'"Mixed ({desc})"',
                                c.MixedGOMEA(
                                    discrete_model=c.UnivariateFOS(), **common_options
                                ),
                            )
                        elif lm == "LT":
                            for intron_strategy in [
                                "none",
                                "any_active",
                                # "all_active", # definitely bad, as expected
                                "mark_only",
                            ]:
                                is_desc = dict(
                                    none="",
                                    any_active=r", $IA_{any}$",
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


def add_reference_results(odir: pathlib.Path | str):
    # copy over results from yet another reference to make the format match...
    path = pathlib.Path("/Users/johannes/Documents/github/pygomea/results/pygomea")

    opath = pathlib.Path(odir) / "pygom.csv"

    problem = "LeadingOnes IA"

    if not (opath.parent / problem).is_dir():
        if opath.exists():
            opath.unlink()
            return

    with open(opath, "w+") as f:
        f.write(
            "status,evaluations,generation,total_time_seconds,alg_time_seconds,eval_time_seconds,problem_name,method_name,dims,run,seed,discrete,discrete_active,continuous,continuous_active,values,objectives,constraint_value\n"
        )
        for method_dir in path.iterdir():
            method = dict(
                BBO_ignore="PyGOMEA BBO",
                IA_ignore="PyGOMEA GP",
                IA_any_active="PyGOMEA $IA_{any}$",
            )[method_dir.name]
            for dim_dir in method_dir.iterdir():
                dims = int(dim_dir.name[:-1])
                for run_file in dim_dir.iterdir():
                    run = int(run_file.name.split(".")[0])
                    # status,evaluations,total_time_seconds,alg_time_seconds,eval_time_seconds,values,fitness
                    # ->
                    # status,evaluations,generation,total_time_seconds,alg_time_seconds,eval_time_seconds,problem_name,method_name,dims,run,seed,discrete,discrete_active,continuous,continuous_active,values,objectives,constraint_value

                    df = pd.read_csv(run_file)
                    row = df.iloc[-1]

                    assert row["status"] == "SUCCESS", row
                    f.write(
                        f"""TargetReached,{row["evaluations"]},0,{row["total_time_seconds"]},{row["alg_time_seconds"]},{row["eval_time_seconds"]},{problem},"{method}",{dims},{run},{run},"[{row["values"]}]","[{row["values"]}]","[]","[]","([{row["values"]}],[])","[{row["fitness"][2:-1].split(",")[0]}]",0\n"""
                    )


def debug_one():
    problem = "LeadingOnes IA"
    method = '"Mixed (MI, LT, FI, IA)"'
    dims = 150
    run = 58

    task_path = f"results/discrete_raw/{problem}/{dims}/{method}/task.yaml"
    logfile = pathlib.Path(
        f"results/debug/discrete/{problem}/{dims}/{method}/{run:03d}.csv"
    )

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
    run_one(task_path, logfile, loginfo, seed=15329365302178898957)


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

    add_reference_results(output_directory)

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


if __name__ == "__main__":
    main()
    # debug_one()
