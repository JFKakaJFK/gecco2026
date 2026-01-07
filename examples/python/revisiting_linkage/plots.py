import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pygom
import seaborn as sns
from pygom import *
from rliable import library as rly
from rliable import metrics, plot_utils
from tqdm import tqdm

from src.config import c, extract, instantiate, load_config
from src.data import prepare_problem, problem_info
from src.plots import plot_convergence_so
from src.postprocessing import load_results, rliable_score_dict
from src.run import compute_run_path, run_tasks

RESULT_DIR = pathlib.Path("results") / "init"  #  "linkage_wrong_init"
LOG_DIR = RESULT_DIR / "raw"
PARQUET_DIR = RESULT_DIR / "processed"
PLOT_DIR = RESULT_DIR / "plots"

# TODO rliable/baycomp comparison
# Per problem comparison


def main():
    with load_results(
        LOG_DIR,
        # enable pre-processing the .csv logs into .parquet files
        # preprocess=True,
        parquet_dir=PARQUET_DIR,
    ) as conn:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        (PLOT_DIR / "rliable").mkdir(parents=True, exist_ok=True)

        score_dict, problems = rliable_score_dict(
            conn,
            run_expr="format('{}.{}', fold, run)",
            problem_query="format('{}\nH={}\nLS={}', problem_name, template_height::STRING,IF(linear_scaling, 'Yes', 'No')::STRING)",  # operator_set::STRING,
            method_query="format('{} {}', method_name, init)",
            normalized_value_expr="1.0 - nmse_train",
        )

        algorithms = list(score_dict.keys())

        aggregate_func = lambda x: np.array(
            [
                metrics.aggregate_median(x),
                metrics.aggregate_iqm(x),
                metrics.aggregate_mean(x),
                metrics.aggregate_optimality_gap(x),
            ]
        )
        aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
            score_dict, aggregate_func, reps=5000
        )
        fig, axes = plot_utils.plot_interval_estimates(
            aggregate_scores,
            aggregate_score_cis,
            metric_names=["Median", "IQM", "Mean", "Optimality Gap"],
            algorithms=algorithms,
            xlabel="Human Normalized Score",
        )

        fig.savefig(
            PLOT_DIR / "rliable" / "interval_estimates.pdf",
            dpi=600,
            bbox_inches="tight",
            transparent=True,
        )

        plt.clf()

        thresholds = np.linspace(0.0, 1.0, 50)
        score_distributions, score_distributions_cis = rly.create_performance_profile(
            score_dict, thresholds
        )

        fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
        plot_utils.plot_performance_profiles(
            score_distributions,
            thresholds,
            performance_profile_cis=score_distributions_cis,
            # use_non_linear_scaling=True,
            # xticks=[],
            colors=dict(zip(algorithms, sns.color_palette("colorblind"))),
            xlabel=r"Human Normalized Score $(\tau)$",
            ax=ax,
        )
        # ax.set_yscale("log")
        fig.legend()

        fig.savefig(
            PLOT_DIR / "rliable" / "pprof.pdf",
            dpi=600,
            bbox_inches="tight",
            transparent=True,
        )
        plt.clf()

        exit()

        for split in [  #
            "train",
            "test",
        ]:
            plot_convergence_so(
                PLOT_DIR / f"convergence_{split}",
                conn,
                # y_var="1.0 - objectives[1]::DOUBLE",  # transform NMSE into R2
                y_var=f"1.0 - nmse_{split}",
                y_agg="MAX",  # higher R^2 is better
                y_label=f"$R^2$ {split.title()}",
                ymin="auto",
                ymax="auto",
                # merge folds and runs into one seaborn "unit"
                unit_query="format('{}.{}', fold, run)",
                # split up the plot into the following rows
                modifier_query="[template_height::STRING,operator_set::STRING,IF(linear_scaling, 'Yes', 'No')::STRING]",
                modifier_labels=[
                    "Template Height",
                    "Operators",
                    "Linear Scaling",
                ],
                nsamples=100,
            )


if __name__ == "__main__":
    main()
