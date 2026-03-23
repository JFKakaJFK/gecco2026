import pathlib
from ast import literal_eval

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pygom import *
from pygom.classic import *
from pygom.voronoi import (
    VoronoiImageReconstruction,
    YourCustomCrossover,
    YourCustomMutation,
)

from src.utils import animate_evolution, load_image_data, save_images

MIN_NUM_CELLS = 25
MAX_NUM_CELLS = 25
IMAGE_MAX_DIM = 64
# Number of active cells needed to switch to using a KD-Tree for determining the nearest voronoi cell during evaluation. This depends on the number of pixels and cells and the hardware, but generally brute-forcing is fast...
KDTREE_THRESHOLD = 200
COMPLEXITY_OBJECTIVE = False
COMPLEXITY_OBJECTIVE = True

# If enabled the complexity objecitve is logged even if it is not optimized
TRACK_COMPLEXITY = True

RESULT_DIR = pathlib.Path("results")

# learning objectives
# - work with a C++ based python library
# - reconstruct image, the point is to have an optimization problem with interesting properties that is fun, not efficiency
#   - no changing the fitness function beyond the provided variables ()
# - conditionally inactive variables
# - linkage
# - mo

# i.e. you know what each decision variable represents, but you are not allowed to change the underlying fitness function


# deliverables
# - initialisation & variation: 2 hypotheses
# - mo: 1 hypothesis
# - proper experiments, >= 30 repeats & statistical testing


if __name__ == "__main__":
    *image, scale_factor = load_image_data("img/reference_image.jpg", IMAGE_MAX_DIM)

    init = RandomInit()

    problem = VoronoiImageReconstruction(
        *image,
        min_num_cells=MIN_NUM_CELLS,
        max_num_cells=MAX_NUM_CELLS,
        complexity_objective=COMPLEXITY_OBJECTIVE,
        track_complexity=TRACK_COMPLEXITY,
        init=init,
        kdtree_threshold=KDTREE_THRESHOLD,
    )

    # Note: the TruncationSelection implementation performs NSGA-II like
    # selection using fast non-dominated sorting with multiple objectives
    selection = (
        TruncationSelection() if COMPLEXITY_OBJECTIVE else TournamentSelection(4)
    )

    ea = SimpleGA(
        population_size=100,
        steady_state=True,
        # apply uniform/one-point/your custom (*) crossover with different probabilities
        # *: uniform crossover until you change it
        crossover=CombinedCrossover(
            [
                (UniformCrossover(0.25), 0.3),
                (NPointCrossover(1), 0.3),
                (YourCustomCrossover(), 0.4),
            ]
        ),
        mutation=CombinedMutation(
            [(RandomMutation(), 0.9), (YourCustomMutation(), 0.1)]
        ),
        selection=selection,
    )

    budget = Budget(  #
        max_time_seconds=600,
        # max_evaluations=int(1e6), max_generations=1000
    )

    seed = None
    odir = RESULT_DIR / "SimpleGA"

    print("Running...", end="")
    if odir is None:
        archive, status = ea.run(problem, budget, seed=seed)
    else:
        archive, status = Tracked.run(
            problem,
            ea,
            budget,
            TrackingOptions(
                str(odir / "stats.csv"),
                # you can log any str,str header/value tuples you want here
                log_info=[("method_name", "SimpleGA")],
            ),
            seed=seed,
        )

    print(f" done.\nBest objective(s): {archive.so_solution(0).quality().objectives}")

    save_images(
        odir / "archive",
        problem,
        archive,
        scale_factor,
        show_objectives=True,  # add text showing the objective value(s)
        clean=True,  # empty the directory beforehand
    )

    if odir:
        multi_objective = TRACK_COMPLEXITY or COMPLEXITY_OBJECTIVE

        df = pd.read_csv(odir / "stats.csv")
        df["objectives"] = df["objectives"].apply(literal_eval)
        df["reconstruction_error"] = df["objectives"].apply(lambda o: o[0])

        elites = df.sort_values(["generation", "reconstruction_error"]).drop_duplicates(
            "generation", keep="first"
        )

        animate_evolution(
            odir / "evolution.gif",
            problem,
            elites,
            sorted(elites["generation"].unique()),
            scale_factor=scale_factor,
        )

        metric = "generation"
        metric_label = "Generations"

        fig, axes = plt.subplots(ncols=1 + int(multi_objective), figsize=(8, 4))
        ax = axes[0] if multi_objective else axes

        sns.lineplot(
            elites,
            x=metric,
            y="reconstruction_error",
            hue="method_name",
            ax=ax,
        )

        ax.set_xlabel("Generations")
        ax.set_title("Reconstruction Error")

        if multi_objective:
            elites["num_cells"] = elites["objectives"].apply(lambda o: o[1])

            nc_ax = axes[1]

            sns.lineplot(
                elites,
                x=metric,
                y="num_cells",
                hue="method_name",
                alpha=0.5,
                ls="dashed",
                ax=nc_ax,
            )

            nc_ax.set_xlabel("Generations")
            nc_ax.set_title("Cell count")

        plt.show()
