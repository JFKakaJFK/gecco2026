import pathlib
from ast import literal_eval

import pandas as pd
from pygom import *
from pygom.classic import *
from pygom.voronoi import (
    VoronoiImageReconstruction,
    YourCustomCrossover,
    YourCustomMutation,
)
from src.utils import (
    animate_evolution,
    example_convergence_plot,
    load_image_data,
    save_images,
)

MIN_NUM_CELLS = 10
MAX_NUM_CELLS = 100
IMAGE_MAX_DIM = 64
# Number of active cells needed to switch to using a KD-Tree for determining the nearest voronoi cell during evaluation. This depends on the number of pixels and cells and the hardware, but generally brute-forcing is fast...
KDTREE_THRESHOLD = 200
COMPLEXITY_OBJECTIVE = False
COMPLEXITY_OBJECTIVE = True

# If enabled, the complexity objecitve is logged even if it is not optimized
TRACK_COMPLEXITY = True

RESULT_DIR = pathlib.Path("results")

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

    # Note: the provided TruncationSelection implementation performs NSGA-II
    # like selection using fast non-dominated sorting with multiple objectives
    selection = (
        TruncationSelection() if COMPLEXITY_OBJECTIVE else TournamentSelection(4)
    )

    ea = SimpleGA(
        population_size=100,
        steady_state=True,
        # apply uniform/one-point crossover with different probabilities
        crossover=CombinedCrossover(
            [
                (UniformCrossover(0.25), 0.4),
                (NPointCrossover(1), 0.6),
            ]
        ),
        # or e.g.
        # crossover=YourCustomCrossover(), # uniform crossover until you change it
        mutation=RandomMutation(),
        # or e.g.
        # mutation=YourCustomMutation(),
        selection=selection,
    )

    budget = Budget(max_time_seconds=10, max_evaluations=int(1e6), max_generations=1000)

    seed = None
    odir = RESULT_DIR / "SimpleGA"

    print("Running...", end="", flush=True)
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
    print(f"Saved best images to {odir / 'archive'}")

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
        print(f"Saved animation to {odir / 'evolution.gif'}")

        example_convergence_plot(
            df,
            elites,
            multi_objective,
            metric="total_time_seconds",
            metric_label="Time [s]",
        )
