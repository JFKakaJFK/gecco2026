import pathlib
import shutil

import numpy as np
import pygom
from PIL import Image, ImageOps
from pygom import *

MIN_NUM_CELLS = 10
MAX_NUM_CELLS = 100
IMAGE_MAX_DIM = 64
COMPLEXITY_OBJECTIVE = False
COMPLEXITY_OBJECTIVE = True

# learning objectives
# - work with a C++ based python library
# - reconstruct image, the point is to have an optimization problem with interesting properties that is fun, not efficiency
#   - no changing the fitness function beyond the provided variables ()
# - conditionally inactive variables
# - linkage
# - mo

def problem_from_image(image_path: str, init: InitBase | None = None):
    im = Image.open(image_path).convert("RGB")

    w, h = im.size
    m = max(w, h)
    if m > IMAGE_MAX_DIM:
        scale_factor = IMAGE_MAX_DIM / m
        im = ImageOps.cover(im, (int(scale_factor * w), int(scale_factor * h)))
        w, h = im.size
    else:
        scale_factor = 1.0

    # im.show()

    data = np.array(im.get_flattened_data(), dtype=np.uint8)
    return VoronoiImageReconstruction(
        target_image=data,
        width=w,
        height=h,
        min_num_cells=MIN_NUM_CELLS,
        max_num_cells=MAX_NUM_CELLS,
        complexity_objective=COMPLEXITY_OBJECTIVE,
        init=init,
        # use_kdtree=True
    ), scale_factor

if __name__ == "__main__":
    problem, scale_factor = problem_from_image(
        "img/reference_image.jpg", init=CompleteInit() # RandomInit()
    )

    alg = pygom.classic.SimpleGA(
        population_size=100,
        steady_state=True,
        crossover=pygom.classic.UniformCrossover(0.25),
        # crossover=pygom.classic.NPointCrossover(1),
        mutation=pygom.classic.LocalizedMutation(),
        selection=(
            pygom.classic.TruncationSelection()
            if COMPLEXITY_OBJECTIVE else
            pygom.classic.TournamentSelection(4)
        ),
    )

    custom_similarity = np.zeros((6 * MAX_NUM_CELLS, 6 * MAX_NUM_CELLS), dtype=np.float64)

    # min num cells are linked with each other
    custom_similarity[:6 * MIN_NUM_CELLS, :6*MIN_NUM_CELLS] = 0.01
    for i in range(MAX_NUM_CELLS):
        ii = 6 * i

        #      / \
        #    /   {X,Y, R, G, B}
        #  /     /       \
        # /     {X, Y}   {R, G, B}

        for j in range(6):
            for k in range(6):
                if 1 <= j <= 2 and 1 <= k <= 2: # X,Y
                    custom_similarity[ii + j, ii + k] = 1.0
                elif 3 <= j and 3 <= k: # R,G,B
                    custom_similarity[ii + j, ii + k] = 1.0
                else:
                    custom_similarity[ii + j, ii + k] = 0.5

    # alg = MixedGOMEA(
    #     ims_options=IMSOptions(initial_population_size=2, max_num_populations=4 + int(COMPLEXITY_OBJECTIVE), restart_stale_populations=True),
    #     population_options=PopulationOptions(forced_improvements=True),
    #     discrete_model=LinkageTreeFOS(custom_similarity=custom_similarity)
    # )

    budget = Budget(max_time_seconds=10)
    # budget = Budget(max_evaluations=int(1e5))
    # budget = Budget(max_generations=500)

    # WHY doesn't this work???
    assert isinstance(problem, InstanceBase)
    assert isinstance(alg, MethodBase)
    assert isinstance(budget, Budget)
    # archive, status = alg.run(problem, budget) # breaks because the defaults for seed/population size are not provided...
    # archive, status = alg.run(problem, budget, seed=None, population_size=None)
    archive, status = alg.run(problem, budget)
    # archive, status = Tracked.run( # works fine
    #     problem,
    #     alg,
    #     budget,
    #     TrackingOptions(
    #         "logs/baseline.csv",
    #         log_info=[  # add any extra fields to log here
    #             ("method_name", "baseline"),
    #         ],
    #     ),
    #     # seed=42
    # )

    odir = pathlib.Path("archive")
    if odir.exists():
        shutil.rmtree(odir)
    odir.mkdir(exist_ok=True, parents=True)
    for i in range(archive.size()):
        print(f"Solution {i}: {archive[i].quality().objectives}")

        # save to archive/img
        data, w, h = problem.image_data(archive[i], scale=1.0 / scale_factor)
        im = Image.fromarray(data.reshape(h, w, 3), "RGB")
        # im.show()
        im.save(odir / f"img{i:04d}.png")
