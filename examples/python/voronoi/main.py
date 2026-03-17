import pathlib
import shutil

import numpy as np
import pygom
from PIL import Image, ImageOps
from pygom import *

MIN_NUM_CELLS = 10
MAX_NUM_CELLS = 100
MAX_NUM_PIXELS = 64
COMPLEXITY_OBJECTIVE = False
COMPLEXITY_OBJECTIVE = True

# kdtree
# direct step loop
# operator combinators?
# better objectives ()

def problem_from_image(image_path: str, init: InitBase | None = None):
    im = Image.open(image_path).convert("RGB")

    w, h = im.size
    m = max(w, h)
    if m > MAX_NUM_PIXELS:
        scale_factor = MAX_NUM_PIXELS / m
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
        "img/reference_image.jpg", init=RandomInit()
    )

    alg = pygom.classic.SimpleGA(
        population_size=100,
        steady_state=True,
        crossover=pygom.classic.NPointCrossover(1),
        mutation=pygom.classic.LocalizedMutation(),
        selection=(
            pygom.classic.TruncationSelection()
            if COMPLEXITY_OBJECTIVE else
            pygom.classic.TournamentSelection(4)
        ),
    )

    # alg = MixedGOMEA()

    budget = Budget(max_time_seconds=10)
    budget = Budget(max_evaluations=int(1e5))

    # WHY doesn't this work???
    assert isinstance(problem, InstanceBase)
    assert isinstance(alg, MethodBase)
    # archive, status = alg.run(problem, budget)
    archive, status = Tracked.run(
        problem,
        alg,
        budget,
        TrackingOptions(
            "logs/baseline.csv",
            log_info=[  # add any extra fields to log here
                ("method_name", "baseline"),
            ],
        ),
    )

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
