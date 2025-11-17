import os
import pathlib
import shutil
from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

sns.set_theme(style="whitegrid")

from pygom import *

METHOD_NAME = "Mixed"
# METHOD_NAME = "RvGOMEA"

RESULTS = pathlib.Path("results") / "rv_path" / METHOD_NAME
POPULATION_LOGFILE = RESULTS / "population.csv"
SELECTION_LOGFILE = RESULTS / "selection.csv"
SUBSETS_LOGFILE = RESULTS / "subsets.csv"
SAMPLE_LOGFILE = RESULTS / "samples.csv"
STATS_LOGFILE = RESULTS / "stats.csv"

PLOT_DIR = pathlib.Path("plots") / "rv_path" / METHOD_NAME

DIMS = 20
VTR = 1e-10
INIT_BOUNDS = (100.0, 125.0)

INITIAL_POPULATION_SIZE = 25
MAX_NUM_POPULATIONS = 1
RESTART_STALE = False

X_VAR = 0
Y_VAR = 1

X_LIM = (-10.0, INIT_BOUNDS[1])
Y_LIM = (-10.0, INIT_BOUNDS[1])
ZOOM_IN = True
# ZOOM_IN = False

MAX_NUM_FRAMES = 200  # np.inf

NUM_LANDSCAPE_SAMPLES = 100

FPS = 1
PLOT_STEPS = True
# PLOT_STEPS = False

NO_VIDEO = True


def run(clean: bool = True):
    budget = Budget(max_evaluations=int(1e5))
    problem = BenchmarkInstance(
        Sphere(DIMS),
        target=[VTR],
        continuous_init_lower_bound=INIT_BOUNDS[0],
        continuous_init_upper_bound=INIT_BOUNDS[1],
    )
    if METHOD_NAME == "Mixed":
        method = MixedGOMEA(
            rv_options=RvOptions(
                max_nis=100,
                init_ams_from_population_mean=False,
                intron_aware=False,
                population_logfile=str(POPULATION_LOGFILE),
                selection_logfile=str(SELECTION_LOGFILE),
                subset_logfile=str(SUBSETS_LOGFILE),
                sample_logfile=str(SAMPLE_LOGFILE),
            ),
            ims_options=IMSOptions(
                initial_population_size=INITIAL_POPULATION_SIZE,
                max_num_populations=MAX_NUM_POPULATIONS,
                subgeneration_factor=8,
                restart_stale_populations=RESTART_STALE,
            ),
        )
    else:
        method = RvGOMEA(
            base_population_size=INITIAL_POPULATION_SIZE,
            max_number_of_populations=MAX_NUM_POPULATIONS,
            subgeneration_factor=8,
            linkage_model="Full",
        )
    print(type(method).__name__)

    if clean or not POPULATION_LOGFILE.is_file():
        Tracked.run(problem, method, budget, TrackingOptions(logpath=STATS_LOGFILE))

    return problem


def draw_solutions(ax, df, palette=None, **kwargs):
    x = np.empty(len(df))
    y = np.empty(len(df))
    h = []

    for i, (_, ind) in enumerate(df.iterrows()):
        values = literal_eval(ind["continuous"])
        x[i] = values[X_VAR]
        y[i] = values[Y_VAR]
        if palette is not None:
            h.append(palette[ind["cluster"]])

    if palette is not None:
        ax.scatter(x, y, c=h, **kwargs)
    else:
        ax.scatter(x, y, **kwargs)

    return x, y


def make_arrow(
    tail: np.ndarray,
    head: np.ndarray,
    ax: plt.Axes,
    tip_width=None,
    tip_length=None,
    color=None,
    **kwargs,
):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    cx = xlim[1] - xlim[0] / ylim[1] - ylim[0]
    cy = ylim[1] - ylim[0] / xlim[1] - xlim[0]
    r = min(xlim[1] - xlim[0], ylim[1] - ylim[0])
    ax.plot([tail[0], head[0]], [tail[1], head[1]], color=color, **kwargs)

    l = head - tail
    l /= np.linalg.norm(l)
    p = np.array([l[1] / cy, -l[0] / cx])
    p /= np.linalg.norm(p)

    if tip_length is None:
        tip_length = 0.04 * r  # tip is 1/4 of arrow
    if tip_width is None:
        tip_width = tip_length * 2 / 3
    tip = [
        head - (tip_length * l + tip_width / 2 * p),
        head,
        head - (tip_length * l - tip_width / 2 * p),
    ]
    ax.plot([c[0] for c in tip], [c[1] for c in tip], color=color, **kwargs)


def bounds(values, prev_bounds: tuple[float, float] | None):
    prev = (np.inf, -np.inf) if prev_bounds is None else prev_bounds
    if len(values) > 0:
        lb, ub = np.min(values), np.max(values)
        return min(lb, prev[0]), max(ub, prev[1])
    else:
        return prev


def limits(bounds, expand=0.05):
    d = bounds[1] - bounds[0]
    return bounds[0] - d * expand, bounds[1] + d * expand


def plot_steps(problem):
    solutions = AoSSet()
    populations = pd.read_csv(POPULATION_LOGFILE)
    selections = pd.read_csv(SELECTION_LOGFILE)
    subsets = pd.read_csv(SUBSETS_LOGFILE)
    samples = pd.read_csv(SAMPLE_LOGFILE)

    if PLOT_DIR.is_dir():
        shutil.rmtree(PLOT_DIR)
    os.makedirs(PLOT_DIR, exist_ok=True)

    clusters = sorted(populations["cluster"].unique())
    palette = {
        c: h for c, h in zip(clusters, sns.color_palette(n_colors=len(clusters)))
    }

    subset_kinds = sorted(subsets["kind"].unique())
    subset_palette = {
        c: h
        for c, h in zip(subset_kinds, sns.color_palette(n_colors=len(subset_kinds)))
    }

    frame = -1
    for g in tqdm(sorted(populations["generation"].unique())):
        frame += 1
        if frame > MAX_NUM_FRAMES:
            break

        pop = populations[populations["generation"] == g]
        sel = selections[selections["generation"] == g]
        fos = subsets[subsets["generation"] == g]
        gom = samples[(samples["generation"] == g) & (samples["step"] == "gom")]
        ams = samples[(samples["generation"] == g) & (samples["step"] == "ams")]

        fig, ax = plt.subplots()

        ax_limits = None, None
        for args, kwargs in [
            (
                (ax, pop, palette),
                dict(
                    marker="s",
                    label="Population",
                    alpha=0.25,
                    zorder=1,
                ),
            ),
            (
                (ax, sel, None),
                dict(color="green", marker="*", label="Selection", zorder=2),
            ),
            (
                (ax, gom, None),
                dict(color="red", marker="d", label="GOM Samples", zorder=3),
            ),
            (
                (ax, ams, None),
                dict(color="purple", marker="v", label="AMS Samples", zorder=4),
            ),
        ]:
            ax_limits = [
                bounds(vs, lim)
                for vs, lim in zip(
                    draw_solutions(*args, **kwargs),
                    ax_limits,
                )
            ]

        next_pop = populations[populations["generation"] == g + 1]
        if len(next_pop) > 0:
            ax_limits = [
                bounds(vs, lim)
                for vs, lim in zip(
                    draw_solutions(
                        ax,
                        next_pop,
                        palette,
                        marker="o",
                        label="Next Population",
                        alpha=0.75,
                        zorder=5,
                    ),
                    ax_limits,
                )
            ]
        else:
            print(f"No next generation found for generation {g}")

        xlim, ylim = ax_limits
        xlim = limits(bounds(xlim, X_LIM if not ZOOM_IN else None))
        ylim = limits(bounds(ylim, Y_LIM if not ZOOM_IN else None))

        x = np.linspace(*xlim, NUM_LANDSCAPE_SAMPLES)
        y = np.linspace(*ylim, NUM_LANDSCAPE_SAMPLES)

        X, Y = np.meshgrid(x, y)
        Z = X**2 + Y**2  # TODO use problem to get the first objective
        cs = ax.contourf(X, Y, Z, cmap="plasma", alpha=0.15, antialiased=True, zorder=0)
        fig.colorbar(cs, label="Objective value")

        for _, subset in fos.iterrows():
            indices = [
                idx for idx in literal_eval(subset["subset"]) if idx in (X_VAR, Y_VAR)
            ]
            if len(indices) == 1:
                pass  # TODO?
            elif len(indices) == 2:
                mean = np.array(literal_eval(subset["mean"]))[indices]
                L = np.array(literal_eval(subset["L"]))[np.ix_(indices, indices)]

                ax.scatter([mean[0]], [mean[1]], marker="*", color="orange", zorder=100)
                if g > 0:
                    mean_shift = np.array(literal_eval(subset["mean_shift"]))[
                        [X_VAR, Y_VAR]
                    ]
                    make_arrow(
                        mean, mean + 2.0 * mean_shift, ax, color="black", zorder=1
                    )

                try:
                    L_inv = np.linalg.pinv(L)

                    # https://en.wikipedia.org/wiki/Mahalanobis_distance
                    distances = np.zeros(
                        shape=(NUM_LANDSCAPE_SAMPLES, NUM_LANDSCAPE_SAMPLES)
                    )
                    for i in range(NUM_LANDSCAPE_SAMPLES):
                        for j in range(NUM_LANDSCAPE_SAMPLES):
                            z = L_inv @ (np.array([x[i], y[j]]) - mean)
                            # After "unscaling" with L_inv, the remaining covariance here is the identity and hence not needed anymore for computing the Mahalanobis distance
                            distances[j, i] = np.sqrt(z.T @ z)

                    label = subset["kind"]
                    color = subset_palette[label]

                    # cmap = sns.light_palette(kwargs.get("color", "black"), as_cmap=True).reversed()
                    c = ax.contour(
                        x,
                        y,
                        distances,
                        colors=color,
                        levels=[1.0, 2.0],  # standard deviations
                        antialiased=True,
                        algorithm="serial",
                        zorder=1,
                    )
                except Exception as e:
                    print("Skipping distribution contour: ", e)

        for l, c in subset_palette.items():
            ax.plot([], [], color=c, label=l)

        ax.set_title(f"Generation {g}")
        fig.suptitle(METHOD_NAME)

        ax.legend()

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_xlabel(rf"$x_{{{X_VAR}}}$")
        ax.set_ylabel(rf"$x_{{{Y_VAR}}}$")

        fig.savefig(PLOT_DIR / f"frame_{frame:05d}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

    if not NO_VIDEO:
        os.system(
            f"ffmpeg -y -i {PLOT_DIR}/frame_%5d.png -framerate {FPS} -s 1920x1080 -c:v libx264 -crf 18 -pix_fmt yuv420p {PLOT_DIR.parent / (METHOD_NAME + '.mp4')}"
        )


def plot_path(problem):
    stats = pd.read_csv(STATS_LOGFILE)

    os.makedirs(PLOT_DIR, exist_ok=True)

    fig, ax = plt.subplots()

    x = np.empty(len(stats))
    y = np.empty(len(stats))
    g = np.empty(len(stats))
    for i, (_, ind) in enumerate(stats.iterrows()):
        values = literal_eval(ind["continuous"])
        x[i] = values[X_VAR]
        y[i] = values[Y_VAR]
        g[i] = ind["generation"]

    c = ax.scatter(x, y, c=g, cmap="viridis")
    fig.colorbar(c, label="Generation")

    x = np.linspace(*ax.get_xlim(), NUM_LANDSCAPE_SAMPLES)
    y = np.linspace(*ax.get_ylim(), NUM_LANDSCAPE_SAMPLES)

    X, Y = np.meshgrid(x, y)

    solutions = AoSSet()
    Z = (
        X**2 + Y**2
    )  # TODO use problem to get the first objective (centered around the last solution in case of more than 2 parameters...)
    cs = ax.contourf(X, Y, Z, cmap="plasma", alpha=0.15, antialiased=True, zorder=0)
    fig.colorbar(cs, label="Objective value")

    ax.set_xlabel(rf"$x_{{{X_VAR}}}$")
    ax.set_ylabel(rf"$x_{{{Y_VAR}}}$")

    ax.set_title(METHOD_NAME)

    fig.savefig(PLOT_DIR.parent / f"{METHOD_NAME}.png", bbox_inches="tight", dpi=600)


if __name__ == "__main__":
    problem = run(
        # .
        # clean=False
    )

    if PLOT_STEPS and METHOD_NAME == "Mixed":
        plot_steps(problem)
    plot_path(problem)
