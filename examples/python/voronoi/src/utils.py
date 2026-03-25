import pathlib
import shutil
from ast import literal_eval

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pygom import *


def example_convergence_plot(
    df,
    elites,
    multi_objective,
    metric: str = "generation",
    metric_label: str = "Generations",
):
    """
    An example plot that shows how the objectives, hypervolume and fronts (if multi-objective) can be plotted
    """
    fig, axes = plt.subplot_mosaic(
        "EC\nHF" if multi_objective else "E",
        figsize=(8, [4, 8][multi_objective]),
    )

    # show first objective
    sns.lineplot(
        elites,
        x=metric,
        y="reconstruction_error",
        hue="method_name",
        ax=axes["E"],
        legend=False,
    )
    axes["E"].set_xlabel(metric_label)
    axes["E"].set_ylabel("Reconstruction Error")

    if multi_objective:
        # add complexity objective
        elites["num_cells"] = elites["objectives"].apply(lambda o: o[1])

        sns.lineplot(
            elites,
            x=metric,
            y="num_cells",
            hue="method_name",
            ax=axes["C"],
            legend=False,
        )

        axes["C"].set_xlabel(metric_label)
        axes["C"].set_ylabel("Cell count")

        # hypervolume

        # normalize each dimension to [0, 1] to ensure that
        # hypervolumes are comparable between generations (or methods)
        all_objectives = np.array(df["objectives"].tolist())
        o_min, o_max = (
            np.min(all_objectives, axis=0),
            np.max(all_objectives, axis=0),
        )
        assert (
            np.isfinite(o_min).all()
            and np.isfinite(o_max).all()
            and (o_max - o_min > 0.0).all()
        )
        normalize = lambda o: (o - o_min) / (o_max - o_min)

        reference_point = np.array([1.1, 1.1])

        x = sorted(df[metric].unique())
        fronts = [np.array(df[df[metric] == g]["objectives"].tolist()) for g in x]

        axes["H"].plot(
            x,
            [hypervolume2D(normalize(front), reference_point) for front in fronts],
        )
        axes["H"].set_xlabel(metric_label)
        axes["H"].set_ylabel("Hypervolume")

        # fronts
        for g, c in enumerate(sns.color_palette("viridis", n_colors=len(x))):
            axes["F"].scatter(fronts[g][:, 0], fronts[g][:, 1], color=c)
        axes["F"].set_xlabel("Reconstruction Error")
        axes["F"].set_ylabel("Cell Count")
        fig.colorbar(
            mpl.cm.ScalarMappable(norm=plt.Normalize(x[0], x[-1]), cmap="viridis"),
            ax=axes["F"],
            label=metric_label,
        )

    fig.suptitle(df["method_name"][0])

    plt.show()


def load_image_data(image_path: str, image_max_dim: int):
    """
    Loads an image, downscales if necessary and returns the image data and scaling factor
    """
    im = Image.open(image_path).convert("RGB")

    w, h = im.size
    m = max(w, h)
    if m > image_max_dim:
        scale_factor = image_max_dim / m
        im = ImageOps.cover(im, (int(scale_factor * w), int(scale_factor * h)))
        w, h = im.size
    else:
        scale_factor = 1.0

    # im.show()

    data = np.array(im.get_flattened_data(), dtype=np.uint8)
    return data, w, h, scale_factor


def save_images(
    odir: str | pathlib.Path,
    problem,
    solutions: SolutionSetBase | ArchiveBase | SolutionBase,
    scale_factor: float = 1.0,
    show_objectives: bool = False,
    clean: bool = False,
):
    """
    Renders solutions to images
    """
    odir = pathlib.Path(odir)

    if clean and odir.exists():
        shutil.rmtree(odir)

    odir.mkdir(exist_ok=True, parents=True)

    if isinstance(solutions, SolutionBase):
        solutions = [solutions]

    for i in range(solutions.size()):
        # render image
        data, w, h = problem.image_data(solutions[i], scale=1.0 / scale_factor)
        im = Image.fromarray(data.reshape(h, w, 3), "RGB")
        # im.show()

        # optionally add fitness
        if show_objectives:
            font_size, margin = min(36, h / 20.0), 10
            text = "\n".join(
                f"{o}: {v:{fmt}}"
                for v, (o, fmt) in zip(
                    solutions[i].quality().objectives,
                    [("Reconstruction Error", ".3f"), ("Cell count", "")],
                )
            )
            font = ImageFont.load_default(size=font_size)
            draw = ImageDraw.Draw(im)
            text = dict(
                xy=(margin, 0),
                text=text,
                font=font,
                align="left",
                stroke_width=0.5,
            )
            *_, textheight = draw.textbbox(**text)
            text["xy"] = (margin, h - margin - textheight)
            draw.text(fill="white", stroke_fill="black", **text)

        im.save(odir / f"solution{i:04d}.png")


def animate_evolution(
    filename,
    problem,
    elites,
    generations: list[int],
    scale_factor: float = 1.0,
    dpi: int = 300,
    fps: int = 5,
):
    """
    Turns logged elite solutions into an animation
    """

    def render(generation):
        values = literal_eval(
            elites[elites["generation"] == generation]["discrete"].iloc[0]
        )
        values = np.array(values, dtype=np.uint16)
        s = Solution(problem.archive_fitness().worst(), values)
        data, w, h = problem.image_data(s, scale=1.0 / scale_factor)
        return data.reshape(h, w, 3)

    dims = np.array(render(generations[0]).shape[:2][::-1]) / dpi
    fig, ax = plt.subplots(figsize=dims)
    ax.set_axis_off()
    fig.subplots_adjust(top=1, right=1, bottom=0, left=0, hspace=0, wspace=0)
    ax.margins(0)

    img = ax.imshow(render(generations[0]), animated=True)
    title = ax.text(
        0.05,
        0.05,
        "",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        animated=True,
        color="white",
    )
    title.set_path_effects(
        [patheffects.Stroke(linewidth=1, foreground="black"), patheffects.Normal()]
    )

    def render_frame(i):
        g = generations[min(i, len(generations) - 1)]
        title.set_text(f"Generation {g}")
        img.set_data(render(g))

        return img, title

    delay_seconds = 2
    anim = animation.FuncAnimation(
        fig,
        render_frame,
        init_func=lambda: render_frame(len(generations) - 1),
        frames=len(generations) + delay_seconds * fps,
        interval=1000 / fps,
        repeat=True,
        blit=True,
    )
    anim.save(filename)

    plt.close(fig)
