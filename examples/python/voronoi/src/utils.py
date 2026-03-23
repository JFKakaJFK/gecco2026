import pathlib
import shutil
from ast import literal_eval

import matplotlib.animation as animation
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pygom import *


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
