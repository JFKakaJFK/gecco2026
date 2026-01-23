from itertools import chain, combinations
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy
import seaborn as sns
from matplotlib.patches import Circle, PathPatch, Rectangle
from matplotlib.path import Path

# from networkx.drawing.nx_agraph import graphviz_layout
from pygom import *
from seaborn.utils import relative_luminance

sns.set_theme(
    context="paper",
    style="ticks",
    font_scale=1.5,
    rc={
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    },
)

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def compute_entropy(
    domain_sizes: np.ndarray,
    values: np.ndarray,
    active: np.ndarray,
    count_inactive: Literal["any_active", "never", "mask", "ignore"] = "any_active",
):
    """
    Computes the pairwise entropy matrix,
    possibly with special handling of conditionally inactive values
    """

    n, d = values.shape

    intron_aware = count_inactive != "ignore" and active is not None

    def entropy(*args):
        counts = np.zeros(
            tuple(domain_sizes[j] + bool(intron_aware) for j in args),
            dtype=np.float64,
        )

        for i in range(n):
            # introns get the same special value
            idx = tuple(
                domain_sizes[j] if intron_aware and not active[i, j] else values[i, j]
                for j in args
            )
            counts[*idx] += 1

        # we normalize w.r.t. introns
        total = (
            n
            if not intron_aware or count_inactive == "mask"
            else (n - counts[*(domain_sizes[j] for j in args)])
        )
        if total <= 0:
            return 0.0
        p = counts / total
        positive = p > 0
        # and don't count full intron pairs
        if intron_aware:
            match count_inactive:
                case "any_active":
                    positive[*(domain_sizes[j] for j in args)] = False
                case "never":
                    for i in args:
                        positive[
                            *(
                                domain_sizes[j] if i == j else slice(domain_sizes[j])
                                for j in args
                            )
                        ] = False
                case "mask":
                    pass
                case other:
                    raise ValueError(f"Unsupported intron handling approach '{other}'")

        assert np.abs(np.sum(p) - 1) < 1e-6, (p, total)
        return np.sum(p * -np.log2(p, where=positive), where=positive)

    # pylint: disable=invalid-name
    H = np.zeros((d, d), dtype=np.float64)
    # entropy matrix
    for i in range(d):
        H[i, i] = entropy(i)

        for j in range(i):
            H[i, j] = H[j, i] = entropy(i, j)
    assert np.isfinite(H).all(), H
    return H


def entropy2MI(H: np.ndarray):
    # pylint: disable=invalid-name
    MI = np.zeros_like(H)
    # https://en.wikipedia.org/wiki/Mutual_information
    for i in range(H.shape[0]):
        for j in range(i):
            MI[i, j] = MI[j, i] = H[i, i] + H[j, j] - H[i, j]
    return MI


def lt2subsets(lt):
    return [set(np.nonzero(subset)[0].tolist()) for subset in lt]


def learn_lt(
    similarity: np.ndarray,
    *,
    include_full_fos: bool = False,
    max_height: int | None = None,
    method="average",
    filter: bool = False,
    tol: float = 1e-6,
):
    """Given the lower triangle of a distance matrix, the linkate tree is learned"""
    d = similarity.shape[0]
    inverted = np.max(similarity) - similarity
    inverted[np.diag_indices_from(similarity)] = 0.0
    dist = scipy.spatial.distance.squareform(inverted)

    # pylint: disable=invalid-name
    Z = scipy.cluster.hierarchy.linkage(dist, method=method)

    subsets = np.empty((2 * d - 2 + int(include_full_fos), d), dtype=np.bool_)
    subsets[:d, :] = np.eye(d, dtype=np.bool_)

    height = np.ones(subsets.shape[0], dtype=np.int16)
    use_subset = np.ones_like(height, dtype=np.bool_)

    for i in range(subsets.shape[0] - d):
        # Z[i-th cluster] ~ [F1, F2, dist]
        height[d + i] = 1 + max(height[int(Z[i, 0])], height[int(Z[i, 1])])
        subsets[d + i, :] = subsets[int(Z[i, 0])] | subsets[int(Z[i, 1])]

        if filter:
            if Z[i, 2] <= tol:  # perfect linkage (filter children)
                # as done in https://arxiv.org/pdf/2109.05259
                # as done in https://homepages.cwi.nl/~bosman/publications/2013_moreconciseand.pdf
                use_subset[int(Z[i, 0])] = False
                use_subset[int(Z[i, 1])] = False

    if max_height is not None and np.max(height) > max_height:
        use_subset = use_subset & (height <= max_height)

    lt = subsets[use_subset]
    return lt2subsets(lt), lt, Z


def custom_tree_layout(graph, root=None, dx: float = 1, dy: float = 0.75):
    if root is None:
        root = list(graph.nodes)[0]

    q = set([root])
    parent = dict()
    info = dict()
    pos = dict()
    while len(q):
        c = q.pop()
        p, nth_child, num_children = parent.get(c, (None, 0, 1))
        pd, pdx = (-1, dx) if p is None else info[p]
        px, py = pos.get(p, (0, dy))
        d = pd + 1

        if num_children < 2:
            xy = px, py - dy
            cdx = pdx
        else:
            cdx = pdx / num_children
            s = pdx / (num_children - 1)
            x = px - pdx / 2 + nth_child * s
            xy = x, py - dy
        info[c] = d, cdx
        pos[c] = xy

        cs = list(graph.adj[c].keys())
        for i, n in enumerate(cs):
            if n not in parent:
                parent[n] = c, i, len(cs)
                q.add(n)

    return pos


def draw_nx(
    graph,
    ax,
    margins=0.2,
    root=None,
    pos=None,
    layout="custom",
    labels=None,
    edgecolors="black",
    node_color="white",
    node_size=1200,
    xscale=1 / 1.75,
    yscale=1 / 3,
    dx=1,
    dy=0.45,
    label_offset=(0, 0),
    hide_ticks=True,
    label_kwargs=dict(),
    **kwargs,
):
    if pos is None:
        # pos = graphviz_layout(
        #     graph,
        #     # root="0",
        #     root=root,
        #     prog="dot",  # , args="ranksep=0,nodesep=0"
        # )
        # pos = {n: (x * xscale, y * yscale) for n, (x, y) in pos.items()}

        pos = custom_tree_layout(graph, root=root, dx=dx, dy=dy)

    nx.draw_networkx(
        graph,
        pos,
        with_labels=labels is None,
        edgecolors=edgecolors,
        node_color=node_color,
        node_size=node_size,
        hide_ticks=hide_ticks,
        margins=margins,
        ax=ax,
        **kwargs,
    )
    if labels is not None and labels:
        lpos = {
            n: (x + label_offset[0], y + label_offset[1]) for n, (x, y) in pos.items()
        }
        nx.draw_networkx_labels(graph, lpos, labels, ax=ax, **label_kwargs)
    ax.set_aspect(1)
    sns.despine(ax=ax, bottom=True, left=True)

    return pos


def draw_lt(
    Z,
    n,
    ax,
    include_full_fos: bool = False,
    label_kwargs=dict(font_size="medium"),
    **kwargs,
):
    fos = nx.DiGraph()
    subsets = []
    for i in range(n):
        fos.add_node(i, label=f"$\\{{{i}\\}}$", show=True)
        subsets.append({i})
    for i in range(Z.shape[0]):
        left = int(Z[i, 0])
        right = int(Z[i, 1])
        merged = {*subsets[left], *subsets[right]}
        subsets.append(merged)
        show = i + 1 < Z.shape[0] or include_full_fos
        fos.add_node(
            n + i, label=f"$\\{{{', '.join(map(str, sorted(merged)))}\\}}$", show=show
        )
        fos.add_edge(n + i, left, show=show)
        fos.add_edge(n + i, right, show=show)

    root = 2 * n - 2
    pos = custom_tree_layout(fos, root=root, dx=2.5, dy=1)
    ymin = min(y for _, y in pos.values())
    pos = {node: (x, ymin) if node < n else (x, y) for node, (x, y) in pos.items()}

    draw_nx(
        fos,
        ax,
        root=root,
        pos=pos,
        labels={n: fos.nodes[n]["label"] for n in fos},
        nodelist=[n for n in fos if fos.nodes[n]["show"]],
        edgelist=[
            e
            for e in fos.edges()
            if fos.nodes[e[0]]["show"] and fos.nodes[e[1]]["show"]
        ],
        edgecolors="none",
        arrows=False,
        label_kwargs=label_kwargs,
        **kwargs,
    )


def draw_measure(H, MI, label, ax, **kwargs):
    if MI is not None:
        triu_indices = np.triu_indices_from(H)

        S = MI.copy()
        S[triu_indices] = H[triu_indices]
    else:
        S = H

    sns.heatmap(
        S,
        square=True,
        annot=True,
        linewidths=4,
        cmap="Blues",
        vmin=0,
        vmax=2,
        ax=ax,
        **kwargs,
    )

    if MI is not None:
        x, y = [], []
        for i in range(1, S.shape[0]):
            x += [i - 1, i, i]
            y += [i, i, i + 1]
        ax.plot(x, y, color="black", linewidth=2)

    ax.set_title(label)


def ctx2graph(ctx):
    template_structure = nx.DiGraph()  # [(0, 1), (0, 2)])
    for node in range(len(ctx.nodes)):
        template_structure.add_node(node)
        p = ctx.parent(node)
        if p is not None:
            template_structure.add_edge(p, node)
    return template_structure


def example_solution():
    solution = [
        r"$+$",
        r"$\sin$",  # r"$\text{sin}$",
        r"$\sqrt{\ }$",
        "$x_0$",
        "$x_2$",
        "$x_1$",
        "$x_3$",
    ]
    active = [True, True, True, True, False, True, False]
    expr = r"$\sin(x_0) + \sqrt{x_1}$"

    all_symbols = sorted({s for s in solution})
    s2v = {s: i for i, s in enumerate(all_symbols)}
    v2s = {i: s for i, s in enumerate(all_symbols)}
    values = [s2v[s] for s in solution]

    cmap = sns.color_palette("Set2", n_colors=len(all_symbols))
    palette = {s: cmap[i] for i, s in enumerate(all_symbols)}

    template = Template([TemplateNode.full_nary(branching_factor=2, depth=2)], [])
    ctx = GPContext(
        num_inputs=sum(1 for s in all_symbols if s.startswith("x")),
        expression_template=template,
        operators=[],
        constant_representation="none",
    )
    template_structure = ctx2graph(ctx)

    for i, s in enumerate(solution):
        template_structure.nodes[i]["label"] = s

    fig, axes = plt.subplot_mosaic(
        """
        TE
        TR
        """,
        figsize=(8, 6),
        gridspec_kw=dict(hspace=0.0, height_ratios=[0.15, 1]),
    )

    axes["E"].set_axis_off()

    node_size = 800
    pos = draw_nx(
        template_structure,
        axes["T"],
        labels=False,  # {n: template_structure.nodes[n]["label"] for n in template_structure},
        alpha=[1.0 if active[n] else 0.5 for n in template_structure.nodes],
        node_color=[
            palette[template_structure.nodes[n]["label"]]
            for n in template_structure.nodes
        ],
        xscale=1,
        yscale=1,
        label_offset=(0, 0),
        node_size=node_size,
        font_color={
            n: ".15"
            if relative_luminance(palette[template_structure.nodes[n]["label"]]) > 0.408
            else "w"
            for n in template_structure
        },
    )
    pos = draw_nx(
        template_structure,
        axes["T"],
        labels=False,
        xscale=1,
        yscale=1,
        node_size=node_size,
        node_color="none",
    )
    for n in template_structure:
        if not active[n]:
            axes["T"].add_patch(
                Circle(
                    pos[n],
                    # radius=27.5,  # np.sqrt(node_size / np.pi),
                    radius=0.16,  # np.sqrt(node_size / np.pi),
                    lw=0,
                    alpha=0.2,  # 5,
                    color="black",
                    fill=False,
                    hatch_linewidth=4,
                    hatch="//",
                    zorder=10,
                )
            )

    nx.draw_networkx_labels(
        template_structure,
        pos,
        {n: template_structure.nodes[n]["label"] for n in template_structure},
        font_color={
            n: ".15"
            if relative_luminance(palette[template_structure.nodes[n]["label"]]) > 0.408
            else "w"
            for n in template_structure
        },
        ax=axes["T"],
    )
    nx.draw_networkx_labels(
        template_structure,
        # {n: (x + 35, y - 15) for n, (x, y) in pos.items()},
        {n: (x + 0.19, y - 0.075) for n, (x, y) in pos.items()},
        {n: f"${{}}_{n}$" for n in template_structure},
        ax=axes["T"],
    )
    axes["T"].set_xlabel("Template Structure")

    # axes["E"].text(
    #     0.5, 0.5, expr, ha="center", va="center", transform=axes["E"].transAxes
    # )
    # axes["E"].set_title("Expression")
    # # axes["E"].set_xlabel("Expression")
    # axes["E"].set_axis_off()
    # # sns.despine(ax=axes["E"], left=True, bottom=True)

    sns.heatmap(
        [values],
        mask=np.array([active], dtype=np.bool_),
        annot=[solution],
        fmt="s",
        cmap=[palette[v2s[v]] for v, a in zip(values, active) if not a],
        square=True,
        alpha=0.5,
        ax=axes["R"],
        cbar=False,
    )
    sns.heatmap(
        [values],
        mask=~np.array([active], dtype=np.bool_),
        annot=[solution],
        fmt="s",
        cmap=[palette[v2s[v]] for v, a in zip(values, active) if a],
        square=True,
        ax=axes["R"],
        cbar=False,
    )
    for i, _ in enumerate(solution):
        if not active[i]:
            axes["R"].add_patch(
                Rectangle(
                    (i, 0),
                    1,
                    1,
                    lw=0,
                    alpha=0.2,  # 5,
                    color="black",
                    fill=False,
                    hatch_linewidth=4,
                    hatch="//",
                )
            )
    axes["R"].set_yticks([])
    axes["R"].set_title("Expression Semantics\n" + expr + "\n")
    axes["R"].set_xlabel("Internal Representation")

    fig.align_labels()
    # fig.align_titles()

    fig.savefig("template_example.pdf", dpi=600, transparent=True, bbox_inches="tight")


def example_node_proximity():
    d = 2
    template = Template([TemplateNode.full_nary(branching_factor=2, depth=d)], [])
    ctx = GPContext(
        num_inputs=1,
        expression_template=template,
        operators=[],
        constant_representation="none",
    )
    template_structure = ctx2graph(ctx)

    node_proximity = ctx.normalized_node_proximity()

    fig, axes = plt.subplot_mosaic(
        """
        TN
        """,
        figsize=(8, 4),
        gridspec_kw=dict(width_ratios=[1.75, 1]),
    )

    # axes["T"].set_axis_off()

    pos = draw_nx(
        template_structure,
        axes["T"],
        # labels={n: template_structure.nodes[n]["label"] for n in template_structure},
        xscale=1,
        yscale=1,
        # label_offset=(0, 0),
        node_size=1200,
        margins=0.1,
    )
    # nx.draw_networkx_labels(
    #     template_structure,
    #     {n: (x + 35, y - 15) for n, (x, y) in pos.items()},
    #     {n: f"${{}}_{n}$" for n in template_structure},
    #     ax=axes["T"],
    # )
    axes["T"].set_xlabel("Template")

    S = node_proximity.copy()
    S[np.triu_indices_from(S)] = ((S - 1.0) * -(1 + 2 * d))[np.triu_indices_from(S)]

    sns.heatmap(
        node_proximity,
        annot=S,
        mask=np.eye(
            node_proximity.shape[0], dtype=np.bool_
        ),  # np.zeros_like(node_proximity, dtype=np.bool_) + np.triu(np.ones_like(node_proximity, dtype=np.bool_)),
        # fmt="s",
        cmap="Blues",
        vmin=0,
        vmax=1,
        square=True,
        annot_kws=dict(fontsize="x-small"),
        ax=axes["N"],
        cbar=False,
    )
    axes["N"].set_title("Node Distance")
    axes["N"].set_xlabel("Node Proximity")

    fig.align_labels()
    # fig.align_titles()

    fig.savefig(
        "node_proximity_example.pdf", dpi=600, transparent=True, bbox_inches="tight"
    )


def example_peter_proximity():
    d = 2
    template = Template([TemplateNode.full_nary(branching_factor=2, depth=d)], [])
    ctx = GPContext(
        num_inputs=1,
        expression_template=template,
        operators=[],
        constant_representation="none",
    )
    template_structure = ctx2graph(ctx)

    node_proximity = ctx.normalized_node_proximity()
    vig_proximity = ctx.normalized_w_vig()
    peter_proximity = ctx.subtree_co_occurrences()

    fig, axes = plt.subplot_mosaic(
        """
        TNP
        """,
        figsize=(10, 4),
        gridspec_kw=dict(width_ratios=[1, 0.75, 0.75], wspace=0.25),
    )

    # axes["T"].set_axis_off()

    pos = draw_nx(
        template_structure,
        axes["T"],
        # labels={n: template_structure.nodes[n]["label"] for n in template_structure},
        xscale=1,
        yscale=1,
        # label_offset=(0, 0),
        node_size=800,
        margins=0.15,  # 75,
    )

    x0, y0 = pos[0]
    x1, y1 = pos[1]
    x2, y2 = pos[2]
    x3, y3 = pos[3]
    x4, y4 = pos[4]
    x5, y5 = pos[5]
    x6, y6 = pos[6]
    xy = np.array([list(pos[i]) for i in range(7)])
    v = np.array([[(xy[j] - xy[i]).tolist() for j in range(7)] for i in range(7)])

    def vl(v_):
        return np.sqrt(np.dot(v_, v_))

    def n(v_):
        return v_ / vl(v_)

    def rot(v_, deg):
        rad = deg / 180.0 * np.pi
        return np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]]) @ v_

    def point(xy, marker="x", s=100, **kwargs):
        axes["T"].scatter([xy[0]], [xy[1]], marker=marker, s=s, **kwargs)

    def vec(xy, v_, pt_kwargs=None, **kwargs):
        if pt_kwargs is None:
            pt_kwargs = {}
        if "color" not in pt_kwargs:
            pt_kwargs["color"] = kwargs.get("color")
        point(xy, **pt_kwargs)
        xy2 = xy + v_
        axes["T"].plot([xy[0], xy2[0]], [xy[1], xy2[1]], **kwargs)

    def shortest_angle(v0, v1):
        n0, n1 = n(v0), n(v1)
        return np.arccos(np.dot(n0, n1)) / np.pi * 180

    def intersect(xy0, v0, xy1, v1):
        xy01 = xy0 + v0
        xy11 = xy1 + v1

        x1, y1 = xy0
        x2, y2 = xy01
        x3, y3 = xy1
        x4, y4 = xy11

        a = x1 * y2 - y1 * x2
        b = x3 * y4 - y3 * x4
        d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        assert np.abs(d) > 1e-6
        xyi = (a * (xy1 - xy11) - b * (xy0 - xy01)) / d

        # vec(xy0, v0, color="red")
        # vec(xy1, v1, color="blue")
        # point(xyi, color="orange")
        return xyi

    X, Y = np.array([1.0, 0.0]), np.array([0.0, 1.0])

    def arc_between(xy0, xy1, cxy):
        txy0, txy1 = xy0 - cxy, xy1 - cxy

        theta1, theta2 = shortest_angle(X, txy0), shortest_angle(X, txy1)
        if txy0[1] < 0.0:
            theta1 = 360.0 - theta1
        if txy1[1] < 0.0:
            theta2 = 360.0 - theta2

        flip = theta1 > theta2
        if flip:
            theta1, theta2 = theta2, theta1

        if theta2 - theta1 > (360.0 - theta2) + theta1:
            flip = not flip
            theta1, theta2 = theta2, theta1
        arc = Path.arc(theta1, theta2)

        r, r1 = vl(xy0 - cxy), vl(xy1 - cxy)
        assert np.abs(r - r1) < 1e-6, (r, r1)

        yield Path.MOVETO, xy1 if flip else xy0
        for c, v in zip(arc.codes[1:-1], arc.vertices[1:-1]):
            yield c, v * r + cxy
        yield Path.LINETO, xy0 if flip else xy1
        yield Path.MOVETO, xy1

    dz = 0.175
    f1_lt = xy[1] + n(rot(v[1, 3], -90.0)) * dz
    f1_rt = xy[1] + n(rot(v[1, 4], 90.0)) * dz
    f1_rb = xy[4] + n(rot(v[4, 1], -90.0)) * dz
    f1_br = xy[4] + n(rot(v[4, 3], 90.0)) * dz
    f1_lb = xy[3] + n(rot(v[3, 1], 90.0)) * dz
    f1_bl = xy[3] + n(rot(v[3, 4], -90.0)) * dz

    path_f1 = [
        (Path.MOVETO, f1_lb),
        (Path.LINETO, f1_lt),
        *arc_between(f1_lt, f1_rt, xy[1]),
        (Path.LINETO, f1_rt),
        #
        (Path.LINETO, f1_rb),
        *arc_between(f1_rb, f1_br, xy[4]),
        (Path.LINETO, f1_br),
        #
        (Path.LINETO, f1_bl),
        *arc_between(f1_bl, f1_lb, xy[3]),
        (Path.LINETO, f1_lb),
        #
        (Path.CLOSEPOLY, f1_lb),
    ]

    f2_lt = xy[2] + n(rot(v[2, 5], -90.0)) * dz
    f2_rt = xy[2] + n(rot(v[2, 6], 90.0)) * dz
    f2_rb = xy[6] + n(rot(v[6, 2], -90.0)) * dz
    f2_br = xy[6] + n(rot(v[6, 3], 90.0)) * dz
    f2_lb = xy[5] + n(rot(v[5, 2], 90.0)) * dz
    f2_bl = xy[5] + n(rot(v[5, 6], -90.0)) * dz
    path_f2 = [
        (Path.MOVETO, f2_lb),
        (Path.LINETO, f2_lt),
        *arc_between(f2_lt, f2_rt, xy[2]),
        (Path.LINETO, f2_rt),
        #
        (Path.LINETO, f2_rb),
        *arc_between(f2_rb, f2_br, xy[6]),
        (Path.LINETO, f2_br),
        #
        (Path.LINETO, f2_bl),
        *arc_between(f2_bl, f2_lb, xy[5]),
        (Path.LINETO, f2_lb),
        #
        (Path.CLOSEPOLY, f2_lb),
    ]

    dz = 0.2

    f0_lt = xy[0] + n(rot(v[0, 1], -90.0)) * dz
    f0_rt = xy[0] + n(rot(v[0, 2], 90.0)) * dz
    f0_rm = xy[2] + n(rot(v[2, 0], -90.0)) * dz
    f0_mr = xy[2] + n(rot(v[2, 3], 90.0)) * dz
    f0_lm = xy[1] + n(rot(v[1, 0], 90.0)) * dz
    f0_ml = xy[1] + n(rot(v[1, 2], -90.0)) * dz

    f1_lt = xy[1] + n(rot(v[1, 3], -90.0)) * dz
    f1_rt = xy[1] + n(rot(v[1, 4], 90.0)) * dz
    f1_rb = xy[4] + n(rot(v[4, 1], -90.0)) * dz
    f1_br = xy[4] + n(rot(v[4, 3], 90.0)) * dz
    f1_lb = xy[3] + n(rot(v[3, 1], 90.0)) * dz
    f1_bl = xy[3] + n(rot(v[3, 4], -90.0)) * dz

    f2_lt = xy[2] + n(rot(v[2, 5], -90.0)) * dz
    f2_rt = xy[2] + n(rot(v[2, 6], 90.0)) * dz
    f2_rb = xy[6] + n(rot(v[6, 2], -90.0)) * dz
    f2_br = xy[6] + n(rot(v[6, 3], 90.0)) * dz
    f2_lb = xy[5] + n(rot(v[5, 2], 90.0)) * dz
    f2_bl = xy[5] + n(rot(v[5, 6], -90.0)) * dz
    path_f0 = [
        (Path.MOVETO, f0_lm),
        (Path.LINETO, f0_lt),
        *arc_between(f0_lt, f0_rt, xy[0]),
        (Path.LINETO, f0_rt),
        #
        (Path.LINETO, f0_rm),
        *arc_between(f0_rm, f2_rt, xy[2]),
        (Path.LINETO, f2_rt),
        #
        (Path.LINETO, f2_rb),
        *arc_between(f2_rb, f2_br, xy[6]),
        (Path.LINETO, f2_br),
        #
        (Path.LINETO, f1_bl),
        *arc_between(f1_bl, f1_lb, xy[3]),
        (Path.LINETO, f1_lb),
        #
        (Path.LINETO, f1_lt),
        *arc_between(f1_lt, f0_lm, xy[1]),
        (Path.LINETO, f0_lm),
        #
        (Path.CLOSEPOLY, f0_lm),
    ]

    path_kwargs = dict(
        edgecolor="k",
        linewidth=1.0,
        ls="--",
        facecolor="none",
        alpha=0.25,
    )

    def draw_path(path, **kwargs):
        kw = {**path_kwargs}
        for k, v in kwargs.items():
            kw[k] = v
        codes, verts = zip(*path)
        axes["T"].add_patch(PathPatch(Path(verts, codes), **kw))

    highglight_subfns = False
    if highglight_subfns:
        draw_path(path_f1)
        draw_path(path_f2)
        draw_path(path_f0)  # , ls="-.")

    # nx.draw_networkx_labels(
    #     template_structure,
    #     {n: (x + 35, y - 15) for n, (x, y) in pos.items()},
    #     {n: f"${{}}_{n}$" for n in template_structure},
    #     ax=axes["T"],
    # )
    axes["T"].set_xlabel("Template")

    axes["T"].set_aspect(1)

    sns.heatmap(
        node_proximity,
        annot=node_proximity,
        mask=np.eye(node_proximity.shape[0], dtype=np.bool_),
        cmap="Blues",
        vmin=0,
        vmax=1,
        square=True,
        annot_kws=dict(fontsize="xx-small"),
        ax=axes["N"],
        cbar=False,
    )
    axes["N"].set_xlabel("Node Proximity")
    axes["N"].tick_params(labelsize="x-small", pad=2, length=2)

    sns.heatmap(
        peter_proximity,
        annot=peter_proximity,
        mask=np.eye(peter_proximity.shape[0], dtype=np.bool_),
        cmap="Blues",
        vmin=0,
        vmax=3,
        square=True,
        annot_kws=dict(fontsize="xx-small"),
        ax=axes["P"],
        cbar=False,
    )
    axes["P"].set_xlabel("#Common Subfunctions")
    axes["P"].tick_params(labelsize="x-small", pad=2, length=2)

    # S = vig_proximity.copy()
    # S[np.triu_indices_from(S)] = peter_proximity[np.triu_indices_from(S)]

    # # S = peter_proximity.copy()

    # A = peter_proximity.copy()
    # A[np.tril_indices_from(A)] = 1.0 / vig_proximity[np.tril_indices_from(A)]

    # mask = np.eye(vig_proximity.shape[0], dtype=np.bool_)
    # m1 = mask.copy()
    # m1[np.tril_indices_from(A)] = np.ones_like(A, dtype=np.bool_)[
    #     np.tril_indices_from(A)
    # ]
    # sns.heatmap(
    #     # 3 - A,
    #     A,
    #     annot=A,
    #     mask=m1,  # np.zeros_like(node_proximity, dtype=np.bool_) + np.triu(np.ones_like(node_proximity, dtype=np.bool_)),
    #     # fmt="s",
    #     cmap="Blues",
    #     vmin=0,
    #     vmax=2,
    #     # cmap="Blues_r",
    #     # vmin=0,
    #     # vmax=4,
    #     square=True,
    #     annot_kws=dict(fontsize="x-small"),
    #     ax=axes["N"],
    #     cbar=False,
    # )
    # m2 = mask.copy()
    # m2[np.triu_indices_from(A)] = np.ones_like(A, dtype=np.bool_)[
    #     np.triu_indices_from(A)
    # ]
    # sns.heatmap(
    #     A,
    #     annot=np.array(
    #         [
    #             [
    #                 rf"${{}}^1\!/\!{{}}_{{{str(A[i, j])[0]}}}$"
    #                 # str(A[i, j])[0]
    #                 for j in range(A.shape[1])
    #             ]
    #             for i in range(A.shape[0])
    #         ]
    #     ),
    #     mask=m2,  # np.zeros_like(node_proximity, dtype=np.bool_) + np.triu(np.ones_like(node_proximity, dtype=np.bool_)),
    #     fmt="s",
    #     cmap="Blues_r",
    #     vmin=1,
    #     vmax=4,
    #     square=True,
    #     annot_kws=dict(fontsize="x-small"),
    #     ax=axes["N"],
    #     cbar=False,
    # )
    # axes["N"].set_title("#Common Subfunctions")
    # axes["N"].set_xlabel("Node Proximity")

    fig.align_labels()
    # fig.align_titles()

    fig.savefig(
        "node_proximity_example_both.pdf",
        dpi=600,
        transparent=True,
        bbox_inches="tight",
    )


def example_node_proximity2():
    d = 2
    template = Template([TemplateNode.full_nary(branching_factor=2, depth=d)], [])
    ctx = GPContext(
        num_inputs=1,
        expression_template=template,
        operators=[],
        constant_representation="none",
    )
    template_structure = ctx2graph(ctx)

    node_proximity = ctx.normalized_node_proximity()

    S = node_proximity.copy()
    S[np.triu_indices_from(S)] = ((S - 1.0) * -(1 + 2 * d))[np.triu_indices_from(S)]

    nodes = list(range(len(template_structure.nodes)))
    co_occurrences = np.zeros((len(nodes), len(nodes)))
    intermediate_nodes = {
        (0, 1): {},
        (0, 2): {},
        (0, 3): {1},
        (0, 4): {1},
        (0, 5): {2},
        (0, 6): {2},
        #
        (1, 2): {0},
        (1, 3): {},
        (1, 4): {},
        (1, 5): {0, 2},
        (1, 6): {0, 2},
        #
        (2, 3): {0, 1},
        (2, 4): {0, 1},
        (2, 5): {},
        (2, 6): {},
        #
        (3, 4): {1},
        (3, 5): {0, 1, 2},
        (3, 6): {0, 1, 2},
        #
        (4, 5): {0, 1, 2},
        (4, 6): {0, 1, 2},
        #
        (5, 6): {2},
    }
    for combination in chain.from_iterable(
        combinations(nodes, l) for l in range(2, len(nodes) + 1)
    ):
        c = set(combination)
        for i in nodes:
            if i in c:
                for j in nodes[:i]:
                    if j in c:
                        # valid if all intermediate nodes are in c as well...
                        if all(n in c for n in intermediate_nodes[j, i]):
                            co_occurrences[j, i] += 1

    show_common_subsets = False
    if show_common_subsets:
        S[np.triu_indices_from(S)] = co_occurrences[np.triu_indices_from(S)]

    fig, axes = plt.subplot_mosaic(
        """
        TN
        """,
        figsize=(8, 4),
        gridspec_kw=dict(width_ratios=[1.4, 1]),
    )

    # axes["T"].set_axis_off()

    pos = custom_tree_layout(template_structure, root=0, dy=0.6)
    draw_nx(
        template_structure,
        axes["T"],
        pos=pos,
        # labels={n: template_structure.nodes[n]["label"] for n in template_structure},
        xscale=1,
        yscale=1,
        # label_offset=(0, 0),
        node_size=1200,
        margins=(0.1, 0.075),
    )
    # nx.draw_networkx_labels(
    #     template_structure,
    #     {n: (x + 35, y - 15) for n, (x, y) in pos.items()},
    #     {n: f"${{}}_{n}$" for n in template_structure},
    #     ax=axes["T"],
    # )

    ne = 0
    dset = [[] for _ in range(1 + int(np.max(S[np.triu_indices_from(S)])))]
    for i in range(ctx.num_discrete):
        for j in range(i):
            dist = int(S[j, i])
            if dist > 1:  # and i == 4 or j == 4:
                template_structure.add_edge(i, j, dist=dist)
                dset[dist].append((i, j))
                ne += 1
    print("number of additional edges:", ne)

    alpha = 0.4

    def draw_edges(edges, ls="-", cs="arc3", alpha=alpha, width=1.0):
        nx.draw_networkx_edges(
            template_structure,
            pos,
            edgelist=edges,
            style=ls,
            alpha=alpha,
            width=width,  # 4 / d,
            # arrows=False,
            arrowstyle="-",
            connectionstyle=cs,
            ax=axes["T"],
        )

    styles = [
        "-",
        (0, (8, 4)),  # "--",
        (0, (4, 4)),  # "-.",
        (0, (1, 4)),  # ":",
    ]
    # for l, (edges, ls) in enumerate(
    #     zip(  #
    #         dset[2:], ["--", "-.", ":"]
    #     )
    # ):
    #     d = l + 2
    #     draw_edges(edges, ls=ls)

    # dist = 2
    w = 1
    ls = styles[1]
    draw_edges([(0, 4), (0, 5), (1, 2), (3, 4), (5, 6)], ls=ls, width=w)
    draw_edges([(0, 3)], ls=ls, cs="arc3,rad=0.45", width=w)
    draw_edges([(0, 6)], ls=ls, cs="arc3,rad=-0.45", width=w)

    # dist = 3
    w = 1
    ls = styles[2]
    draw_edges([(1, 5), (1, 6), (2, 3), (2, 4)], ls=ls, width=w)

    # dist = 4
    w = 1
    ls = styles[3]
    draw_edges([(4, 5)], ls=ls, width=w)
    draw_edges([(3, 5), (3, 6), (4, 6)], ls=ls, cs="arc3,rad=0.45", width=w)

    axes["T"].set_xlabel("Template")

    handles, labels = [], []
    for d, ls in zip([1, 2, 3, 4], styles):
        handles.append(
            axes["T"].plot(
                [], [], c="k", ls=ls, alpha=alpha if d > 1 else 1, label="d"
            )[0]
        )
        labels.append(str(d))
    axes["T"].legend(
        handles,
        labels,
        title="Distance",
        # title_fontproperties=dict(title_fontsize="small"),
        frameon=False,
        bbox_to_anchor=(0.075, 0.35),
        # ncols=4,
        title_fontsize="small",
        fontsize="x-small",
    )
    # axes["T"].margins(y=0, tight=True)

    sns.heatmap(
        node_proximity,
        annot=S,
        mask=np.eye(
            node_proximity.shape[0], dtype=np.bool_
        ),  # np.zeros_like(node_proximity, dtype=np.bool_) + np.triu(np.ones_like(node_proximity, dtype=np.bool_)),
        # fmt="s",
        cmap="Blues",
        vmin=0,
        vmax=1,
        square=True,
        annot_kws=dict(fontsize="x-small"),
        ax=axes["N"],
        cbar=False,
    )
    if show_common_subsets:
        axes["N"].set_title("Common Subset Count")
    else:
        axes["N"].set_title("Node Distance")

    axes["N"].set_xlabel("Node Proximity")

    fig.align_labels()
    # fig.align_titles()

    fig.savefig(
        "node_proximity_example2.pdf", dpi=600, transparent=True, bbox_inches="tight"
    )

    print(co_occurrences)
    print(S)


def example_constants():
    solution = [r"$+$", r"$\times$", r"$2.7$", "$x_0$", "$3.1$", "$x_2$", "$3.1$"]
    active = [True, True, True, True, True, False, False]

    solution_pool = [r"$+$", r"$\times$", r"$c_1$", "$x_0$", "$c_2$", "$x_2$", "$c_2$"]

    pool = [4.2, 3.1, 2.7]
    pool_active = [False, True, True]

    all_symbols = sorted({s for s in solution})
    s2v = {s: i for i, s in enumerate(all_symbols)}
    values = [s2v[s] for s in solution]

    cmap = sns.color_palette("Set2", n_colors=len(all_symbols) + 1)

    template = Template([TemplateNode.full_nary(branching_factor=2, depth=2)], [])
    ctx = GPContext(
        num_inputs=sum(1 for s in all_symbols if s.startswith("x")),
        expression_template=template,
        operators=[],
        constant_representation="none",
    )
    template_structure = ctx2graph(ctx)

    for i, s in enumerate(solution):
        template_structure.nodes[i]["label"] = s

    fig, axes = plt.subplot_mosaic(
        """
        TE
        TP
        """,
        figsize=(8, 4),
        gridspec_kw=dict(hspace=0.0, height_ratios=[1, 2]),
    )

    pos = draw_nx(
        template_structure,
        axes["T"],
        labels={n: template_structure.nodes[n]["label"] for n in template_structure},
        xscale=1,
        yscale=1,
        label_offset=(0, 0),
        node_size=800,
    )
    nx.draw_networkx_labels(
        template_structure,
        {n: (x + 35, y - 15) for n, (x, y) in pos.items()},
        {n: f"${{}}_{n}$" for n in template_structure},
        ax=axes["T"],
    )
    axes["T"].set_xlabel("Template Structure")

    print(values, solution)

    sns.heatmap(
        [values],
        annot=[solution],
        fmt="s",
        cmap=cmap,
        square=True,
        ax=axes["E"],
        cbar=False,
    )
    for i, _ in enumerate(solution):
        if not active[i]:
            axes["E"].add_patch(
                Rectangle(
                    (i, 0),
                    1,
                    1,
                    lw=0,
                    alpha=0.2,  # 5,
                    color="black",
                    fill=False,
                    hatch_linewidth=4,
                    hatch="//",
                )
            )
    axes["E"].set_yticks([])
    axes["E"].set_xlabel("ERC Representation")

    pad = len(active) - len(pool_active)
    p_values = [values, [0 for _ in range(len(values))]]
    p_annot = [solution_pool, pool + [0 for _ in range(pad)]]
    p_mask = ~np.array(
        [
            [True for _ in active],
            [True for _ in pool_active] + [False for _ in range(pad)],
        ]
    )
    print(p_values)
    print(p_annot)
    print(p_mask)
    sns.heatmap(
        p_values,
        annot=p_annot,
        mask=p_mask,
        fmt="s",
        cmap=cmap,
        square=True,
        linewidths=(0, 4),
        ax=axes["P"],
        cbar=False,
    )
    for x, y in [(i, 0) for i, a in enumerate(active) if not a] + [
        (i, 1) for i, a in enumerate(pool_active) if not a
    ]:
        axes["P"].add_patch(
            Rectangle(
                (x, y),
                1,
                1,
                lw=0,
                alpha=0.2,  # 5,
                color="black",
                fill=False,
                hatch_linewidth=4,
                hatch="//",
            )
        )
    axes["P"].set_yticks([])
    axes["P"].set_xlabel("Pool Representation")

    fig.align_labels()
    # fig.align_titles()

    fig.savefig("constants_example.pdf", dpi=600, transparent=True, bbox_inches="tight")


def linkage_example():
    solutions, active = (
        [
            [r"$\sin$", "$x_0$", "$x_1$"],
            ["$x_0$", "$x_1$", "$x_0$"],
            [r"$\sin$", "$x_1$", "$x_0$"],
            ["$+$", "$x_0$", "$x_1$"],
        ],
        np.array(
            [
                [True, True, False],
                [True, False, False],
                [True, True, False],
                [True, True, True],
            ],
            dtype=np.bool_,
        ),
    )

    all_symbols = sorted({s for solution in solutions for s in solution})
    symbol2value = {s: i for i, s in enumerate(all_symbols)}

    template = Template([TemplateNode.full_nary(branching_factor=2, depth=1)], [])
    ctx = GPContext(
        num_inputs=sum(1 for s in all_symbols if s.startswith("x")),
        expression_template=template,
        operators=[OpAdd(), OpSin()],
        constant_representation="none",
    )
    node_proximity = np.array(ctx.normalized_node_proximity().tolist())
    template_structure = ctx2graph(ctx)

    n = len(template_structure.nodes)

    solutions, values = (
        np.array(solutions),
        np.array(
            [[symbol2value[symbol] for symbol in s] for s in solutions], dtype=np.uint8
        ),
    )
    domain_sizes = np.array(np.max(values, axis=0) + 1, dtype=np.uint8)

    H = compute_entropy(domain_sizes, values, active, "ignore")
    MI = entropy2MI(H)

    sets, lt, Z = learn_lt(MI)

    H_masked = compute_entropy(domain_sizes, values, active, "mask")
    MI_masked = entropy2MI(H_masked)

    sets_masked, lt_masked, Z_masked = learn_lt(MI_masked)

    sets_node, lt_node, Z_node = learn_lt(node_proximity)

    nrows = 2
    ncols = 3
    # ncols = 4

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4 * ncols, 3 * nrows),
        gridspec_kw=dict(hspace=0.4, height_ratios=[1, 0.5]),
    )

    # population
    cmap = "Set2"
    cmap = sns.color_palette("Set2", n_colors=np.max(values) + 1)
    pop_ax = axes[0, 0]
    sns.heatmap(
        values,
        # mask=active,
        annot=solutions,
        fmt="s",
        cmap=cmap,
        square=True,
        # alpha=0.75,
        ax=pop_ax,
        cbar=False,
    )
    sns.heatmap(
        values,
        mask=~active,
        annot=solutions,
        fmt="s",
        cmap=cmap,
        square=True,
        ax=pop_ax,
        cbar=False,
    )
    for i in range(solutions.shape[0]):
        for j in range(n):
            if not active[i, j]:
                pop_ax.add_patch(
                    Rectangle(
                        (j, i),
                        1,
                        1,
                        lw=0,
                        alpha=0.2,  # 5,
                        color="black",
                        fill=False,
                        hatch_linewidth=4,
                        hatch="//",
                    )
                )
    pop_ax.set_yticks([])
    pop_ax.set_title("Solutions")

    draw_measure(H, MI, "$MI$", axes[0, 1], cbar=False)
    draw_measure(H_masked, MI_masked, "$MI_{masked}$", axes[0, 2], cbar=False)
    if ncols > 3:
        draw_measure(node_proximity, None, "Node", axes[0, 3], cbar=False)

    draw_nx(
        template_structure,
        root=0,
        margins=(0.2, 0.3),
        dx=0.7,
        ax=axes[1, 0],
    )
    axes[1, 0].set_title("Template")

    # scipy.cluster.hierarchy.dendrogram(Z, ax=axes[1, 1])
    # scipy.cluster.hierarchy.dendrogram(Z_masked, ax=axes[1, 2])

    indices = np.tile(np.arange(n), (lt_masked.shape[0], 1))
    cmap = "tab10"
    cmap = sns.color_palette("tab10", n_colors=n)

    as_tree = True

    if as_tree:
        draw_lt(
            Z,
            n,
            axes[1, 1],
        )
    else:
        sns.heatmap(
            indices,
            mask=~lt,
            annot=indices,
            fmt="d",
            cmap=cmap,
            square=True,
            ax=axes[1, 1],
            cbar=False,
        )
        axes[1, 1].set_yticks([])
    axes[1, 1].set_title(r"$\downarrow$")  # "LT $MI$")

    if as_tree:
        draw_lt(
            Z_masked,
            n,
            axes[1, 2],
        )
    else:
        sns.heatmap(
            indices,
            mask=~lt_masked,
            annot=indices,
            fmt="d",
            cmap=cmap,
            square=True,
            ax=axes[1, 2],
            cbar=False,
        )
        axes[1, 2].set_yticks([])
    axes[1, 2].set_title(r"$\downarrow$")  # "LT $MI_{masked}$")

    if ncols > 3:
        if as_tree:
            draw_lt(
                Z_node,
                n,
                axes[1, 3],
            )
        else:
            sns.heatmap(
                indices,
                mask=~lt_node,
                annot=indices,
                fmt="d",
                cmap=cmap,
                square=True,
                ax=axes[1, 3],
                cbar=False,
            )
            axes[1, 3].set_yticks([])
        axes[1, 3].set_title("LT Node")

    fig.align_labels()
    fig.align_titles()

    fig.savefig("linkage_example.pdf", dpi=600, transparent=True, bbox_inches="tight")


if __name__ == "__main__":
    # example_solution()
    # plt.show()
    # example_node_proximity()
    # plt.show()
    # example_node_proximity2()
    example_peter_proximity()
    plt.show()
    # example_constants()
    # plt.show()
    # linkage_example()
    # plt.show()
