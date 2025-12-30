from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy
import seaborn as sns
from matplotlib.patches import Rectangle
from networkx.drawing.nx_agraph import graphviz_layout
from pygom import *

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


def draw_nx(
    graph,
    ax,
    margins=0.2,
    root=None,
    labels=None,
    edgecolors="black",
    node_color="white",
    node_size=1200,
    xscale=1 / 1.75,
    yscale=1 / 3,
    label_offset=(0, 0),
    hide_ticks=True,
    label_kwargs=dict(),
    **kwargs,
):
    pos = graphviz_layout(
        graph,
        # root="0",
        root=root,
        prog="dot",  # , args="ranksep=0,nodesep=0"
    )
    pos = {n: (x * xscale, y * yscale) for n, (x, y) in pos.items()}
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
    if labels is not None:
        lpos = {
            n: (x + label_offset[0], y + label_offset[1]) for n, (x, y) in pos.items()
        }
        nx.draw_networkx_labels(graph, lpos, labels, ax=ax, **label_kwargs)
    ax.set_aspect(1)
    sns.despine(ax=ax, bottom=True, left=True)

    return pos


def draw_lt(Z, n, ax, include_full_fos: bool = False, **kwargs):
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

    draw_nx(
        fos,
        ax,
        root=2 * n - 1,
        labels={n: fos.nodes[n]["label"] for n in fos},
        nodelist=[n for n in fos if fos.nodes[n]["show"]],
        edgelist=[
            e
            for e in fos.edges()
            if fos.nodes[e[0]]["show"] and fos.nodes[e[1]]["show"]
        ],
        edgecolors="none",
        arrows=False,
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
    solution = [r"$+$", r"$\sin$", r"$\sqrt{\ }$", "$x_0$", "$x_2$", "$x_1$", "$x_3$"]
    active = [True, True, True, True, False, True, False]
    expr = r"$\sin(x_0) + \sqrt{x_1}$"

    all_symbols = sorted({s for s in solution})
    s2v = {s: i for i, s in enumerate(all_symbols)}
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

    # axes["E"].text(
    #     0.5, 0.5, expr, ha="center", va="center", transform=axes["E"].transAxes
    # )
    # axes["E"].set_title("Expression")
    # # axes["E"].set_xlabel("Expression")
    # axes["E"].set_axis_off()
    # # sns.despine(ax=axes["E"], left=True, bottom=True)

    sns.heatmap(
        [values],
        annot=[solution],
        fmt="s",
        cmap=cmap,
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

    draw_nx(template_structure, root=0, margins=(0.2, 0.3), ax=axes[1, 0])
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
    example_solution()
    plt.show()
    example_constants()
    plt.show()
    linkage_example()
    plt.show()
