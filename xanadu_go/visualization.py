"""Visualization helpers for Go board states and sequences."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

_COLOR_MAP = {-1: "black", 0: "none", 1: "white"}


def _star_points(size: int) -> List[Tuple[int, int]]:
    if size == 9:
        return [(2, 2), (2, 6), (4, 4), (6, 2), (6, 6)]
    if size == 13:
        return [(3, 3), (3, 9), (6, 6), (9, 3), (9, 9)]
    if size == 19:
        return [(3, 3), (3, 9), (3, 15), (9, 3), (9, 9), (9, 15), (15, 3), (15, 9), (15, 15)]
    return []


def plot_board(
    board: Sequence[Sequence[int]],
    *,
    last_move: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    size = len(board)
    data = np.array(board)
    if data.shape != (size, size):
        raise ValueError("Board must be square")

    show_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(min(8, size / 1.2), min(8, size / 1.2)))
        show_fig = True

    ax.set_aspect("equal")
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(-0.5, size - 0.5)
    ax.invert_yaxis()
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels([chr(ord("A") + i) for i in range(size)])
    ax.set_yticklabels([str(i + 1) for i in range(size)])
    ax.grid(True, color="#b8874a", linewidth=1)

    for (r, c) in _star_points(size):
        ax.scatter(c, r, s=40, color="#5c3b09")

    for r in range(size):
        for c in range(size):
            stone = data[r, c]
            if stone == 0:
                continue
            facecolor = _COLOR_MAP[stone]
            edgecolor = "black"
            ax.add_patch(plt.Circle((c, r), 0.45, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5))

    if last_move is not None:
        lr, lc = last_move
        ax.scatter(lc, lr, s=200, facecolors="none", edgecolors="red", linewidths=2)

    if title:
        ax.set_title(title)

    if show_fig:
        plt.tight_layout()
    return ax


def plot_board_with_analysis(
    board: Sequence[Sequence[int]],
    *,
    last_move: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
    highlight_nodes: Optional[Iterable[Tuple[int, int]]] = None,
    max_cut_nodes: Optional[Iterable[Tuple[int, int]]] = None,
    vertex_cover_nodes: Optional[Iterable[Tuple[int, int]]] = None,
    cut_edges: Optional[Iterable[Tuple[Tuple[int, int], Tuple[int, int]]]] = None,
    annotation: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot a Go board with overlays for analysis results."""

    ax = plot_board(board, last_move=last_move, title=title, ax=ax)

    def _scatter_nodes(nodes: Iterable[Tuple[int, int]], **kwargs):
        coords = [(r, c) for r, c in nodes]
        if not coords:
            return
        xs = [c for _, c in coords]
        ys = [r for r, _ in coords]
        ax.scatter(xs, ys, **kwargs)

    if highlight_nodes:
        _scatter_nodes(
            highlight_nodes,
            s=180,
            facecolors="none",
            edgecolors="cyan",
            linewidths=2,
        )

    if max_cut_nodes:
        _scatter_nodes(
            max_cut_nodes,
            s=260,
            facecolors="none",
            edgecolors="green",
            linewidths=2,
        )

    if vertex_cover_nodes:
        _scatter_nodes(
            vertex_cover_nodes,
            s=220,
            facecolors="none",
            edgecolors="orange",
            linewidths=2,
        )

    if cut_edges:
        for u, v in cut_edges:
            ru, cu = u
            rv, cv = v
            ax.plot(
                [cu, cv],
                [ru, rv],
                color="red",
                linewidth=2,
                alpha=0.8,
            )

    if annotation:
        ax.text(
            0.02,
            0.02,
            annotation,
            transform=ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.75, "boxstyle": "round,pad=0.3"},
        )

    return ax


def plot_board_sequence(
    boards: Iterable[Sequence[Sequence[int]]],
    *,
    last_moves: Optional[Iterable[Optional[Tuple[int, int]]]] = None,
    cols: int = 4,
    figsize: Optional[Tuple[float, float]] = None,
    titles: Optional[Iterable[str]] = None,
) -> plt.Figure:
    boards = list(boards)
    count = len(boards)
    if count == 0:
        raise ValueError("No boards to plot")

    if last_moves is None:
        last_moves = [None] * count
    else:
        last_moves = list(last_moves)
        if len(last_moves) != count:
            raise ValueError("last_moves length must match number of boards")

    if titles is None:
        titles = [f"Move {idx}" for idx in range(count)]
    else:
        titles = list(titles)
        if len(titles) != count:
            raise ValueError("titles length must match number of boards")

    rows = (count + cols - 1) // cols
    if figsize is None:
        figsize = (cols * 3.0, rows * 3.2)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()

    for idx, (board, move, title) in enumerate(zip(boards, last_moves, titles)):
        plot_board(board, last_move=move, title=title, ax=axes[idx])

    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    return fig
