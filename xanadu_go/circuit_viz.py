"""Utilities to visualize QAOA circuits and GBS interferometers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pennylane as qml

from .qaoa import GraphEmbedding


def draw_qaoa_circuit(
    graph: nx.Graph,
    *,
    depth: int = 2,
    max_nodes: Optional[int] = None,
    params: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    device_name: str = "default.qubit",
) -> plt.Figure:
    """Render a QAOA circuit diagram for Max-Cut on the provided graph."""

    embedding = GraphEmbedding(graph, max_nodes=max_nodes)
    cost_h, mixer_h = qml.qaoa.maxcut(embedding.relabelled)
    wires = list(range(embedding.n_wires))
    dev = qml.device(device_name, wires=wires)

    @qml.qnode(dev)
    def circuit(theta):
        betas = theta[:depth]
        gammas = theta[depth:]
        for w in wires:
            qml.Hadamard(w)
        for layer in range(depth):
            qml.qaoa.cost_layer(gammas[layer], cost_h)
            qml.qaoa.mixer_layer(betas[layer], mixer_h)
        return qml.probs(wires=wires)

    if params is None:
        params = np.concatenate([
            0.5 * np.ones(depth),
            1.0 * np.ones(depth),
        ])

    drawer = qml.draw_mpl(circuit)
    fig, ax = drawer(params)
    ax.set_title("Circuito QAOA (Max-Cut)")
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_gbs_interferometer(
    graph: nx.Graph,
    *,
    save_path: Optional[Path] = None,
    title: str = "Interferómetro GBS (matriz de adyacencia)",
    cmap: str = "viridis",
) -> plt.Figure:
    """Visualize the weighted adjacency matrix driving the GBS interferometer."""

    ordered_nodes = list(graph.nodes())
    matrix = nx.to_numpy_array(graph, nodelist=ordered_nodes, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Nodo")
    ax.set_ylabel("Nodo")
    ax.set_xticks(range(len(ordered_nodes)))
    ax.set_yticks(range(len(ordered_nodes)))
    ax.set_xticklabels([str(n) for n in ordered_nodes], rotation=90)
    ax.set_yticklabels([str(n) for n in ordered_nodes])
    fig.colorbar(im, ax=ax, label="Peso")
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig
