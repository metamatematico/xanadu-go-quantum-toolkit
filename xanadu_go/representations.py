"""Graph representations derived from Go board states."""

from __future__ import annotations

from typing import List, Tuple

import networkx as nx

from .quantum_graph import QuantumGoGraph


def build_cfg_from_board(board_matrix: List[List[int]], threshold: float = 0.75) -> nx.Graph:
    """Return a Common Fate Graph (CFG) built via QuantumGoGraph contractions."""

    go_graph = QuantumGoGraph(board_matrix, threshold=threshold)
    go_graph.create_cfg()
    cfg = nx.Graph()
    if go_graph.cfg is None:
        return cfg
    for root, data in go_graph.cfg["nodes"].items():
        cfg.add_node(root, **data)
    cfg.add_edges_from(go_graph.cfg["edges"])
    return cfg


def build_weighted_adjacency(board_matrix: List[List[int]]) -> nx.Graph:
    """Create a weighted adjacency graph capturing liberties/influence heuristics."""

    size = len(board_matrix)
    graph = nx.grid_2d_graph(size, size)
    liberties: dict[Tuple[int, int], int] = {}
    for r in range(size):
        for c in range(size):
            color = board_matrix[r][c]
            neighbors = [
                (r + dr, c + dc)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= r + dr < size and 0 <= c + dc < size
            ]
            liberty_count = sum(board_matrix[nr][nc] == 0 for nr, nc in neighbors)
            liberties[(r, c)] = liberty_count
            graph.nodes[(r, c)]["color"] = color
            graph.nodes[(r, c)]["liberties"] = liberty_count
    for (u, v) in graph.edges():
        lib_u = liberties.get(u, 0)
        lib_v = liberties.get(v, 0)
        graph.edges[u, v]["weight"] = lib_u + lib_v + 1
    return graph
