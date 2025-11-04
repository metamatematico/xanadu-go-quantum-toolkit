"""Pipelines that orchestrate Go graph processing, GBS features, and QML models."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import networkx as nx

from .gbs_features import gbs_feature_vector
from .quantum_ml import quantum_kernel
from .representations import build_cfg_from_board, build_weighted_adjacency
from .sgf import SGFGame, load_sgf


def pipeline_cfg_to_gbs_features(board: List[List[int]], *, samples: int = 100) -> Dict[str, float]:
    cfg = build_cfg_from_board(board)
    return gbs_feature_vector(cfg, node_count=max(cfg.number_of_nodes(), 1), samples=samples)


def pipeline_adjacency_to_kernel(
    board_a: List[List[int]],
    board_b: List[List[int]],
) -> float:
    graph_a = build_weighted_adjacency(board_a)
    graph_b = build_weighted_adjacency(board_b)
    vec_a = [data.get("color", 0) for _, data in graph_a.nodes(data=True)]
    vec_b = [data.get("color", 0) for _, data in graph_b.nodes(data=True)]
    wires = max(len(vec_a), len(vec_b))
    return quantum_kernel(vec_a, vec_b, wires=wires)


def load_sgf_board_sequence(path: str, *, threshold: float = 0.75) -> List[nx.Graph]:
    """Load an SGF game and return CFG graphs for each board state."""

    game = load_sgf(path)
    boards = game.board_sequence()
    return [build_cfg_from_board(board, threshold=threshold) for board in boards]


def sgf_pipeline_gbs_features(
    path: str,
    *,
    threshold: float = 0.75,
    node_count: Optional[int] = None,
    samples: int = 100,
) -> List[Dict[str, float]]:
    """Compute GBS features for each state in an SGF game."""

    cfg_sequence = load_sgf_board_sequence(path, threshold=threshold)
    features: List[Dict[str, float]] = []
    for cfg in cfg_sequence:
        count = node_count if node_count is not None else max(cfg.number_of_nodes(), 1)
        features.append(gbs_feature_vector(cfg, node_count=count, samples=samples))
    return features
