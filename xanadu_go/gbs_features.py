"""Feature extraction utilities leveraging Gaussian Boson Sampling (GBS)."""

from __future__ import annotations

from typing import Dict, List, Tuple

import networkx as nx

try:  # Optional dependency (Strawberry Fields / PennyLane photonic backends)
    import strawberryfields as sf
    from strawberryfields.apps import subgraph
except ImportError:  # pragma: no cover - fallback when photonic stack is unavailable
    sf = None
    subgraph = None


def is_gbs_available() -> bool:
    """Return True if Strawberry Fields GBS utilities are importable."""

    return subgraph is not None


def sample_dense_subgraphs(
    graph: nx.Graph,
    node_count: int,
    samples: int = 100,
    seed: int | None = None,
) -> List[Tuple[int, ...]]:
    """Sample dense subgraphs using Strawberry Fields GBS approximations.

    If the photonic stack is unavailable, an empty list is returned.
    """

    if not is_gbs_available():
        return []
    state = subgraph.density_matrix(graph)
    rng = seed if seed is not None else 0
    return list(subgraph.sample(state, samples=samples, max_count=node_count, random_seed=rng))


def gbs_feature_vector(graph: nx.Graph, node_count: int, samples: int = 100) -> Dict[str, float]:
    """Compute simple statistics from GBS subgraph samples."""

    subgraphs = sample_dense_subgraphs(graph, node_count=node_count, samples=samples)
    if not subgraphs:
        return {"samples": 0, "avg_size": 0.0}
    sizes = [len(s) for s in subgraphs]
    return {
        "samples": float(len(subgraphs)),
        "avg_size": sum(sizes) / len(sizes),
    }
