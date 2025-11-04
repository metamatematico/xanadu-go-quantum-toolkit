"""QAOA-based combinatorial solvers for Go board graphs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import pennylane as qml
from collections import defaultdict, deque
from dataclasses import dataclass
from pennylane import numpy as np
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass
class QAQAResult:
    """Container for QAOA outcomes."""

    bitstring: str
    losses: Sequence[float]
    probability_vector: Sequence[float]
    energy: float
    subgraph_nodes: Sequence[Tuple[int, int]]
    subgraph: nx.Graph


class GraphEmbedding:
    """Selects (optionally limited) subgraphs and relabels nodes for PennyLane wires."""

    def __init__(self, graph: nx.Graph, max_nodes: Optional[int] = None):
        self.original_graph = graph
        self.max_nodes = max_nodes
        self.subgraph_nodes = self._select_nodes()
        self.subgraph = graph.subgraph(self.subgraph_nodes).copy()
        self.node_to_index = {node: idx for idx, node in enumerate(self.subgraph_nodes)}
        self.index_to_node = {idx: node for node, idx in self.node_to_index.items()}
        self.relabelled = nx.relabel_nodes(self.subgraph, self.node_to_index, copy=True)
        self.n_wires = len(self.subgraph_nodes)

    def _select_nodes(self) -> List[Tuple[int, int]]:
        nodes = list(self.original_graph.nodes())
        if not nodes:
            return []
        if self.max_nodes is None or len(nodes) <= self.max_nodes:
            return nodes

        start = max(self.original_graph.degree, key=lambda item: item[1])[0]
        queue: deque = deque([start])
        visited: List[Tuple[int, int]] = []
        seen = set()

        while queue and len(visited) < self.max_nodes:
            node = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            visited.append(node)
            for neighbor in self.original_graph.neighbors(node):
                if neighbor not in seen:
                    queue.append(neighbor)

        if len(visited) < self.max_nodes:
            for node in nodes:
                if node not in seen:
                    visited.append(node)
                    if len(visited) == self.max_nodes:
                        break

        return visited

    def bitstring_to_nodes(self, bitstring: str) -> Set[Tuple[int, int]]:
        return {
            self.index_to_node[idx]
            for idx, bit in enumerate(bitstring)
            if bit == "1"
        }


def _run_qaoa(
    cost_h: qml.Hamiltonian,
    mixer_h: qml.Hamiltonian,
    n_wires: int,
    depth: int,
    steps: int,
    stepsize: float,
    seed: int,
    device_name: str,
) -> QAQAResult:
    if n_wires == 0:
        return QAQAResult("", [], [], 0.0, [], nx.Graph())

    dev = qml.device(device_name, wires=n_wires)
    wires = list(range(n_wires))

    @qml.qnode(dev, interface="autograd")
    def cost_qnode(params):
        betas = params[:depth]
        gammas = params[depth:]
        for w in wires:
            qml.Hadamard(w)
        for layer in range(depth):
            qml.qaoa.cost_layer(gammas[layer], cost_h)
            qml.qaoa.mixer_layer(betas[layer], mixer_h)
        return qml.expval(cost_h)

    @qml.qnode(dev, interface="autograd")
    def prob_qnode(params):
        betas = params[:depth]
        gammas = params[depth:]
        for w in wires:
            qml.Hadamard(w)
        for layer in range(depth):
            qml.qaoa.cost_layer(gammas[layer], cost_h)
            qml.qaoa.mixer_layer(betas[layer], mixer_h)
        return qml.probs(wires=wires)

    np.random.seed(seed)
    params = np.array(np.random.uniform(0, np.pi, size=2 * depth), requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize)
    losses: List[float] = []

    for _ in range(steps):
        params, cost_val = opt.step_and_cost(cost_qnode, params)
        losses.append(float(cost_val))

    probs = prob_qnode(params)
    energy = float(cost_qnode(params))
    index = int(np.argmax(probs))
    bitstring = format(index, f"0{n_wires}b")

    return QAQAResult(
        bitstring=bitstring,
        losses=losses,
        probability_vector=[float(x) for x in probs],
        energy=energy,
        subgraph_nodes=[],  # populated by caller
        subgraph=nx.Graph(),
    )


def solve_max_cut_qaoa(
    graph: nx.Graph,
    *,
    max_nodes: int = 12,
    depth: int = 2,
    steps: int = 60,
    stepsize: float = 0.2,
    seed: int = 1337,
    device_name: str = "default.qubit",
) -> Dict[str, object]:
    embedding = GraphEmbedding(graph, max_nodes=max_nodes)
    cost_h, mixer_h = qml.qaoa.maxcut(embedding.relabelled)
    result = _run_qaoa(cost_h, mixer_h, embedding.n_wires, depth, steps, stepsize, seed, device_name)
    result.subgraph_nodes = embedding.subgraph_nodes
    result.subgraph = embedding.subgraph

    solution_nodes = embedding.bitstring_to_nodes(result.bitstring)
    cut_edges = [
        (u, v)
        for u, v in embedding.subgraph.edges()
        if (u in solution_nodes) != (v in solution_nodes)
    ]

    return {
        "solution_nodes": solution_nodes,
        "cut_edges": cut_edges,
        "num_cut_edges": len(cut_edges),
        "bitstring": result.bitstring,
        "loss_history": result.losses,
        "probabilities": result.probability_vector,
        "energy": result.energy,
        "subgraph_nodes": result.subgraph_nodes,
        "subgraph": result.subgraph,
        "note": (
            "Solution computed on induced subgraph of size"
            f" {embedding.n_wires}. Increase `max_nodes` with caution."
        ),
    }


def _build_vertex_cover_qubo(graph: nx.Graph, penalty: float) -> Dict[Tuple[int, int], float]:
    qubo = defaultdict(float)
    for node in graph.nodes():
        degree = graph.degree(node)
        qubo[(node, node)] += 1.0 - penalty * degree
    for u, v in graph.edges():
        i, j = sorted((u, v))
        qubo[(i, j)] += penalty
    return qubo


def _qubo_hamiltonian(qubo: Dict[Tuple[int, int], float]):
    coeffs = []
    ops = []
    for (i, j), weight in qubo.items():
        if i == j:
            coeffs.append(weight / 2)
            ops.append(qml.PauliZ(i))
        else:
            coeffs.append(weight / 4)
            ops.append(qml.PauliZ(i) @ qml.PauliZ(j))
            coeffs.append(-weight / 4)
            ops.append(qml.PauliZ(i))
            coeffs.append(-weight / 4)
            ops.append(qml.PauliZ(j))
            coeffs.append(weight / 4)
            ops.append(qml.Identity(0))
    return qml.Hamiltonian(coeffs, ops)


def solve_vertex_cover_qaoa(
    graph: nx.Graph,
    *,
    max_nodes: int = 12,
    depth: int = 2,
    steps: int = 60,
    stepsize: float = 0.2,
    seed: int = 1337,
    penalty: float = 2.0,
    device_name: str = "default.qubit",
) -> Dict[str, object]:
    embedding = GraphEmbedding(graph, max_nodes=max_nodes)
    qubo = _build_vertex_cover_qubo(embedding.relabelled, penalty)
    try:
        cost_h, mixer_h = qml.qaoa.qubo(qubo)
    except AttributeError:
        cost_h = _qubo_hamiltonian(qubo)
        mixer_h = qml.qaoa.x_mixer(range(embedding.n_wires))
    result = _run_qaoa(cost_h, mixer_h, embedding.n_wires, depth, steps, stepsize, seed, device_name)
    result.subgraph_nodes = embedding.subgraph_nodes
    result.subgraph = embedding.subgraph

    cover_nodes = embedding.bitstring_to_nodes(result.bitstring)

    return {
        "cover_nodes": cover_nodes,
        "bitstring": result.bitstring,
        "loss_history": result.losses,
        "probabilities": result.probability_vector,
        "energy": result.energy,
        "subgraph_nodes": result.subgraph_nodes,
        "subgraph": result.subgraph,
        "penalty": penalty,
        "note": (
            "Solution computed on induced subgraph of size"
            f" {embedding.n_wires}. Increase `max_nodes` with caution."
        ),
    }


class GoGraphAdvancedXanadu:
    """PennyLane analogue of the D-Wave GoGraphAdvanced workflow."""

    def __init__(
        self,
        board_matrix: List[List[int]],
        *,
        qaoa_max_nodes: int = 12,
        qaoa_depth: int = 2,
        qaoa_steps: int = 60,
        qaoa_stepsize: float = 0.2,
        vertex_penalty: float = 2.0,
        seed: int = 1337,
        device_name: str = "default.qubit",
    ) -> None:
        self.board = board_matrix
        self.size = len(board_matrix)
        self.fgr = self.create_fgr()
        self.cfg = None
        self.rsf = None

        self.qaoa_max_nodes = qaoa_max_nodes
        self.qaoa_depth = qaoa_depth
        self.qaoa_steps = qaoa_steps
        self.qaoa_stepsize = qaoa_stepsize
        self.vertex_penalty = vertex_penalty
        self.seed = seed
        self.device_name = device_name

    def create_fgr(self) -> nx.Graph:
        fgr = nx.grid_2d_graph(self.size, self.size)
        for i in range(self.size):
            for j in range(self.size):
                fgr.nodes[(i, j)]["color"] = self.board[i][j]
        return fgr

    def create_cfg(self) -> None:
        self.cfg = self.fgr.copy()
        changed = True
        while changed:
            changed = False
            for node in list(self.cfg.nodes()):
                for neighbor in list(self.cfg.neighbors(node)):
                    if (
                        self.cfg.nodes[node]["color"]
                        == self.cfg.nodes[neighbor]["color"]
                        and self.cfg.nodes[node]["color"] != 0
                    ):
                        self.cfg = nx.contracted_nodes(
                            self.cfg, node, neighbor, self_loops=False
                        )
                        changed = True
                        break
                if changed:
                    break

    def create_rsf(self) -> None:
        self.rsf = self.fgr.copy()
        for node in list(self.rsf.nodes()):
            if self.rsf.nodes[node]["color"] == 0:
                self.rsf.remove_node(node)

    def visualize_graph(
        self,
        graph: nx.Graph,
        title: str = "Graph",
        highlighted_nodes: Optional[Iterable[Tuple[int, int]]] = None,
        highlighted_edges: Optional[Iterable[Tuple[Tuple[int, int], Tuple[int, int]]]] = None,
        max_cut_nodes: Optional[Iterable[Tuple[int, int]]] = None,
        vertex_cover_nodes: Optional[Iterable[Tuple[int, int]]] = None,
        cut_edges: Optional[Iterable[Tuple[Tuple[int, int], Tuple[int, int]]]] = None,
    ) -> None:
        pos = {(x, y): (y, -x) for x, y in graph.nodes()}
        colors = [graph.nodes[node].get("color", 0) for node in graph.nodes()]
        color_map = {-1: "black", 1: "white", 0: "lightgray"}
        node_colors = [color_map.get(color, "lightgray") for color in colors]

        plt.figure(figsize=(10, 10))
        nx.draw(graph, pos, node_color=node_colors, with_labels=False, node_size=300)

        if highlighted_nodes:
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=list(highlighted_nodes),
                node_color="cyan",
                node_size=500,
            )

        if max_cut_nodes:
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=list(max_cut_nodes),
                node_color="green",
                node_size=500,
            )

        if vertex_cover_nodes:
            nx.draw_networkx_nodes(
                graph,
                pos,
                nodelist=list(vertex_cover_nodes),
                node_color="orange",
                node_size=500,
            )

        if cut_edges:
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=list(cut_edges),
                edge_color="red",
                width=2,
            )

        if highlighted_edges:
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=list(highlighted_edges),
                edge_color="blue",
                width=2,
            )

        plt.title(title)
        plt.axis("off")
        plt.show()

    def solve_problem_xanadu(
        self,
        graph: nx.Graph,
        problem: str = "max_cut",
    ) -> Tuple[Set[Tuple[int, int]], Dict[str, object]]:
        if problem == "max_cut":
            info = solve_max_cut_qaoa(
                graph,
                max_nodes=self.qaoa_max_nodes,
                depth=self.qaoa_depth,
                steps=self.qaoa_steps,
                stepsize=self.qaoa_stepsize,
                seed=self.seed,
                device_name=self.device_name,
            )
            return info["solution_nodes"], info
        if problem == "vertex_cover":
            info = solve_vertex_cover_qaoa(
                graph,
                max_nodes=self.qaoa_max_nodes,
                depth=self.qaoa_depth,
                steps=self.qaoa_steps,
                stepsize=self.qaoa_stepsize,
                seed=self.seed,
                penalty=self.vertex_penalty,
                device_name=self.device_name,
            )
            return info["cover_nodes"], info
        raise ValueError("Problema no reconocido. Usa 'max_cut' o 'vertex_cover'.")

    @staticmethod
    def verify_max_cut(graph: nx.Graph, cut_solution: Set[Tuple[int, int]]) -> Tuple[int, int]:
        cut_edges = [
            (u, v) for u, v in graph.edges() if (u in cut_solution) != (v in cut_solution)
        ]
        return len(cut_edges), graph.number_of_edges()

    @staticmethod
    def verify_vertex_cover(graph: nx.Graph, vertex_cover: Set[Tuple[int, int]]) -> bool:
        for u, v in graph.edges():
            if u not in vertex_cover and v not in vertex_cover:
                return False
        return True

    @staticmethod
    def extract_features_from_solutions(
        max_cut_solution: Set[Tuple[int, int]],
        vertex_cover_solution: Set[Tuple[int, int]],
    ) -> Dict[str, int]:
        return {
            "max_cut_size": len(max_cut_solution),
            "vertex_cover_size": len(vertex_cover_solution),
        }
