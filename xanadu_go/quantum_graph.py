"""Quantum Go board representations built on PennyLane."""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np
from typing import Dict, List, Tuple


class UnionFind:
    """Disjoint-set data structure used to contract nodes that share a quantum signature."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while item != self.parent[item]:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: int, b: int) -> bool:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return False
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1
        return True

    def clusters(self) -> Dict[int, List[int]]:
        groups: Dict[int, List[int]] = {}
        for idx in range(len(self.parent)):
            root = self.find(idx)
            groups.setdefault(root, []).append(idx)
        return groups


class QuantumGoGraph:
    """Quantum-native representation of a Go board using PennyLane."""

    def __init__(
        self,
        board_matrix: List[List[int]],
        threshold: float = 0.75,
        device_name: str = "default.qubit",
    ) -> None:
        self.board = np.array(board_matrix, dtype=float, requires_grad=False)
        if self.board.ndim != 2 or self.board.shape[0] != self.board.shape[1]:
            raise ValueError("The Go board must be a square matrix.")
        self.size = self.board.shape[0]
        self.n_wires = self.size * self.size
        self.threshold = threshold
        self.adjacency = self._build_adjacency()

        self.dev_single = qml.device(device_name, wires=1)
        self.dev_pair = qml.device(device_name, wires=2)

        self._single_qnode = qml.QNode(
            self._single_circuit, self.dev_single, interface="autograd"
        )
        self._pair_qnode = qml.QNode(
            self._pair_circuit, self.dev_pair, interface="autograd"
        )

        self._angles = self._encode_angles(self.board)
        self._angles_flat = self._angles.reshape(-1)
        self._fgr_expectations = self._evaluate_single_expectations()
        self._edge_expectations = None
        self.fgr = self._create_fgr_snapshot()
        self.cfg = None

    def _encode_angles(self, board: np.ndarray) -> np.ndarray:
        flat = board.reshape(-1)
        return (1 - flat) * np.pi / 2

    def _build_adjacency(self) -> List[Tuple[int, int]]:
        edges: List[Tuple[int, int]] = []
        for r in range(self.size):
            for c in range(self.size):
                idx = self._coord_to_index(r, c)
                if c + 1 < self.size:
                    edges.append((idx, self._coord_to_index(r, c + 1)))
                if r + 1 < self.size:
                    edges.append((idx, self._coord_to_index(r + 1, c)))
        return edges

    def _single_circuit(self, angle: float):  # type: ignore[override]
        qml.RY(angle, wires=0)
        return qml.expval(qml.PauliZ(0))

    def _pair_circuit(self, angle_a: float, angle_b: float):  # type: ignore[override]
        qml.RY(angle_a, wires=0)
        qml.RY(angle_b, wires=1)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    def _evaluate_single_expectations(self) -> np.ndarray:
        results = [self._single_qnode(angle) for angle in self._angles_flat]
        return np.array(results)

    def _decode_colors(self, expectations: np.ndarray) -> List[int]:
        colors: List[int] = []
        for value in expectations:
            if value > self.threshold:
                colors.append(1)
            elif value < -self.threshold:
                colors.append(-1)
            else:
                colors.append(0)
        return colors

    def _create_fgr_snapshot(self) -> Dict[Tuple[int, int], Dict[str, float]]:
        colors = self._decode_colors(self._fgr_expectations)
        nodes: Dict[Tuple[int, int], Dict[str, float]] = {}
        for idx, color in enumerate(colors):
            row, col = divmod(idx, self.size)
            nodes[(row, col)] = {
                "color": color,
                "expval": float(self._fgr_expectations[idx]),
            }
        return nodes

    def create_cfg(self) -> None:
        if self.cfg is not None:
            return
        if self._edge_expectations is None:
            pair_expectations = []
            for u, v in self.adjacency:
                angle_u = float(self._angles_flat[u])
                angle_v = float(self._angles_flat[v])
                pair_expectations.append(self._pair_qnode(angle_u, angle_v))
            self._edge_expectations = np.array(pair_expectations)

        colors = self._decode_colors(self._fgr_expectations)
        uf = UnionFind(self.n_wires)

        for idx, (u, v) in enumerate(self.adjacency):
            if colors[u] == 0 or colors[v] == 0:
                continue
            if self._edge_expectations[idx] > self.threshold:
                uf.union(u, v)

        clusters = uf.clusters()
        root_map = self._build_root_map(clusters)
        cfg_nodes = self._assemble_cfg_nodes(clusters, colors)
        cfg_edges = self._assemble_cfg_edges(root_map)

        self.cfg = {
            "nodes": cfg_nodes,
            "edges": cfg_edges,
            "edge_expectations": [float(val) for val in self._edge_expectations],
        }

    def _build_root_map(self, clusters: Dict[int, List[int]]) -> Dict[int, int]:
        mapping: Dict[int, int] = {}
        for root, members in clusters.items():
            for member in members:
                mapping[member] = root
        return mapping

    def _assemble_cfg_nodes(
        self, clusters: Dict[int, List[int]], colors: List[int]
    ) -> Dict[int, Dict[str, object]]:
        cfg_nodes: Dict[int, Dict[str, object]] = {}
        for root, members in clusters.items():
            if not members:
                continue
            color = colors[members[0]]
            coords = [self._index_to_coord(m) for m in members]
            cfg_nodes[root] = {
                "color": color,
                "members": coords,
                "count": len(members),
            }
        return cfg_nodes

    def _assemble_cfg_edges(self, root_map: Dict[int, int]) -> List[Tuple[int, int]]:
        edge_set = set()
        for u, v in self.adjacency:
            root_u, root_v = root_map[u], root_map[v]
            if root_u != root_v:
                edge_set.add(tuple(sorted((root_u, root_v))))
        return sorted(edge_set)

    def _coord_to_index(self, row: int, col: int) -> int:
        return row * self.size + col

    def _index_to_coord(self, index: int) -> Tuple[int, int]:
        return divmod(index, self.size)

    def print_board(self) -> None:
        header = "  " + " ".join("ABCDEFGHJ"[: self.size])
        print(header)
        print(" +" + "-" * (2 * self.size - 1) + "+")
        for r in range(self.size):
            row_label = f"{self.size - r}|"
            line = []
            for c in range(self.size):
                value = int(self.board[r, c])
                if value == -1:
                    line.append("●")
                elif value == 1:
                    line.append("○")
                else:
                    line.append(".")
            print(row_label + " ".join(line) + f"|{self.size - r}")
        print(" +" + "-" * (2 * self.size - 1) + "+")
        print(header)

    def summarize_cfg(self) -> None:
        if self.cfg is None:
            raise RuntimeError("Common Fate Graph has not been created yet.")
        print("CFG super-nodes (root -> members):")
        for root, data in self.cfg["nodes"].items():
            members = [f"({r},{c})" for r, c in data["members"]]
            color = "white" if data["color"] == 1 else "black"
            print(
                f"  root {root}: color={color}, size={data['count']}, members={', '.join(members)}"
            )
        print("CFG edges (between roots):")
        for edge in self.cfg["edges"]:
            print(f"  {edge}")
