"""Quantum machine learning utilities for Go board analysis."""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List

import pennylane as qml
from pennylane import numpy as np


def basic_feature_map(board_vector: Iterable[float], wires: List[int]) -> None:
    """Encode board values into a hardware-efficient feature map."""

    for wire, value in zip(wires, board_vector):
        qml.RY(value, wires=wire)
        qml.RZ(value * value, wires=wire)


def quantum_kernel(
    board_a: Iterable[float],
    board_b: Iterable[float],
    *,
    wires: int,
    feature_map: Callable[[Iterable[float], List[int]], None] = basic_feature_map,
    device_name: str = "default.qubit",
) -> float:
    """Compute a simple quantum kernel using state overlap."""

    dev = qml.device(device_name, wires=wires)

    @qml.qnode(dev)
    def kernel_circuit(x, y):
        feature_map(x, list(range(wires)))
        qml.adjoint(lambda: feature_map(y, list(range(wires))))()
        return qml.probs(wires=list(range(wires)))[0]

    x = np.array(list(board_a), requires_grad=False)
    y = np.array(list(board_b), requires_grad=False)
    return float(kernel_circuit(x, y))
