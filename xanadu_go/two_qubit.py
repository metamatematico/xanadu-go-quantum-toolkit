"""Two-qubit Hamiltonian utilities for Go energetic analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import pennylane as qml
from pennylane import numpy as np

STATE_INTERPRETATIONS: Dict[str, str] = {
    "00": "Blanco - Blanco",
    "01": "Blanco - Negro",
    "0+": "Blanco - Vacío (+)",
    "0-": "Blanco - Vacío (-)",
    "10": "Negro - Blanco",
    "11": "Negro - Negro",
    "1+": "Negro - Vacío (+)",
    "1-": "Negro - Vacío (-)",
    "+0": "Vacío (+) - Blanco",
    "+1": "Vacío (+) - Negro",
    "++": "Vacío (+) - Vacío (+)",
    "+-": "Vacío (+) - Vacío (-)",
    "-0": "Vacío (-) - Blanco",
    "-1": "Vacío (-) - Negro",
    "-+": "Vacío (-) - Vacío (+)",
    "--": "Vacío (-) - Vacío (-)",
}


@dataclass
class EnergyResult:
    state: str
    description: str
    quantum: float
    analytic: float
    classical: float
    delta_quantum: float
    delta_classical: float


def create_hamiltonian() -> qml.Hamiltonian:
    coeffs = [1.0, 1.0, 1.0]
    observables = [
        qml.Identity(0) @ qml.PauliZ(1),
        qml.PauliZ(0) @ qml.PauliX(1),
        qml.PauliX(0) @ qml.PauliZ(1),
    ]
    return qml.Hamiltonian(coeffs, observables)


def _initialize_qubit_from_char(wire: int, char: str) -> None:
    if char == "0":
        return
    if char == "1":
        qml.PauliX(wires=wire)
        return
    if char == "+":
        qml.Hadamard(wires=wire)
        return
    if char == "-":
        qml.Hadamard(wires=wire)
        qml.PauliZ(wires=wire)
        return
    raise ValueError(f"Carácter inválido: {char}")


def _initialize_two_qubits(state: str) -> None:
    if len(state) != 2:
        raise ValueError("El estado debe tener exactamente 2 caracteres.")
    _initialize_qubit_from_char(0, state[0])
    _initialize_qubit_from_char(1, state[1])


def _analytic_expectations() -> Dict[str, Dict[str, float]]:
    return {
        "0": {"Z": 1.0, "X": 0.0, "I": 1.0},
        "1": {"Z": -1.0, "X": 0.0, "I": 1.0},
        "+": {"Z": 0.0, "X": 1.0, "I": 1.0},
        "-": {"Z": 0.0, "X": -1.0, "I": 1.0},
    }


def _term_expect_on_state(term, state: str) -> float:
    expectations = _analytic_expectations()

    def factors(op) -> Iterable:
        if hasattr(op, "obs"):
            return op.obs
        if hasattr(op, "operands"):
            return op.operands
        return [op]

    label_per_wire = {0: "I", 1: "I"}
    for factor in factors(term):
        wires = getattr(factor, "wires", [])
        if not wires:
            continue
        wire = int(list(wires)[0])
        name = getattr(factor, "name", str(factor))
        if "PauliZ" in name:
            label_per_wire[wire] = "Z"
        elif "PauliX" in name:
            label_per_wire[wire] = "X"
        elif "Identity" in name:
            label_per_wire[wire] = "I"
    return expectations[state[0]][label_per_wire[0]] * expectations[state[1]][label_per_wire[1]]


def analytic_energy(H: qml.Hamiltonian, state: str) -> float:
    total = 0.0
    for coeff, term in zip(H.coeffs, H.ops):
        total += float(coeff) * _term_expect_on_state(term, state)
    return total


def classical_params_from_quantum(H: qml.Hamiltonian) -> Dict[str, float]:
    def factors(op) -> Iterable:
        if hasattr(op, "obs"):
            return op.obs
        if hasattr(op, "operands"):
            return op.operands
        return [op]

    coeff_IZ = coeff_ZX = coeff_XZ = 0.0
    for coeff, term in zip(H.coeffs, H.ops):
        label_per_wire = {0: "I", 1: "I"}
        for factor in factors(term):
            wires = getattr(factor, "wires", [])
            if not wires:
                continue
            wire = int(list(wires)[0])
            name = getattr(factor, "name", str(factor))
            if "PauliZ" in name:
                label_per_wire[wire] = "Z"
            elif "PauliX" in name:
                label_per_wire[wire] = "X"
            elif "Identity" in name:
                label_per_wire[wire] = "I"
        if (label_per_wire[0], label_per_wire[1]) == ("I", "Z"):
            coeff_IZ += float(coeff)
        elif (label_per_wire[0], label_per_wire[1]) == ("Z", "X"):
            coeff_ZX += float(coeff)
        elif (label_per_wire[0], label_per_wire[1]) == ("X", "Z"):
            coeff_XZ += float(coeff)

    h0 = coeff_ZX
    h1 = coeff_IZ + coeff_XZ
    J = 0.0
    K = -coeff_ZX
    L = -coeff_XZ
    return {"h0": h0, "h1": h1, "J": J, "K": K, "L": L}


def _spin_value(char: str) -> float:
    if char == "0":
        return 1.0
    if char == "1":
        return -1.0
    return 0.0


def classical_energy(params: Dict[str, float], state: str) -> float:
    h0, h1, J, K, L = (
        params["h0"],
        params["h1"],
        params["J"],
        params["K"],
        params["L"],
    )
    s0, s1 = _spin_value(state[0]), _spin_value(state[1])
    return h0 * s0 + h1 * s1 + J * s0 * s1 + K * s0 * (s1**2) + L * (s0**2) * s1


def build_two_qubit_circuit(H: qml.Hamiltonian, device_name: str = "default.qubit"):
    dev = qml.device(device_name, wires=2)

    @qml.qnode(dev)
    def circuit(state: str):
        _initialize_two_qubits(state)
        return qml.expval(H)

    return circuit


def verify_states(H: qml.Hamiltonian, circuit) -> List[EnergyResult]:
    params = classical_params_from_quantum(H)
    results: List[EnergyResult] = []
    for state, description in STATE_INTERPRETATIONS.items():
        quantum_val = float(circuit(state))
        analytic_val = analytic_energy(H, state)
        classical_val = classical_energy(params, state)
        results.append(
            EnergyResult(
                state=state,
                description=description,
                quantum=quantum_val,
                analytic=analytic_val,
                classical=classical_val,
                delta_quantum=analytic_val - quantum_val,
                delta_classical=analytic_val - classical_val,
            )
        )
    return results


def summarize_results(results: List[EnergyResult]) -> None:
    print("Estado | ⟨H⟩_quantum | ⟨H⟩_analítico | ⟨H⟩_clásico | Δ_q | Δ_cls | Descripción")
    for item in sorted(results, key=lambda r: r.state):
        print(
            f" {item.state:>3} | {item.quantum:>12.3f} | {item.analytic:>13.3f} |"
            f" {item.classical:>12.3f} | {item.delta_quantum:>5.2f} |"
            f" {item.delta_classical:>7.2f} | {item.description}"
        )

    max_delta_q = max(abs(r.delta_quantum) for r in results)
    max_delta_c = max(abs(r.delta_classical) for r in results)
    print(f"\nMáx |analítico - circuito| = {max_delta_q:.3e}")
    print(f"Máx |analítico - clásico|  = {max_delta_c:.3e}")


def run_full_verification(device_name: str = "default.qubit") -> None:
    H = create_hamiltonian()
    circuit = build_two_qubit_circuit(H, device_name=device_name)
    results = verify_states(H, circuit)

    print("Hamiltoniano H = I⊗Z + Z⊗X + X⊗Z")
    print(H)
    print("\nVerificación cuantitativa de estados:")
    summarize_results(results)

    params = classical_params_from_quantum(H)
    print("\nParámetros clásicos derivados de H:")
    print(params)
