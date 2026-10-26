"""Cálculos de fidelidad compartidos por los dos ejercicios."""

import pennylane as qml


def state_fidelity(target_state, circuit_state):
    """Fidelidad entre dos estados puros, incluyendo sus fases relativas."""
    overlap = qml.math.sum(qml.math.conj(target_state) * circuit_state)
    return qml.math.abs(overlap) ** 2


def unitary_fidelity(target_unitary, circuit_unitary):
    """Fidelidad normalizada entre dos matrices unitarias."""
    dimension = target_unitary.shape[0]
    overlap = qml.math.trace(
        qml.math.conj(qml.math.transpose(target_unitary)) @ circuit_unitary
    )
    return qml.math.abs(overlap) ** 2 / dimension**2
