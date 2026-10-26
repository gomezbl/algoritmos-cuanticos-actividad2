"""Busca las puertas U3 que hacen equivalente el circuito a una Toffoli."""

import pennylane as qml
from pennylane import numpy as np

from funciones_auxiliares import unitary_fidelity


# Problema y parámetros de optimización
WIRES = [0, 1, 2]
NUM_UNKNOWN_GATES = 2
PARAMETERS_PER_GATE = 3
NUM_PARAMETERS = NUM_UNKNOWN_GATES * PARAMETERS_PER_GATE
LEARNING_RATE = 0.05
MAX_ITERATIONS = 1000
TOLERANCE = 1e-10
RANDOM_SEED = 42
REPORT_INTERVAL = 100

# Objetivo: U(parameters) ≈ U_Toffoli
target_unitary = qml.matrix(
    qml.Toffoli(wires=WIRES),
    wire_order=WIRES,
)


def parameterized_circuit(parameters):
    """Circuito del enunciado, con dos puertas U3 desconocidas en el qubit 2."""
    qml.T(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.adjoint(qml.T)(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.T(wires=1)

    qml.U3(parameters[0], parameters[1], parameters[2], wires=2)

    qml.CNOT(wires=[1, 2])
    qml.adjoint(qml.T)(wires=2)
    qml.CNOT(wires=[0, 2])
    qml.T(wires=2)
    qml.CNOT(wires=[1, 2])

    qml.U3(parameters[3], parameters[4], parameters[5], wires=2)

    qml.CNOT(wires=[0, 2])
    qml.T(wires=2)
    qml.Hadamard(wires=2)


circuit_unitary = qml.matrix(parameterized_circuit, wire_order=WIRES)


def fidelity(parameters):
    """Fidelidad entre la matriz del circuito y la Toffoli objetivo."""
    return unitary_fidelity(target_unitary, circuit_unitary(parameters))


def cost_function(parameters):
    return 1.0 - fidelity(parameters)


def optimize_parameters():
    np.random.seed(RANDOM_SEED)
    parameters = np.array(
        np.random.uniform(-np.pi, np.pi, size=NUM_PARAMETERS),
        requires_grad=True,
    )
    optimizer = qml.AdamOptimizer(stepsize=LEARNING_RATE)

    for iteration in range(1, MAX_ITERATIONS + 1):
        parameters = optimizer.step(cost_function, parameters)
        cost = cost_function(parameters)

        if iteration % REPORT_INTERVAL == 0:
            print(
                f"Iteración {iteration:4d} | "
                f"Coste: {cost:.6f} | "
                f"Fidelidad: {1.0 - cost:.6f}"
            )

        if cost < TOLERANCE:
            break

    return parameters, iteration


def known_solution_unitary():
    """Construye la solución conocida, solo para validarla tras optimizar."""
    qml.T(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.adjoint(qml.T)(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.T(wires=1)

    qml.Hadamard(wires=2)

    qml.CNOT(wires=[1, 2])
    qml.adjoint(qml.T)(wires=2)
    qml.CNOT(wires=[0, 2])
    qml.T(wires=2)
    qml.CNOT(wires=[1, 2])

    qml.adjoint(qml.T)(wires=2)

    qml.CNOT(wires=[0, 2])
    qml.T(wires=2)
    qml.Hadamard(wires=2)


def validate_known_solution():
    solution_unitary = qml.matrix(known_solution_unitary, wire_order=WIRES)()
    return unitary_fidelity(target_unitary, solution_unitary)


def print_results(parameters, iterations):
    final_fidelity = fidelity(parameters)
    final_cost = cost_function(parameters)
    u3_parameters = np.reshape(
        parameters,
        (NUM_UNKNOWN_GATES, PARAMETERS_PER_GATE),
    )

    print("\nResultados del ejercicio 1")
    print(f"Iteraciones realizadas: {iterations}")
    print(f"Coste final: {final_cost:.10f}")
    print(f"Fidelidad final: {final_fidelity:.10f}")
    print("Parámetros U3 encontrados (theta, phi, lambda):")
    for gate_number, angles in enumerate(u3_parameters, start=1):
        print(
            f"  U3 {gate_number}: "
            f"theta={angles[0]:+.8f}, phi={angles[1]:+.8f}, "
            f"lambda={angles[2]:+.8f}"
        )

    known_fidelity = validate_known_solution()
    print("\nValidación de la solución conocida:")
    print(f"Fidelidad con Toffoli: {known_fidelity:.10f}")
    print(f"Error: {1.0 - known_fidelity:.10e}")


def main():
    parameters, iterations = optimize_parameters()
    print_results(parameters, iterations)


if __name__ == "__main__":
    main()
