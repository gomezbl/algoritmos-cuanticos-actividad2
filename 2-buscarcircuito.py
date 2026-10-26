"""Optimiza un circuito variacional para preparar un estado de superposición."""

import pennylane as qml
from pennylane import numpy as np

from funciones_auxiliares import state_fidelity


# Problema y parámetros de optimización
NUM_QUBITS = 3
NUM_LAYERS = 3
DIMENSION = 2**NUM_QUBITS
NUM_TARGET_BASIS_STATES = DIMENSION - 1
LEARNING_RATE = 0.05
INITIAL_PARAMETER_SCALE = 0.1
MAX_ITERATIONS = 1000
TOLERANCE = 1e-10
RANDOM_SEED = 42
REPORT_INTERVAL = 100

device = qml.device("default.qubit", wires=NUM_QUBITS)
basis_states = [format(index, f"0{NUM_QUBITS}b") for index in range(DIMENSION)]

# |φ⟩ = (1/√7) Σ(i=1..7) |i⟩:
# amplitud cero para |000⟩ e igual a 1/√7 para los otros siete estados.
target_amplitude = 1 / np.sqrt(NUM_TARGET_BASIS_STATES)
target_state = np.array(
    [0] + [target_amplitude] * NUM_TARGET_BASIS_STATES,
    dtype=complex,
)


@qml.qnode(device)
def parameterized_circuit(parameters):
    for layer in range(NUM_LAYERS):
        for qubit in range(NUM_QUBITS):
            qml.RY(parameters[layer, qubit], wires=qubit)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

    return qml.state()


def fidelity(parameters):
    circuit_state = parameterized_circuit(parameters)
    return state_fidelity(target_state, circuit_state)


def cost_function(parameters):
    return 1.0 - fidelity(parameters)


def optimize_parameters():
    np.random.seed(RANDOM_SEED)
    parameters = np.array(
        INITIAL_PARAMETER_SCALE * np.random.randn(NUM_LAYERS, NUM_QUBITS),
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


def validate_probabilities(probabilities):
    uniform_probability = 1.0 / NUM_TARGET_BASIS_STATES
    maximum_probability_error = np.max(
        np.abs(probabilities[1:] - uniform_probability)
    )
    return probabilities[0], maximum_probability_error


def print_results(parameters, iterations):
    final_state = parameterized_circuit(parameters)
    probabilities = np.abs(final_state) ** 2
    final_fidelity = fidelity(parameters)
    final_cost = cost_function(parameters)
    probability_zero, maximum_probability_error = validate_probabilities(
        probabilities
    )

    print("\nResultados del ejercicio 2")
    print(f"Iteraciones realizadas: {iterations}")
    print(f"Coste final: {final_cost:.10f}")
    print(f"Fidelidad final: {final_fidelity:.10f}")
    print("Parámetros encontrados:")
    print(parameters)

    print("\nAmplitudes y probabilidades finales:")
    for basis_state, amplitude, probability in zip(
        basis_states,
        final_state,
        probabilities,
    ):
        print(
            f"|{basis_state}>  "
            f"amplitud = {amplitude.real:+.6f}{amplitude.imag:+.6f}j  "
            f"P = {probability:.6f}"
        )

    print("\nValidación de probabilidades:")
    print(f"P(000): {probability_zero:.10f} (objetivo: 0)")
    print(
        "Máxima desviación de P(001..111) respecto a 1/7: "
        f"{maximum_probability_error:.10e}"
    )
    print("\nCircuito obtenido:")
    circuit_diagram = qml.draw(parameterized_circuit)(parameters)
    circuit_diagram = circuit_diagram.translate(
        str.maketrans(
            {
                "─": "-",
                "│": "|",
                "┤": "|",
                "╭": "+",
                "╰": "+",
                "●": "o",
            }
        )
    )
    print(circuit_diagram)


def main():
    parameters, iterations = optimize_parameters()
    print_results(parameters, iterations)


if __name__ == "__main__":
    main()
