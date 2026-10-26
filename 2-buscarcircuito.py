# ---------------------------------------------------------
# Ejercicio 2 de la Actividad 1
# ---------------------------------------------------------

from functools import partial

import pennylane as qml
from pennylane import numpy as np

from funciones_auxiliares import (
    fidelidad_estado,
    funcion_coste as calcular_coste,
)

NUMERO_QUBITS = 3
NUMERO_CAPAS = 3
MAX_ITERACIONES = 1000
TOLERANCIA = 1e-10
TASA_APRENDIZAJE = 0.05

dispositivo = qml.device("default.qubit", wires=NUMERO_QUBITS)

# ---------------------------------------------------------
# Estado objetivo
# ---------------------------------------------------------

estado_objetivo = np.array(
    [
        0,
        1 / np.sqrt(7),
        1 / np.sqrt(7),
        1 / np.sqrt(7),
        1 / np.sqrt(7),
        1 / np.sqrt(7),
        1 / np.sqrt(7),
        1 / np.sqrt(7),
    ],
    dtype=complex,
)

# ---------------------------------------------------------
# Circuito variacional
# ---------------------------------------------------------

@qml.qnode(dispositivo)
def circuito_variacional(parametros):

    for capa in range(NUMERO_CAPAS):

        for qubit in range(NUMERO_QUBITS):
            qml.RY(
                parametros[capa, qubit],
                wires=qubit,
            )

        # Cadena de puertas CNOT que conecta los tres qubits
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

    return qml.state()


# ---------------------------------------------------------
# Fidelidad
# ---------------------------------------------------------

fidelidad = partial(
    fidelidad_estado,
    circuito=circuito_variacional,
    estado_objetivo=estado_objetivo,
)
funcion_coste = partial(calcular_coste, calcular_fidelidad=fidelidad)


# ---------------------------------------------------------
# Inicialización
# ---------------------------------------------------------

# Semilla fija para poder reproducir la inicialización aleatoria
np.random.seed(42)

parametros = np.array(
    0.1 * np.random.randn(NUMERO_CAPAS, NUMERO_QUBITS),
    requires_grad=True,
)


# ---------------------------------------------------------
# Optimización
# ---------------------------------------------------------

optimizador = qml.AdamOptimizer(stepsize=TASA_APRENDIZAJE)

for iteracion in range(MAX_ITERACIONES):

    parametros = optimizador.step(funcion_coste, parametros)

    coste_actual = funcion_coste(parametros)

    if iteracion % 25 == 0:

        print(
            f"Iteración {iteracion:4d} | "
            f"Coste = {coste_actual:.10f} | "
            f"Fidelidad = {1 - coste_actual:.10f}"
        )

    if coste_actual < TOLERANCIA:
        break


# ---------------------------------------------------------
# Resultados
# ---------------------------------------------------------

estado_final = circuito_variacional(parametros)

probabilidades = np.abs(estado_final) ** 2

estados_computacionales = [
    "000",
    "001",
    "010",
    "011",
    "100",
    "101",
    "110",
    "111",
]

print("\nParámetros finales:")
print(parametros)

print("\nEstado final:")

for estado_base, amplitud, probabilidad in zip(
    estados_computacionales,
    estado_final,
    probabilidades,
):

    print(
        f"|{estado_base}> "
        f"amplitud = {amplitud.real:+.6f}"
        f"{amplitud.imag:+.6f}j | "
        f"P = {probabilidad:.6f}"
    )

print("\nFidelidad final:", fidelidad(parametros))
print("Coste final:", funcion_coste(parametros))

# ---------------------------------------------------------
# Circuito obtenido
# ---------------------------------------------------------

print("\nCircuito:")
print(qml.draw(circuito_variacional)(parametros))