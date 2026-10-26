# ---------------------------------------------------------
# Ejercicio 1 de la Actividad 1
# ---------------------------------------------------------

from functools import partial

import pennylane as qml
from pennylane import numpy as np

from funciones_auxiliares import (
    fidelidad_matrices,
    fidelidad_unitaria,
    funcion_coste as calcular_coste,
)

WIRES = [0, 1, 2]
DIMENSION_ESTADO = 2 ** len(WIRES)
MAX_ITERACIONES = 1000
TOLERANCIA = 1e-10
TASA_APRENDIZAJE = 0.05


def circuito_con_puertas_desconocidas(parametros):
    """
    Los tres primeros parámetros definen la primera puerta U3;
    los tres últimos definen la segunda.
    """
    # Bloque conocido anterior a la primera puerta desconocida
    qml.T(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.adjoint(qml.T)(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.T(wires=1)

    # Primera puerta U3 desconocida
    qml.U3(
        parametros[0],
        parametros[1],
        parametros[2],
        wires=2,
    )

    # Bloque conocido entre las dos puertas desconocidas
    qml.CNOT(wires=[1, 2])
    qml.adjoint(qml.T)(wires=2)
    qml.CNOT(wires=[0, 2])
    qml.T(wires=2)
    qml.CNOT(wires=[1, 2])

    # Segunda puerta U3 desconocida
    qml.U3(
        parametros[3],
        parametros[4],
        parametros[5],
        wires=2,
    )

    # Bloque conocido posterior a la segunda puerta desconocida
    qml.CNOT(wires=[0, 2])
    qml.T(wires=2)
    qml.Hadamard(wires=2)


# Matriz del circuito para cada conjunto de parámetros
matriz_circuito_parametrizado = qml.matrix(
    circuito_con_puertas_desconocidas,
    wire_order=WIRES,
)

# Matriz de referencia: puerta Toffoli de tres qubits
puerta_toffoli = qml.Toffoli(wires=WIRES)
matriz_toffoli = qml.matrix(puerta_toffoli, wire_order=WIRES)

fidelidad = partial(
    fidelidad_unitaria,
    circuito=matriz_circuito_parametrizado,
    matriz_objetivo=matriz_toffoli,
    dimension=DIMENSION_ESTADO,
)
funcion_coste = partial(calcular_coste, calcular_fidelidad=fidelidad)


# Inicialización aleatoria reproducible
np.random.seed(42)

parametros = np.array(
    np.random.uniform(-np.pi, np.pi, size=6),
    requires_grad=True,
)

optimizador = qml.AdamOptimizer(stepsize=TASA_APRENDIZAJE)

for iteracion in range(MAX_ITERACIONES):
    parametros = optimizador.step(funcion_coste, parametros)

    if iteracion % 50 == 0:
        coste = funcion_coste(parametros)
        fidelidad_actual = fidelidad(parametros)

        print(
            f"Iteración {iteracion:4d} | "
            f"Coste = {coste:.10f} | "
            f"Fidelidad = {fidelidad_actual:.10f}"
        )

    if funcion_coste(parametros) < TOLERANCIA:
        break


# Resultados de la búsqueda
print("\nParámetros encontrados:")
print(parametros)

print("\nCoste final:")
print(funcion_coste(parametros))

print("\nFidelidad final:")
print(fidelidad(parametros))

# Comprueba una solución conocida para las dos puertas U3
def circuito_solucion():
    qml.T(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.adjoint(qml.T)(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.T(wires=1)

    # Primera interrogación
    qml.Hadamard(wires=2)

    qml.CNOT(wires=[1, 2])
    qml.adjoint(qml.T)(wires=2)
    qml.CNOT(wires=[0, 2])
    qml.T(wires=2)
    qml.CNOT(wires=[1, 2])

    # Segunda interrogación
    qml.adjoint(qml.T)(wires=2)

    qml.CNOT(wires=[0, 2])
    qml.T(wires=2)
    qml.Hadamard(wires=2)


matriz_solucion = qml.matrix(
    circuito_solucion,
    wire_order=WIRES,
)()

fidelidad_final = fidelidad_matrices(
    matriz_toffoli,
    matriz_solucion,
    DIMENSION_ESTADO,
)

print("Fidelidad:", fidelidad_final)
print("Error:", 1.0 - fidelidad_final)