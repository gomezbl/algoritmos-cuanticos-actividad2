"""Funciones compartidas para calcular fidelidades y costes."""

import pennylane as qml


def fidelidad_estado(parametros, circuito, estado_objetivo):
    """Calcula la fidelidad entre el estado objetivo y el generado."""
    estado_obtenido = circuito(parametros)
    solapamiento = qml.math.sum(
        qml.math.conj(estado_objetivo) * estado_obtenido
    )
    return qml.math.abs(solapamiento) ** 2


def fidelidad_matrices(matriz_objetivo, matriz_obtenida, dimension):
    """Calcula la fidelidad normalizada entre dos matrices unitarias."""
    solapamiento = qml.math.trace(
        qml.math.conj(qml.math.transpose(matriz_objetivo))
        @ matriz_obtenida
    )
    return qml.math.abs(solapamiento) ** 2 / dimension**2


def fidelidad_unitaria(parametros, circuito, matriz_objetivo, dimension):
    """Calcula la fidelidad entre la matriz del circuito y la matriz objetivo."""
    matriz_obtenida = circuito(parametros)
    return fidelidad_matrices(matriz_objetivo, matriz_obtenida, dimension)


def funcion_coste(parametros, calcular_fidelidad):
    """Devuelve el coste complementario a la fidelidad."""
    return 1.0 - calcular_fidelidad(parametros)
