# Actividad 1: Programación cuántica diferencial

Este repositorio contiene dos scripts para los ejercicios del laboratorio de programación cuántica diferencial. Ambos utilizan PennyLane para construir y simular circuitos cuánticos y optimizar parámetros.

El módulo `funciones_auxiliares.py` contiene las funciones compartidas para
calcular fidelidades entre estados y entre matrices unitarias.

## Ejercicios

### Ejercicio 1: búsqueda de las puertas del circuito Toffoli
El script `1-buscasolucion.py` representa un circuito de tres qubits con dos puertas `U3` parametrizadas. Utiliza un optimizador para buscar los parámetros que hacen que el circuito sea equivalente a una puerta Toffoli, midiendo la fidelidad entre sus matrices. Al final muestra los parámetros, el coste y la fidelidad, y comprueba una solución concreta.

### Ejercicio 2: preparación de un estado de superposición

El script `2-buscarcircuito.py` construye un circuito variacional de tres qubits y tres capas, con rotaciones `RY` y puertas `CNOT`. Optimiza sus parámetros para aproximar el estado objetivo con amplitud cero para `|000>` y amplitudes iguales para los otros siete estados computacionales. Muestra los
parámetros optimizados, las amplitudes y probabilidades del estado final, la fidelidad, el coste y el circuito obtenido.

## Requisitos

- Python 3.
- PennyLane (`pennylane`), que proporciona también el módulo `pennylane.numpy`
  utilizado por los scripts.

No es necesario instalar NumPy por separado para ejecutar estos programas.

## Instalación

Se recomienda crear un entorno virtual desde la carpeta del proyecto.

En Windows, desde PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install pennylane
```

Si el comando `py` no está disponible, se puede sustituir por `python`.
En macOS o Linux, los comandos equivalentes son:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pennylane
```

En VS Code, selecciona el intérprete de Python del entorno `.venv` para que el editor y la terminal utilicen el mismo entorno.

## Ejecución
Ejecuta cada ejercicio por separado desde la carpeta que contiene los scripts:

```powershell
python .\1-buscasolucion.py
python .\2-buscarcircuito.py
```

En macOS o Linux, se pueden ejecutar con `python` seguido del nombre del script, por ejemplo `python 1-buscasolucion.py`.

Los scripts realizan la optimización al ejecutarse e imprimen los resultados directamente en la terminal. El proceso puede tardar unos instantes, ya que cada programa puede realizar hasta 1000 iteraciones."# algoritmos-cuanticos-actividad2" 
