"""
Medidas de concentración: curva de Lorenz y coeficiente de Gini.

Ningún framework de propósito general (polars, pandas, numpy) las implementa de
fábrica, por lo que viven aquí para reutilizarse desde src/eda y los notebooks.
"""

import numpy as np

def curva_lorenz(x) -> tuple[np.ndarray, np.ndarray]:
    """Proporción acumulada de unidades y de la cantidad, con x ordenado de forma ascendente.

    Ambos arreglos empiezan en 0 para que la curva parta del origen.
    """
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    unidades = np.arange(0, n + 1) / n
    acumulado = np.concatenate([[0.0], np.cumsum(x) / x.sum()])
    return unidades, acumulado


def coeficiente_gini(x) -> float:
    """G = (2 sum(i x_(i)) - (n + 1) sum(x_(i))) / (n sum(x_(i))), con x_(i) ascendente e i = 1..n."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    total = x.sum()
    if n == 0 or total == 0:
        return 0.0
    i = np.arange(1, n + 1)
    return float((2 * np.sum(i * x) - (n + 1) * total) / (n * total))


def gini_maximo(n: int) -> float:
    """Valor máximo que alcanza la fórmula de Gini con n unidades: (n - 1) / n."""
    return (n - 1) / n
