"""
Funciones de medidas descriptivas que reutilizan los scripts de src/eda y los notebooks.

Ningún framework de propósito general (polars, pandas, numpy) implementa de fábrica
los índices de heterogeneidad y concentración, por eso viven aquí.

 * Frecuencias de una variable categórica (simples o ponderadas)
 * Heterogeneidad: Gini-Simpson, IQV y entropía de Shannon
 * Concentración: curva de Lorenz y coeficiente de Gini
 * Medidas ponderadas: media y mediana ponderadas por factor_expansion
"""

import numpy as np
import polars as pl


# ---------- Frecuencias ----------
def frecuencias(df: pl.DataFrame, variable: str, peso: str | None = None) -> pl.DataFrame:
    """Frecuencia `n` y proporción `p` de cada categoría, de mayor a menor. Los nulos se excluyen.

    peso=None cuenta filas (describe a la muestra); peso="factor_expansion" suma el
    factor de expansión (describe a la población).
    """
    columnas = [variable] + ([peso] if peso else [])
    conteo = pl.col(peso).cast(pl.Float64).sum() if peso else pl.len().cast(pl.Float64)
    return (
        df.select(columnas)
        .drop_nulls()
        .group_by(variable)
        .agg(conteo.alias("n"))
        .with_columns((pl.col("n") / pl.col("n").sum()).alias("p"))
        .sort("p", descending=True)
    )


# ---------- Heterogeneidad ----------
def _proporciones(p) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return p / p.sum()


def gini_simpson(p) -> float:
    """GS = 1 - sum(p_i^2)."""
    p = _proporciones(p)
    return float(1 - np.sum(p**2))


def iqv(p) -> float:
    """IQV = k (1 - sum(p_i^2)) / (k - 1), con k = número de categorías. Con k = 1 vale 0."""
    k = len(p)
    return float(k * gini_simpson(p) / (k - 1)) if k > 1 else 0.0


def entropia_shannon(p) -> float:
    """H = -sum(p_i log2 p_i), en bits; 0 log(0) = 0."""
    p = _proporciones(p)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p))) + 0.0


def entropia_maxima(k: int) -> float:
    """log2(k): entropía cuando las k categorías están balanceadas."""
    return float(np.log2(k)) if k > 1 else 0.0


# ---------- Concentración ----------
def curva_lorenz(x) -> tuple[np.ndarray, np.ndarray]:
    """Proporción acumulada de unidades y de la cantidad, con x en orden ascendente; ambas parten de 0."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    return np.arange(0, n + 1) / n, np.concatenate([[0.0], np.cumsum(x) / x.sum()])


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
    """Valor máximo de la fórmula de Gini con n unidades: (n - 1) / n."""
    return (n - 1) / n


# ---------- Medidas ponderadas ----------
def media_ponderada(valores, pesos) -> float:
    return float(np.average(np.asarray(valores, dtype=float), weights=np.asarray(pesos, dtype=float)))


def mediana_ponderada(valores, pesos) -> float:
    """Menor valor cuyo peso acumulado alcanza el 50 % del total."""
    valores = np.asarray(valores, dtype=float)
    pesos = np.asarray(pesos, dtype=float)
    orden = np.argsort(valores)
    acumulado = np.cumsum(pesos[orden]) / pesos.sum()
    return float(valores[orden][np.searchsorted(acumulado, 0.5)])
