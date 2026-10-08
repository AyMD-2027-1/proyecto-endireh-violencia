"""
Comparación entre Gini y entropía sobre la misma variable categórica (estado_civil_desc).

 * Entropía de Shannon (heterogeneidad), normalizada H / log2(k)
 * Coeficiente de Gini sobre las frecuencias de las categorías (concentración), normalizado G * k / (k - 1)

Se calcula ponderado con factor_expansion (población) y sin ponderar (muestra), para
todas las mujeres y solo para las que reportaron violencia de pareja.

Ejecutar desde la raíz del proyecto con python -m src.eda.comparacion_gini_entropia
"""

import numpy as np
import polars as pl

from config.rutas import ARCHIVO_PROCESSED
from src.eda.indices import (
    coeficiente_gini,
    entropia_maxima,
    entropia_shannon,
    frecuencias,
    gini_maximo,
    gini_simpson,
    iqv,
)

VAR = "estado_civil_desc"          # variable categórica elegida
PESO = "factor_expansion"          # factor de expansión (FAC_MUJ)
VIOL = "sufrio_violencia_pareja"   # 1 = sí sufrió violencia de pareja, 0 = no


def indices_desde_frecuencias(n) -> dict[str, float]:
    """Entropía, Gini, Gini-Simpson e IQV a partir de las frecuencias (ponderadas o no) de cada categoría."""
    n = np.asarray(n, dtype=float)
    p = n / n.sum()
    k = len(n)
    h = entropia_shannon(p)
    g = coeficiente_gini(n)
    return {
        "k": k,
        "H (bits)": h,
        "H max (bits)": entropia_maxima(k),
        "H_norm": h / entropia_maxima(k),
        "G": g,
        "G max": gini_maximo(k),
        "G_norm": g / gini_maximo(k),
        "Gini-Simpson": gini_simpson(p),
        "IQV": iqv(p),
    }


def escenarios(df: pl.DataFrame, var: str = VAR) -> dict[str, pl.DataFrame]:
    """Frecuencias de la variable para todas las mujeres y las que reportaron violencia, ponderadas y sin ponderar."""
    df_viol = df.filter(pl.col(VIOL) == 1)
    return {
        "Todas · ponderado": frecuencias(df, var, PESO),
        "Todas · sin ponderar": frecuencias(df, var),
        "Con violencia · ponderado": frecuencias(df_viol, var, PESO),
        "Con violencia · sin ponderar": frecuencias(df_viol, var),
    }


def tabla_indices(frecs: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Una fila por índice y una columna por escenario."""
    resultados = {nombre: indices_desde_frecuencias(f["n"].to_numpy()) for nombre, f in frecs.items()}
    claves = list(next(iter(resultados.values())))
    return pl.DataFrame({
        "Índice": claves,
        **{nombre: [round(float(r[c]), 4) for c in claves] for nombre, r in resultados.items()},
    })


if __name__ == "__main__":
    with pl.Config(tbl_cols=-1, tbl_width_chars=250):
        print(tabla_indices(escenarios(pl.read_csv(ARCHIVO_PROCESSED))))
