"""
Medidas de heterogeneidad para variables cualitativas:
 * Índice de Gini-Simpson:         GS  = 1 - sum(p_i^2)
 * Índice de Variación Cualitativa: IQV = k * (1 - sum(p_i^2)) / (k - 1)
 * Entropía de Shannon:             H   = -sum(p_i * log2(p_i)),  0 <= H <= log2(k)

p_i = proporción de observaciones en la categoría i; k = número de categorías.

Se calculan sobre el dataset CRUDO y sobre el PROCESADO, y de forma
simple (cada fila cuenta 1) y ponderada con factor_expansion.
"""

import math

import polars as pl

from config.rutas import RUTA_DATA_RAW, RUTA_DATA_PROCESSED

FUENTES = {
    "crudo": RUTA_DATA_RAW / "endireh_2021.csv",
    "procesado": RUTA_DATA_PROCESSED / "endireh_2021_clean.csv",
}

# Variables cualitativas. nom_municipio (1,206 categorías) se omite a propósito:
# con tantas categorías casi todo queda cerca del máximo y no aporta lectura.
COLUMNAS = [
    "nom_entidad",
    "nivel_escolaridad",
    "estado_civil_desc",
    "estrato_socioeconomico",
    "pareja_trabaja_desc",
    "dinero_propio_desc",
    "apoyo_gobierno_desc",
    "tiene_ahorros_desc",
    "propietaria_vivienda_desc",
    "sufrio_violencia_pareja",
    "nunca_tuvo_union",  # solo existe en el dataset procesado
]
COL_PESO = "factor_expansion"


def _proporciones(df: pl.DataFrame, col: str, peso: str | None):
    """Devuelve (lista de p_i, n de filas válidas). Los nulos se excluyen."""
    cols = [col] + ([peso] if peso else [])
    sub = df.select(cols).drop_nulls()

    if peso:
        conteo = sub.group_by(col).agg(pl.col(peso).sum().alias("w"))
    else:
        conteo = sub.group_by(col).agg(pl.len().alias("w"))

    total = conteo["w"].sum()
    return (conteo["w"] / total).to_list(), sub.height


def calcular_heterogeneidad(df: pl.DataFrame, columnas: list[str], peso: str | None = None) -> pl.DataFrame:
    filas = []
    for col in columnas:
        if col not in df.columns:
            continue

        p, n = _proporciones(df, col, peso)
        k = len(p)
        suma_p2 = sum(x * x for x in p)

        gs = 1 - suma_p2
        # Con una sola categoría IQV no está definido (k-1 = 0): heterogeneidad nula.
        iqv = k * (1 - suma_p2) / (k - 1) if k > 1 else 0.0
        h = 0.0 - sum(x * math.log2(x) for x in p if x > 0)
        h_max = math.log2(k) if k > 1 else 0.0
        h_norm = h / h_max if h_max > 0 else 0.0

        filas.append({
            "variable": col,
            "n": n,
            "k": k,
            "gini_simpson": gs,
            "gs_max(1-1/k)": 1 - 1 / k if k > 0 else 0.0,
            "iqv": iqv,
            "shannon_H": h,
            "H_max(log2 k)": h_max,
            "H/H_max": h_norm,
        })

    return pl.DataFrame(filas)


if __name__ == "__main__":
    for etiqueta, ruta in FUENTES.items():
        print(f"\n{'=' * 130}\nDATASET: {etiqueta.upper()}  ({ruta.name})\n{'=' * 130}")
        df = pl.read_csv(ruta)

        print(f"\n--- Heterogeneidad ({etiqueta}): proporciones simples ---")
        with pl.Config(tbl_cols=-1, tbl_width_chars=250, tbl_rows=-1):
            print(calcular_heterogeneidad(df, COLUMNAS))

        if COL_PESO in df.columns:
            print(f"\n--- Heterogeneidad ({etiqueta}): ponderada con {COL_PESO} ---")
            with pl.Config(tbl_cols=-1, tbl_width_chars=250, tbl_rows=-1):
                print(calcular_heterogeneidad(df, COLUMNAS, peso=COL_PESO))