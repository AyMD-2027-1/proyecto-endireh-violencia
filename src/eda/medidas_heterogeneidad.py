"""
Medidas de heterogeneidad para variables cualitativas:
 * Índice de Gini-Simpson:         GS  = 1 - sum(p_i^2)
 * Índice de Variación Cualitativa: IQV = k * (1 - sum(p_i^2)) / (k - 1)
 * Entropía de Shannon:             H   = -sum(p_i * log2(p_i)),  0 <= H <= log2(k)

p_i = proporción de observaciones en la categoría i; k = número de categorías.

Se calculan sobre el dataset CRUDO y sobre el PROCESADO, y de forma
simple (cada fila cuenta 1) y ponderada con factor_expansion.

Ejecutar desde la raíz del proyecto con python -m src.eda.medidas_heterogeneidad
"""

import polars as pl

from config.rutas import ARCHIVO_PROCESSED, ARCHIVO_RAW
from src.eda.indices import entropia_maxima, entropia_shannon, frecuencias, gini_simpson, iqv

FUENTES = {
    "crudo": ARCHIVO_RAW,
    "procesado": ARCHIVO_PROCESSED,
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


def calcular_heterogeneidad(df: pl.DataFrame, columnas: list[str], peso: str | None = None) -> pl.DataFrame:
    """Gini-Simpson, IQV y entropía de Shannon (absoluta y normalizada) de cada variable."""
    filas = []
    for col in columnas:
        if col not in df.columns:
            continue
        p = frecuencias(df, col, peso)["p"].to_numpy()
        k = len(p)
        h = entropia_shannon(p)
        h_max = entropia_maxima(k)
        filas.append({
            "variable": col,
            "n": df.select([col] + ([peso] if peso else [])).drop_nulls().height,
            "k": k,
            "gini_simpson": gini_simpson(p),
            "gs_max": 1 - 1 / k if k > 0 else 0.0,
            "iqv": iqv(p),
            "shannon_H": h,
            "H_max": h_max,
            "H_norm": h / h_max if h_max > 0 else 0.0,
        })
    return pl.DataFrame(filas)


def calcular_todo() -> pl.DataFrame:
    """Índices de todas las variables para cada dataset (crudo, procesado) y tipo de proporción (simple, ponderada)."""
    resultados = []
    for etiqueta, ruta in FUENTES.items():
        df = pl.read_csv(ruta)
        for peso, nombre_peso in [(None, "simple"), (COL_PESO, "ponderada")]:
            resultados.append(
                calcular_heterogeneidad(df, COLUMNAS, peso=peso).with_columns(
                    pl.lit(etiqueta).alias("dataset"),
                    pl.lit(nombre_peso).alias("proporciones"),
                )
            )
    return pl.concat(resultados)


if __name__ == "__main__":
    with pl.Config(tbl_cols=-1, tbl_width_chars=250, tbl_rows=-1):
        print(calcular_todo())
