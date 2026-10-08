"""
Medidas de localización de las variables cuantitativas, sobre el dataset crudo
(data/data-raw) y sobre el ya preprocesado (data/data-processed):
 * Percentiles P10, Q1, Q3, P90
 * Moda, mediana y media
 * Media y mediana ponderadas por factor_expansion

En el crudo se quitan antes los nulos y los códigos especiales
(98/99 en edad_primer_union, 999998/999999 en ingreso_pareja).

Ejecutar desde la raíz del proyecto con python -m src.eda.medidas_localizacion
"""

import polars as pl

from config.rutas import ARCHIVO_PROCESSED, ARCHIVO_RAW
from src.eda.indices import media_ponderada, mediana_ponderada

FUENTES = {
    "crudo": ARCHIVO_RAW,
    "procesado": ARCHIVO_PROCESSED,
}

# Variables cuantitativas de interés para las medidas de localización.
CUANTITATIVAS = [
    "edad_primer_union",
    "num_hijos",
    "ingreso_pareja",
]
COL_PESO = "factor_expansion"
CODIGOS_ESPECIALES = {
    "edad_primer_union": [98, 99],
    "ingreso_pareja": [999998, 999999],
}


def limpiar_cuantitativa(df: pl.DataFrame, variable: str) -> pl.DataFrame:
    """Quita nulos, factores de expansión no positivos y los códigos especiales de la variable."""
    return df.filter(
        pl.col(variable).is_not_null()
        & pl.col(COL_PESO).is_not_null()
        & (pl.col(COL_PESO) > 0)
        & ~pl.col(variable).is_in(CODIGOS_ESPECIALES.get(variable, []))
    )


def calcular_medidas_localizacion(df: pl.DataFrame, variable: str) -> pl.DataFrame:
    """Percentiles, moda, mediana, media y media/mediana ponderadas de una variable ya limpia."""
    resumen = df.select(
        pl.col(variable).quantile(0.1).alias("P10"),
        pl.col(variable).quantile(0.25).alias("Q1"),
        pl.col(variable).quantile(0.75).alias("Q3"),
        pl.col(variable).quantile(0.9).alias("P90"),
        pl.col(variable).mode().first().alias("Moda"),
        pl.col(variable).median().alias("Mediana"),
        pl.col(variable).mean().alias("Media"),
    )
    valores, pesos = df[variable].to_numpy(), df[COL_PESO].to_numpy()
    return resumen.with_columns(
        pl.lit(media_ponderada(valores, pesos)).alias("Media ponderada"),
        pl.lit(mediana_ponderada(valores, pesos)).alias("Mediana ponderada"),
    )


def tabla_localizacion(df: pl.DataFrame, dataset: str) -> pl.DataFrame:
    """Medidas de localización de todas las variables cuantitativas de un dataset."""
    return pl.concat([
        calcular_medidas_localizacion(limpiar_cuantitativa(df, v), v).select(
            pl.lit(dataset).alias("dataset"), pl.lit(v).alias("variable"), pl.all()
        )
        for v in CUANTITATIVAS
    ])


def calcular_todo() -> pl.DataFrame:
    return pl.concat([tabla_localizacion(pl.read_csv(ruta), etiqueta) for etiqueta, ruta in FUENTES.items()])


if __name__ == "__main__":
    with pl.Config(tbl_cols=-1, tbl_width_chars=250):
        print(calcular_todo())
