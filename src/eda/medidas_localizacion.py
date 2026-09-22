"""
Pipeline de medidas de localización sobre el dataset ya limpio.

A diferencia de notebooks/medidas_localizacion.ipynb (que calcula las
medidas directamente sobre data-raw), este script parte del CSV ya
preprocesado por src/cleaning/limpieza_de_datos.py
(data/data-processed/endireh_2021_clean.csv).

Ejecutar desde la raíz del proyecto con python -m src.eda.medidas_localizacion
"""

import numpy as np
import polars as pl

from config.rutas import RUTA_DATA_PROCESSED

# Variables cuantitativas de interés para las medidas de localización.
CUANTITATIVAS = [
    "edad_primer_union",
    "num_hijos",
    "ingreso_pareja",
]

def cargar_datos_limpios() -> pl.LazyFrame:
    """Carga el dataset ya limpio de forma perezosa."""
    return pl.scan_csv(RUTA_DATA_PROCESSED / "endireh_2021_clean.csv")

def calcular_medidas_localizacion(df: pl.DataFrame, variable: str) -> pl.DataFrame:
    """Media, mediana, moda, cuantiles/percentiles y media ponderada por factor_expansion."""
    resumen = df.select(
        pl.col(variable).quantile(0.1).alias("P10"),
        pl.col(variable).quantile(0.25).alias("Q1"),
        pl.col(variable).quantile(0.75).alias("Q3"),
        pl.col(variable).quantile(0.9).alias("P90"),
        pl.col(variable).drop_nulls().mode().first().alias("Moda"),
        pl.col(variable).median().alias("Mediana"),
        pl.col(variable).mean().alias("Media"),
    )

    df_no_nulo = df.drop_nulls([variable, "factor_expansion"])
    valores = df_no_nulo[variable].to_numpy()
    pesos = df_no_nulo["factor_expansion"].to_numpy()
    media_ponderada = np.average(valores, weights=pesos)
    resumen = resumen.with_columns(pl.lit(media_ponderada).alias("Media ponderada"))

    return resumen


def ejecutar_pipeline() -> dict[str, pl.DataFrame]:
    """Ejecuta el pipeline de medidas de localización sobre todas las variables cuantitativas."""
    lf = cargar_datos_limpios()

    resultados = {}
    for variable in CUANTITATIVAS:
        df = lf.select(["factor_expansion", variable]).collect()
        resumen = calcular_medidas_localizacion(df, variable)
        resultados[variable] = resumen

        print(f"Resumen de {variable} (n = {df.height}):")
        print(resumen)
        print()

    return resultados


if __name__ == "__main__":
    ejecutar_pipeline()
    