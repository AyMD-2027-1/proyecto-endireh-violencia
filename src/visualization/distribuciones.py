"""Distribución de las variables cuantitativas del dataset ya limpio.

Ejecutar desde la raíz del proyecto con python -m src.visualization.distribuciones
"""

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from config.rutas import RUTA_DATA_PROCESSED

CUANTITATIVAS = [
    "edad_primer_union",
    "num_hijos",
    "ingreso_pareja",
]


def cargar_datos_limpios() -> pl.LazyFrame:
    """Carga el dataset ya limpio de forma perezosa."""
    return pl.scan_csv(RUTA_DATA_PROCESSED / "endireh_2021_clean.csv")


def graficar_distribucion(df: pl.DataFrame, variable: str) -> None:
    """Grafica la distribución de una variable cuantitativa."""
    sns.histplot(df[variable].drop_nulls().to_list(), bins=30, kde=True)
    plt.title(f"Distribución de {variable}")
    plt.xlabel(variable)
    plt.ylabel("Frecuencia")
    plt.show()


def ejecutar_pipeline() -> None:
    """Ejecuta el pipeline de visualización de distribuciones para todas las variables cuantitativas."""
    lf = cargar_datos_limpios()
    for variable in CUANTITATIVAS:
        df = lf.select(variable).collect()
        graficar_distribucion(df, variable)


if __name__ == "__main__":
    """Punto de entrada del script."""
    ejecutar_pipeline()
