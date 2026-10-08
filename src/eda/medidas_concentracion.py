"""
Paso 3 - Medidas de concentración: curva de Lorenz y coeficiente de Gini.

Se analiza si los casos de violencia de pareja reportados, ponderados por
factor_expansion, se concentran en pocas entidades federativas. Para no
confundir concentración de casos con tamaño de población, el mismo análisis se
repite sobre la población estimada de cada entidad.

Ejecutar desde la raíz del proyecto con python -m src.eda.medidas_concentracion
"""

import numpy as np
import polars as pl

from src.eda.carga import PESO, cargar_datos_limpios
from src.eda.indices import coeficiente_gini, gini_maximo

COL_VIOLENCIA = "sufrio_violencia_pareja"

# Columna de tabla_por_entidad -> descripción
MEDIDAS = {
    "casos": "Casos de violencia de pareja (ponderados)",
    "poblacion": "Mujeres de 15 años y más (ponderado)",
    "casos_muestra": "Casos de violencia en la muestra (sin ponderar)",
}


def tabla_por_entidad(df: pl.DataFrame) -> pl.DataFrame:
    """Casos ponderados, población estimada y participación de cada entidad, de menor a mayor número de casos."""
    return (
        df.group_by("cve_entidad", "nom_entidad")
        .agg(
            pl.len().alias("registros"),
            pl.col(COL_VIOLENCIA).sum().alias("casos_muestra"),
            pl.col(PESO).sum().alias("poblacion"),
            (pl.col(PESO) * pl.col(COL_VIOLENCIA)).sum().alias("casos"),
        )
        .with_columns(
            (pl.col("casos") / pl.col("casos").sum()).alias("participacion_casos"),
            (pl.col("poblacion") / pl.col("poblacion").sum()).alias("participacion_poblacion"),
        )
        .sort("casos")
    )


def tabla_gini(entidades: pl.DataFrame) -> pl.DataFrame:
    """Coeficiente de Gini de cada medida entre las n entidades."""
    n = entidades.height
    return pl.DataFrame([
        {
            "medida": col,
            "descripcion": desc,
            "gini": coeficiente_gini(entidades[col]),
            "gini_normalizado": coeficiente_gini(entidades[col]) / gini_maximo(n),
        }
        for col, desc in MEDIDAS.items()
    ])


def participacion_extremos(x, k: int = 5) -> dict[str, float]:
    """Fracción del total que acumulan las k unidades con más y con menos valor."""
    x = np.sort(np.asarray(x, dtype=float))
    return {f"top_{k}": x[-k:].sum() / x.sum(), f"bottom_{k}": x[:k].sum() / x.sum(), "mitad_inferior": x[: len(x) // 2].sum() / x.sum()}


if __name__ == "__main__":
    df = cargar_datos_limpios()
    entidades = tabla_por_entidad(df)
    with pl.Config(tbl_cols=-1, tbl_rows=-1, tbl_width_chars=250, float_precision=4):
        print(entidades)
        print(tabla_gini(entidades))
    print("Participación de casos:", participacion_extremos(entidades["casos"]))
