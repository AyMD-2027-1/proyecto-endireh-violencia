"""
Carga de los datos de ENDIREH 2021 (Práctica 4).

Los archivos en data/ no se modifican: la corrección de codificación de los
nombres de entidad y municipio se aplica solo en memoria.
"""

import polars as pl

from config.rutas import ARCHIVO_PROCESSED, ARCHIVO_RAW

PESO = "factor_expansion"
COLUMNAS_TEXTO = ["nom_entidad", "nom_municipio"]


def corregir_codificacion(texto: str) -> str:
    """Corrige textos UTF-8 que quedaron guardados como latin-1 (p. ej. 'NUEVO LEÃ\x93N' -> 'NUEVO LEÓN')."""
    try:
        return texto.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return texto


def _corregir_nombres(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col(c).map_elements(corregir_codificacion, return_dtype=pl.String)
        for c in COLUMNAS_TEXTO
        if c in df.columns
    )


def cargar_datos_crudos() -> pl.DataFrame:
    """Archivo descargado en data/data-raw/, tal como se obtuvo."""
    return pl.read_csv(ARCHIVO_RAW)


def cargar_datos_limpios() -> pl.DataFrame:
    """Dataset preprocesado en la Práctica 3, con los nombres de entidad legibles."""
    return _corregir_nombres(pl.read_csv(ARCHIVO_PROCESSED))
