"""Rutas estáticas de archivos del proyecto ENDIREH 2021."""

from pathlib import Path

RUTA_PROYECTO = Path(__file__).resolve().parent.parent

# Rutas de archivos de datos
RUTA_DATA = RUTA_PROYECTO / "data"
RUTA_DATA_RAW = RUTA_DATA / "data-raw"
RUTA_DATA_PROCESSED = RUTA_DATA / "data-processed"