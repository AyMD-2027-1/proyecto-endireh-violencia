"""Rutas estáticas de archivos del proyecto ENDIREH 2021."""

from pathlib import Path

RUTA_PROYECTO = Path(__file__).resolve().parent.parent

# Rutas de archivos de datos
RUTA_DATA = RUTA_PROYECTO / "data"
RUTA_DATA_RAW = RUTA_DATA / "data-raw"
RUTA_DATA_PROCESSED = RUTA_DATA / "data-processed"
RUTA_DATA_INPUT_MODEL = RUTA_DATA / "data-input-model"
RUTA_DATA_MODEL = RUTA_DATA / "data-model"

# Archivos de datos
ARCHIVO_RAW = RUTA_DATA_RAW / "endireh_2021.csv"
ARCHIVO_PROCESSED = RUTA_DATA_PROCESSED / "endireh_2021_clean.csv"

# Carpeta donde los notebooks guardan sus figuras
RUTA_FIGURAS = RUTA_PROYECTO / "figuras"
