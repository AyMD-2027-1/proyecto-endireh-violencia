import polars as pl
from config.rutas import RUTA_DATA_RAW

df_crudo = pl.read_csv(RUTA_DATA_RAW / "endireh_2021.csv")
print(df_crudo.shape)
print(df_crudo.head())