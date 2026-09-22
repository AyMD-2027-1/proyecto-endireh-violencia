"""
Medidas de variabilidad: 
 * Rango
 * Rango Intercuartílico (IQR)
 * Varianza muestral
 * Desviación estándar.

"""

import polars as pl

from config.rutas import RUTA_DATA_RAW

source = RUTA_DATA_RAW / "endireh_2021.csv"

COLUMNAS = ["edad_primer_union", "num_hijos", "ingreso_pareja"]


def calcular_medidas_variabilidad(df: pl.DataFrame, columnas: list[str]) -> pl.DataFrame:
    
    filas = []
    for col in columnas:
        serie = df.get_column(col)

        minimo = serie.min()
        maximo = serie.max()
        rango = maximo - minimo

        q1 = serie.quantile(0.25, interpolation="linear")
        q3 = serie.quantile(0.75, interpolation="linear")
        iqr = q3 - q1

        varianza = serie.var(ddof=1)
        desviacion = serie.std(ddof=1)

        filas.append({
            "variable": col,
            "minimo": minimo,
            "maximo": maximo,
            "rango": rango,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "varianza_muestral": varianza,
            "desviacion_estandar": desviacion,
        })

    return pl.DataFrame(filas)


if __name__ == "__main__":
    df_crudo = pl.read_csv(source)

    resultado = calcular_medidas_variabilidad(df_crudo, COLUMNAS)

    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(resultado)