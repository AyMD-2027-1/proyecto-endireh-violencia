"""
Medidas de variabilidad:
 * Rango
 * Rango Intercuartílico (IQR)
 * Varianza muestral
 * Desviación estándar
 * Coeficiente de variación (CV)

"""

import polars as pl

from config.rutas import RUTA_DATA_RAW, RUTA_DATA_PROCESSED

FUENTES = {
    "crudo": RUTA_DATA_RAW / "endireh_2021.csv",
    "procesado": RUTA_DATA_PROCESSED / "endireh_2021_clean.csv",
}

COLUMNAS = ["edad_primer_union", "num_hijos", "ingreso_pareja"]
COL_GRUPO = "sufrio_violencia_pareja"


def calcular_medidas_variabilidad(df: pl.DataFrame, columnas: list[str], grupo: str | None = None) -> pl.DataFrame:
    filas = []

    if grupo is None:
        grupos = [(None, df)]
    else:
        grupos = [(g, sub) for g, sub in df.group_by(grupo)]

    for valor_grupo, sub_df in grupos:
        for col in columnas:
            if col not in sub_df.columns:
                continue
            serie = sub_df.get_column(col)

            minimo = serie.min()
            maximo = serie.max()
            rango = maximo - minimo

            q1 = serie.quantile(0.25, interpolation="linear")
            q3 = serie.quantile(0.75, interpolation="linear")
            iqr = q3 - q1

            varianza = serie.var(ddof=1)
            desviacion = serie.std(ddof=1)

            media = serie.mean()
            cv = (desviacion / media * 100) if media not in (None, 0) else None

            fila = {
                "variable": col,
                "n": serie.drop_nulls().len(),
                "media": media,
                "minimo": minimo,
                "maximo": maximo,
                "rango": rango,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "varianza_muestral": varianza,
                "desviacion_estandar": desviacion,
                "cv_%": cv,
            }
            if grupo is not None:
                fila = {grupo: valor_grupo, **fila}

            filas.append(fila)

    return pl.DataFrame(filas)


if __name__ == "__main__":
    for etiqueta, ruta in FUENTES.items():
        print(f"\n{'='*80}\nDATASET: {etiqueta.upper()}  ({ruta.name})\n{'='*80}")

        df = pl.read_csv(ruta)

        print(f"\n         Medidas de variabilidad ({etiqueta}): población total           ")
        resultado_total = calcular_medidas_variabilidad(df, COLUMNAS)
        with pl.Config(tbl_cols=-1, tbl_width_chars=250):
            print(resultado_total)

        if COL_GRUPO in df.columns:
            print(f"\n            Medidas de variabilidad ({etiqueta}) por grupo: {COL_GRUPO} (1 = sufrió violencia, 0 = no)     ")
            resultado_por_grupo = calcular_medidas_variabilidad(df, COLUMNAS, grupo=COL_GRUPO)
            resultado_por_grupo = resultado_por_grupo.sort([COL_GRUPO, "variable"])
            with pl.Config(tbl_cols=-1, tbl_width_chars=250):
                print(resultado_por_grupo)
        else:
            print(f"\n(La columna '{COL_GRUPO}' no está en el dataset {etiqueta}; se omite la comparación por grupo)")