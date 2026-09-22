"""Limpieza de datos ENDIREH 2021."""

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from config.rutas import RUTA_DATA_RAW

df = pl.read_csv(RUTA_DATA_RAW / "endireh_2021.csv")
print(df.shape)
df.head()

# Variables no consideradas todavía en el EDA previo, se revisan aquí.
schema = df.schema
print("Todas las columnas del dataset:")
for col, tipo in schema.items():
    print(f" {col}: {tipo}")

columnas_no_revisadas = [
    "cve_entidad",
    "cve_municipio",
    "nom_municipio",
    "estrato_socioeconomico",
    "pareja_trabaja_id",
    "pareja_trabaja_desc",
    "dinero_propio_id",
    "dinero_propio_desc",
    "apoyo_gobierno_id",
    "apoyo_gobierno_desc",
    "tiene_ahorros_id",
    "tiene_ahorros_desc",
    "propietaria_vivienda_id",
    "propietaria_vivienda_desc",
    "sufrio_violencia_pareja",
]

resumen_no_revisadas = df.select(
    [pl.col(c).null_count().alias(f"{c}__nulos") for c in columnas_no_revisadas]
)
print(
    resumen_no_revisadas.transpose(
        include_header=True, header_name="columna", column_names=["nulos"]
    )
)

for c in columnas_no_revisadas:
    print(f"\n{c} — valores únicos:")
    print(df[c].value_counts().sort("count", descending=True))

# Por qué existen los nulos: relaciones entre variables.
# edad_primer_union depende de estado_civil_desc (P3_8/P13_14).
total_nulos_edad = df.filter(pl.col("edad_primer_union").is_null()).height
print(f"Total de nulos en edad_primer_union: {total_nulos_edad}")

nulos_edad_por_estado_civil = (
    df.filter(pl.col("edad_primer_union").is_null())
    .group_by("estado_civil_desc")
    .len()
    .sort("len", descending=True)
    .with_columns(
        (pl.col("len") / total_nulos_edad * 100).round(1).alias("% de los nulos")
    )
)
print(nulos_edad_por_estado_civil)

total_nulos_hijos = df.filter(pl.col("num_hijos").is_null()).height
print(f"Total de nulos en num_hijos: {total_nulos_hijos}")

nulos_hijos_por_estado_civil = (
    df.filter(pl.col("num_hijos").is_null())
    .group_by("estado_civil_desc")
    .len()
    .sort("len", descending=True)
    .with_columns(
        (pl.col("len") / total_nulos_hijos * 100).round(1).alias("% de los nulos")
    )
)
print(nulos_hijos_por_estado_civil)

# Consulta de contexto: nulos de ingreso_pareja vs otras variables económicas.
comparacion_economica = (
    df.with_columns(pl.col("ingreso_pareja").is_null().alias("sin_dato_ingreso_pareja"))
    .group_by("sin_dato_ingreso_pareja")
    .agg(
        [
            (pl.col("dinero_propio_desc") == "Sí").mean().alias("% con dinero propio"),
            (pl.col("apoyo_gobierno_desc") == "Sí")
            .mean()
            .alias("% con apoyo de gobierno"),
            (pl.col("tiene_ahorros_desc") == "Sí").mean().alias("% con ahorros"),
        ]
    )
)
print(comparacion_economica)

# Duplicados: se detectan y eliminan considerando todas las columnas.
n_filas_antes = df.height
n_duplicados = df.is_duplicated().sum()
print(f"Filas totales: {n_filas_antes}")
print(f"Filas duplicadas (exactas): {n_duplicados}")

df = df.unique()
n_filas_despues = df.height
print(f"Filas después de eliminar duplicados: {n_filas_despues}")
print(f"Filas eliminadas: {n_filas_antes - n_filas_despues}")

# Códigos especiales a nulos reales (no sabe/no especificado, no top-coding).
count_999997 = df.filter(pl.col("ingreso_pareja") == 999997).height
print(
    f"Registros con ingreso_pareja = 999997 (top-coded, NO es no-respuesta): {count_999997}"
)

df = df.with_columns(
    [
        pl.when(pl.col("edad_primer_union").is_in([98, 99]))
        .then(None)
        .otherwise(pl.col("edad_primer_union"))
        .alias("edad_primer_union"),
        pl.when(pl.col("ingreso_pareja").is_in([999998, 999999]))
        .then(None)
        .otherwise(pl.col("ingreso_pareja"))
        .alias("ingreso_pareja"),
    ]
)

print(df.filter(pl.col("edad_primer_union").is_in([98, 99])).height)  # debe ser 0
print(df.filter(pl.col("ingreso_pareja").is_in([999998, 999999])).height)  # debe ser 0

# Imputación.
total_filas = df.height
for col in ["edad_primer_union", "num_hijos", "ingreso_pareja"]:
    n_nulos = df[col].null_count()
    print(f"{col}: {n_nulos} nulos ({n_nulos / total_filas * 100:.1f}%)")

# edad_primer_union: nulo estructural, no se imputa; se agrega indicador.
df = df.with_columns(pl.col("edad_primer_union").is_null().alias("nunca_tuvo_union"))
print(df["nunca_tuvo_union"].value_counts())

# num_hijos: se imputa con la mediana por estado_civil_desc.
mediana_num_hijos = df["num_hijos"].median()
print(f"Mediana global de num_hijos: {mediana_num_hijos}")

df = df.with_columns(
    pl.col("num_hijos").fill_null(
        pl.col("num_hijos").median().over("estado_civil_desc")
    )
)
print(f"Nulos restantes en num_hijos: {df['num_hijos'].null_count()}")

# ingreso_pareja: se verifica si los nulos se explican por estado_civil_desc
# o pareja_trabaja_desc antes de decidir si se imputan.
total_nulos_ingreso = df.filter(pl.col("ingreso_pareja").is_null()).height
print(f"Total de nulos en ingreso_pareja: {total_nulos_ingreso}")

nulos_por_estado_civil = (
    df.filter(pl.col("ingreso_pareja").is_null())
    .group_by("estado_civil_desc")
    .len()
    .sort("len", descending=True)
    .with_columns(
        (pl.col("len") / total_nulos_ingreso * 100).round(1).alias("% de los nulos")
    )
)
print(nulos_por_estado_civil)

nulos_por_pareja_trabaja = (
    df.filter(pl.col("ingreso_pareja").is_null())
    .group_by("pareja_trabaja_desc")
    .len()
    .sort("len", descending=True)
    .with_columns(
        (pl.col("len") / total_nulos_ingreso * 100).round(1).alias("% de los nulos")
    )
)
print(nulos_por_pareja_trabaja)

cruce_nulos = (
    df.filter(pl.col("ingreso_pareja").is_null())
    .group_by(["estado_civil_desc", "pareja_trabaja_desc"])
    .len()
    .sort("len", descending=True)
)
print(cruce_nulos)

nulos_sin_explicacion = df.filter(
    pl.col("ingreso_pareja").is_null()
    & (pl.col("estado_civil_desc") != "Soltera")
    & (pl.col("pareja_trabaja_desc") == "Sí")
).height
print(f"Nulos sin explicación estructural aparente: {nulos_sin_explicacion}")

nulos_por_estrato = (
    df.filter(pl.col("ingreso_pareja").is_null())
    .group_by("estrato_socioeconomico")
    .len()
    .sort("len", descending=True)
)
print(nulos_por_estrato)

# Solo se imputa el remanente sin explicación estructural.
mediana_ingreso = df.filter(
    (pl.col("estado_civil_desc") != "Soltera")
    & (pl.col("pareja_trabaja_desc") == "Sí")
    & pl.col("ingreso_pareja").is_not_null()
)["ingreso_pareja"].median()
print(
    f"Mediana de ingreso_pareja (con pareja que trabaja, dato reportado): {mediana_ingreso}"
)

df = df.with_columns(
    pl.when(
        pl.col("ingreso_pareja").is_null()
        & (pl.col("estado_civil_desc") != "Soltera")
        & (pl.col("pareja_trabaja_desc") == "Sí")
    )
    .then(mediana_ingreso)
    .otherwise(pl.col("ingreso_pareja"))
    .alias("ingreso_pareja")
)

for col in ["edad_primer_union", "num_hijos", "ingreso_pareja"]:
    n_nulos = df[col].null_count()
    print(f"{col}: {n_nulos} nulos restantes")


# Outliers: se detectan con IQR pero no se eliminan, son parte del fenómeno de estudio.
def detectar_outliers_iqr(df, columna):
    q1 = df[columna].quantile(0.25)
    q3 = df[columna].quantile(0.75)
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    outliers = df.filter(
        (pl.col(columna) < limite_inferior) | (pl.col(columna) > limite_superior)
    )
    print(f"{columna}: límites [{limite_inferior:.2f}, {limite_superior:.2f}]")
    print(
        f"  Outliers detectados: {outliers.height} de {df.height} ({outliers.height / df.height * 100:.2f}%)"
    )
    return outliers


for col in ["edad_primer_union", "num_hijos", "ingreso_pareja"]:
    detectar_outliers_iqr(df, col)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, col in zip(axes, ["edad_primer_union", "num_hijos", "ingreso_pareja"]):
    sns.boxplot(y=df[col].to_list(), ax=ax)
    ax.set_title(col)
plt.tight_layout()
plt.show()

# Conversión de variables Sí/No a binario (Sí=1, No=0).
# Primero se verifica que las columnas _id (1/2) coincidan con las _desc.
pares_id_desc = [
    ("pareja_trabaja_id", "pareja_trabaja_desc"),
    ("dinero_propio_id", "dinero_propio_desc"),
    ("apoyo_gobierno_id", "apoyo_gobierno_desc"),
    ("tiene_ahorros_id", "tiene_ahorros_desc"),
    ("propietaria_vivienda_id", "propietaria_vivienda_desc"),
]

for id_col, desc_col in pares_id_desc:
    cruce = df.group_by([id_col, desc_col]).len().sort(id_col)
    print(f"--- {id_col} vs {desc_col} ---")
    print(cruce)

columnas_si_no_desc = [
    "pareja_trabaja_desc",
    "dinero_propio_desc",
    "apoyo_gobierno_desc",
    "tiene_ahorros_desc",
    "propietaria_vivienda_desc",
]

df = df.with_columns(
    [
        pl.col(col).replace({"Sí": 1, "No": 0}).cast(pl.Int8).alias(col)
        for col in columnas_si_no_desc
    ]
)

# Confirmado arriba: 1 = Sí, 2 = No en las columnas _id. Se recodifican a 1/0.
columnas_si_no_id = [
    "pareja_trabaja_id",
    "dinero_propio_id",
    "apoyo_gobierno_id",
    "tiene_ahorros_id",
    "propietaria_vivienda_id",
]

df = df.with_columns(
    [
        pl.col(col).replace({1: 1, 2: 0}).cast(pl.Int8).alias(col)
        for col in columnas_si_no_id
    ]
)

for col in columnas_si_no_desc + columnas_si_no_id:
    print(f"{col}: {df.schema[col]}")
    print(df[col].value_counts().sort(col))

# Guardar el dataset limpio.
from config.rutas import RUTA_DATA_PROCESSED

ruta_salida = RUTA_DATA_PROCESSED / "endireh_2021_clean.csv"
df.write_csv(ruta_salida)
print(f"Dataset limpio guardado en: {ruta_salida}")
print(df.shape)
