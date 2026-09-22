# Práctica 3 - Técnicas y Frameworks de Minería de Datos. Arquitectura de Proyectos de Datos, Preprocesamiento y Análisis Exploratorio sobre Violencia contra las Mujeres

Curso de Almacenes y Minería de Datos — Facultad de Ciencias, UNAM.

## Objetivo del Proyecto

El objetivo de esta práctica es aprender a estructurar correctamente un proyecto de minería de datos siguiendo una arquitectura de carpetas estándar en la industria, seleccionar y justificar el uso de un framework de análisis de datos (pandas, polars o PySpark), aplicar un preprocesamiento riguroso (normalización, eliminación de duplicados, conversión de tipos de datos e imputación) sobre un conjunto de datos real del gobierno mexicano relacionado con la violencia contra las mujeres, y calcular e interpretar medidas de localización y variabilidad como parte del análisis exploratorio, previo a cualquier modelado.

Para este proyecto se seleccionó **polars** como framework de análisis de datos.

## Fuente de Datos

- **Encuesta Nacional sobre la Dinámica de las Relaciones en los Hogares
  (ENDIREH) 2021**, levantada por el Instituto Nacional de Estadística y Geografía (INEGI): <https://www.inegi.org.mx/programas/endireh/2021/>

## Estructura del Proyecto

<!-- ToDO: Actualizar conforme se agreguen-->

```sh
proyecto-endireh-violencia/
│
├── config/
│   └── rutas.py            # Rutas de archivos estaticos
│
├── data/
│   ├── data-raw/           # Datos originales, tal como se descargaron (nunca se editan)
│   ├── data-processed/     # Datos ya limpios: sin duplicados, tipos corregidos, imputados
│   ├── data-input-model/   # Datos ya transformados y listos como entrada de un modelo
│   └── data-model/         # Salidas del modelo: predicciones, clusters, reglas obtenidas
│
├── src/
│   ├── cleaning/           # Scripts de limpieza y preprocesamiento
│   ├── eda/                # Scripts de exploración
│   ├── visualization/      # Scripts de graficas
│   └── models/             # Scripts de entrenamiento y evaluacion de modelos
│
├── notebooks/              # Notebooks exploratorios (no productivos)
├── README.md               # Documentacion del proyecto
└── requirements.txt        # Dependencias exactas del proyecto
```

## Bibliotecas Utilizadas

<!-- ToDO: Actualizar conforme se agreguen -->

- polars
- pyarrow
- ipykernel
- matplotlib
- seaborn


## Cómo instalar el entorno

```sh
# Clonar el repositorio
git clone https://github.com/jzarcoo/proyecto-endireh-violencia.git

# Entrar al directorio del proyecto
cd proyecto-endireh-violencia

# Crear entorno virtual
python3 -m venv .venv

# Activarlo
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

Colocar el archivo de microdatos descargado, sin modificar, en `data/data-raw/endireh_2021.csv`.

## Cómo ejecutar un archivo

```sh
python -m src.eda.load_raw_data
```

## Cómo ejecutar el pipeline

El pipeline se ejecuta en orden desde la raíz del proyecto:

```sh
# 1. Limpieza y preprocesamiento: genera data/data-processed/endireh_2021_clean.csv
python -m src.cleaning.limpieza_de_datos

# 2. Medidas de localización sobre el dataset ya limpio
python -m src.eda.medidas_localizacion

# 3. Medidas de variabilidad sobre el dataset ya limpio
python -m src.eda.medidas_variabilidad

# 4. Distribución de las variables cuantitativas
python -m src.visualization.distribuciones
```

## Equipo

- Flores Morán Julieta Melina
- García Landa Brenda Yareli
- Jiménez Rivera Emiliano Kaleb
- Zarco Romero José Antonio
