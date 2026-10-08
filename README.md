# Práctica 4

**Universidad Nacional Autónoma de México — Facultad de Ciencias**
Curso de Almacenes y Minería de Datos.

**Práctica 4: Medidas de Heterogeneidad y Concentración. Visualización de Datos.
Índices de Gini y Entropía, aplicados al Análisis Exploratorio de la Violencia
contra las Mujeres.**

- Profesora: Jessica Santizo Galicia
- Ayudante: Diego Antonio Villalba González
- Ayudante de Laboratorio: Emma Alicia Jiménez Sánchez

## Objetivo

Partiendo del dataset ENDIREH 2021 ya preprocesado en la Práctica 3
(`data/data-processed/`), calcular e interpretar medidas de **localización**,
**variabilidad**, **heterogeneidad** y **concentración** —incluyendo el
coeficiente de Gini y la entropía de Shannon— y comunicarlas mediante
visualizaciones adecuadas, reflexionando sobre lo que estas medidas revelan y
ocultan acerca de la violencia contra las mujeres.

Se continúa con **polars** como framework de análisis, elegido en la Práctica 3.

## Fuente de Datos

- **Encuesta Nacional sobre la Dinámica de las Relaciones en los Hogares
  (ENDIREH) 2021**, INEGI: <https://www.inegi.org.mx/programas/endireh/2021/>

## Qué se hizo en esta práctica

Sobre el archivo ya limpio de la Práctica 3 se extendió el análisis exploratorio
con cuatro familias de medidas y una síntesis:

1. **Medidas de localización** (`src/eda/medidas_localizacion.py`): percentiles
   (P10, Q1, Q3, P90), moda, mediana y media, más media y mediana ponderadas por
   `factor_expansion`, sobre las variables cuantitativas. Se calculan tanto en el
   dataset crudo como en el procesado para contrastar el efecto de la limpieza.
2. **Medidas de variabilidad** (`src/eda/medidas_variabilidad.py`): rango, IQR,
   varianza muestral, desviación estándar y coeficiente de variación, para la
   población total y desagregadas por `sufrio_violencia_pareja`.
3. **Medidas de heterogeneidad** (`src/eda/medidas_heterogeneidad.py`): sobre
   variables cualitativas, el índice de Gini-Simpson, el IQV y la entropía de
   Shannon (absoluta y normalizada), con proporciones simples y ponderadas.
4. **Medidas de concentración** (`src/eda/medidas_concentracion.py`): curva de
   Lorenz y coeficiente de Gini de los casos de violencia de pareja por entidad
   federativa, contrastados contra la población estimada para no confundir
   concentración de casos con tamaño poblacional.
5. **Síntesis Gini vs. Entropía** (`src/eda/comparacion_gini_entropia.py`): sobre
   la misma variable categórica (`estado_civil_desc`) se comparan la entropía de
   Shannon (heterogeneidad) y el coeficiente de Gini aplicado a las frecuencias
   (concentración).

Las funciones propias de los índices (Gini-Simpson, IQV, entropía, curva de
Lorenz, coeficiente de Gini, medidas ponderadas) viven en el módulo reutilizable
`src/eda/indices.py`, ya que ningún framework de propósito general las implementa
de fábrica. El estilo y las gráficas compartidas están en
`src/visualization/graficas.py`, y las figuras generadas se guardan en `figuras/`.

## Estructura de los notebooks

En `notebooks/` hay un notebook por tipo de análisis solicitado, con las tablas,
gráficas e interpretación de cada medida:

```sh
notebooks/
├── medidas_localizacion.ipynb       # Localización: percentiles, moda, media/mediana (ponderadas)
├── medidas_variabilidad.ipynb       # Variabilidad: rango, IQR, varianza, desv. estándar, CV
├── medidas_heterogeneidad.ipynb     # Heterogeneidad: Gini-Simpson, IQV, entropía de Shannon
├── medidas_concentracion.ipynb      # Concentración: curva de Lorenz y coeficiente de Gini
└── comparacion_gini_entropia.ipynb  # Síntesis: Gini vs. entropía sobre la misma variable
```

## Estructura del Proyecto

```sh
proyecto-endireh-violencia/
│
├── config/
│   └── rutas.py                 # Rutas estáticas de archivos y figuras
│
├── data/
│   ├── data-raw/                # Datos originales, sin modificar
│   ├── data-processed/          # Datos limpios de la Práctica 3
│   ├── data-input-model/        # Datos listos como entrada de un modelo
│   └── data-model/              # Salidas del modelo
│
├── src/
│   ├── cleaning/                # Limpieza y preprocesamiento (Práctica 3)
│   ├── eda/                     # Medidas descriptivas e índices
│   │   ├── indices.py           # Gini, Lorenz, entropía, Gini-Simpson, IQV, ponderadas
│   │   ├── carga.py             # Carga del dataset limpio
│   │   ├── medidas_localizacion.py
│   │   ├── medidas_variabilidad.py
│   │   ├── medidas_heterogeneidad.py
│   │   ├── medidas_concentracion.py
│   │   └── comparacion_gini_entropia.py
│   ├── visualization/           # Estilo y funciones de gráficas (graficas.py)
│   └── models/
│
├── figuras/                     # Figuras generadas por los notebooks
├── notebooks/                   # Un notebook por tipo de análisis
├── README.md
└── requirements.txt
```

## Bibliotecas Utilizadas

- polars
- numpy
- pyarrow
- matplotlib
- seaborn
- ipykernel

## Cómo instalar el entorno

```sh
# Clonar el repositorio
git clone https://github.com/jzarcoo/proyecto-endireh-violencia.git
cd proyecto-endireh-violencia

# Crear y activar el entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

Colocar el archivo de microdatos descargado, sin modificar, en
`data/data-raw/endireh_2021.csv`.

## Cómo ejecutar

Los scripts se ejecutan como módulos desde la raíz del proyecto:

```sh
# Medidas de localización (dataset crudo y procesado)
python -m src.eda.medidas_localizacion

# Medidas de variabilidad
python -m src.eda.medidas_variabilidad

# Medidas de heterogeneidad
python -m src.eda.medidas_heterogeneidad

# Medidas de concentración (curva de Lorenz y Gini)
python -m src.eda.medidas_concentracion

# Síntesis Gini vs. entropía
python -m src.eda.comparacion_gini_entropia
```

## Equipo

- Flores Morán Julieta Melina
- García Landa Brenda Yareli
- Jiménez Rivera Emiliano Kaleb
- Zarco Romero José Antonio
