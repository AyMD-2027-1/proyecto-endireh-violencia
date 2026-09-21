# Práctica 3 - Técnicas y Frameworks de Minería de Datos. Arquitectura de Proyectos de Datos, Preprocesamiento y Análisis Exploratorio sobre Violencia contra las Mujeres

Curso de Almacenes y Minería de Datos — Facultad de Ciencias, UNAM.

## Objetivo del Proyecto

## Fuente de Datos

- **Encuesta Nacional sobre la Dinámica de las Relaciones en los Hogares
  (ENDIREH) 2021**, levantada por el Instituto Nacional de Estadística y Geografía (INEGI): <https://www.inegi.org.mx/programas/endireh/2021/>

## Estructura del Proyecto

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
│   ├── visualization/      # Scripts de graficas y EDA
│   └── models/             # Scripts de entrenamiento y evaluacion de modelos
│
├── notebooks/              # Notebooks exploratorios (no productivos)
├── README.md               # Documentacion del proyecto
└── requirements.txt        # Dependencias exactas del proyecto
```

## Bibliotecas Utilizadas

<!-- ToDO: Actualizar conforme se agreguen -->

- Polars
- Pyarrow


## Cómo instalar el entorno

```sh
git clone https://github.com/jzarcoo/proyecto-endireh-violencia.git

cd proyecto-endireh-violencia

# Crear entorno virtual
python3 -m venv .venv

# Activarlo
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Cómo ejecutar el pipeline

## Equipo

- Flores Morán Julieta Melina
- García Landa Brenda Yareli
- Jiménez Rivera Emiliano Kaleb
- Zarco Romero José Antonio
