"""
Estilo y funciones de gráficas compartidas por los notebooks de la Práctica 4.

Todos los notebooks llaman a aplicar_estilo() (seaborn, paleta pastel) y guardan
sus figuras con guardar_figura() en la carpeta figuras/ (RUTA_FIGURAS).
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.ticker import FuncFormatter, PercentFormatter

from config.rutas import RUTA_FIGURAS
from src.eda.indices import coeficiente_gini, curva_lorenz

PALETA = sns.color_palette("pastel")


def aplicar_estilo() -> None:
    """Estilo común de las gráficas: seaborn con paleta pastel."""
    sns.set_theme(style="whitegrid", palette="pastel", context="notebook")
    plt.rcParams["figure.dpi"] = 110


def guardar_figura(fig: plt.Figure, nombre: str) -> None:
    """Guarda la figura como figuras/<nombre>.png."""
    RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(RUTA_FIGURAS / f"{nombre}.png", bbox_inches="tight")


def _miles(x, _):
    return f"{x / 1e6:.1f} M" if abs(x) >= 1e6 else f"{x / 1e3:.0f} mil"


def casos_por_entidad(entidades: pl.DataFrame) -> plt.Figure:
    """Casos ponderados por entidad, con su participación en el total de casos."""
    entidades = entidades.sort("casos")
    nombres = [n.title().replace(" De ", " de ").replace(" La ", " la ") for n in entidades["nom_entidad"]]
    y = np.arange(entidades.height)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.barh(y, entidades["casos"], color=PALETA[0], height=0.7)
    for yi, c, p in zip(y, entidades["casos"], entidades["participacion_casos"]):
        ax.text(c, yi, f"  {p:.1%}", va="center", fontsize=7.5)
    ax.xaxis.set_major_formatter(FuncFormatter(_miles))
    ax.set_xlim(0, entidades["casos"].max() * 1.12)
    ax.set_yticks(y, nombres, fontsize=8.5)
    ax.set_ylim(-0.6, entidades.height - 0.4)
    ax.set_title("Casos de violencia de pareja por entidad (ponderados)")
    ax.set_xlabel("Mujeres que reportaron violencia de pareja (etiqueta: % del total de casos)")
    fig.tight_layout()
    return fig


def lorenz(series: dict[str, np.ndarray], titulo: str, etiqueta_unidades: str = "Proporción acumulada de entidades") -> plt.Figure:
    """Curvas de Lorenz de una o más cantidades sobre las mismas unidades, con su Gini en la leyenda."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Igualdad perfecta (G = 0)")
    for (nombre, x), color in zip(series.items(), PALETA):
        u, acum = curva_lorenz(x)
        ax.plot(u, acum, color=color, linewidth=2.2, marker="o", markersize=4, label=f"{nombre} (G = {coeficiente_gini(x):.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel(etiqueta_unidades)
    ax.set_ylabel("Proporción acumulada del total")
    ax.set_title(titulo)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    return fig
