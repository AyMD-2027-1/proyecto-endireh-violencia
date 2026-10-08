"""
Gráficas de medidas de concentración (Práctica 4): casos por entidad y curva de Lorenz.
Cada función regresa la figura para que el notebook la muestre.
"""

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.ticker import FuncFormatter, PercentFormatter

from src.eda.indices import curva_lorenz, coeficiente_gini

# Paleta categórica validada (orden fijo) y tinta de texto
AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRIS_DIAGONAL = "#9a9893"
TEXTO, TEXTO_SECUNDARIO, REJILLA = "#0b0b0b", "#52514e", "#e4e2dc"

ESTILO = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": REJILLA,
    "axes.labelcolor": TEXTO_SECUNDARIO,
    "axes.titlecolor": TEXTO,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": REJILLA,
    "grid.linewidth": 0.8,
    "xtick.color": TEXTO_SECUNDARIO,
    "ytick.color": TEXTO_SECUNDARIO,
    "legend.frameon": False,
    "font.size": 10,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
}
plt.rcParams.update(ESTILO)

def _miles(x, _):
    return f"{x / 1e6:.1f} M" if abs(x) >= 1e6 else f"{x / 1e3:.0f} mil"


def casos_por_entidad(entidades: pl.DataFrame) -> plt.Figure:
    """Casos ponderados por entidad, con su participación en el total de casos."""
    entidades = entidades.sort("casos")
    nombres = [n.title().replace(" De ", " de ").replace(" La ", " la ") for n in entidades["nom_entidad"]]
    y = np.arange(entidades.height)
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.barh(y, entidades["casos"], color=AZUL, height=0.7)
    for yi, c, p in zip(y, entidades["casos"], entidades["participacion_casos"]):
        ax.text(c, yi, f"  {p:.1%}", va="center", fontsize=7.5, color=TEXTO_SECUNDARIO)
    ax.xaxis.set_major_formatter(FuncFormatter(_miles))
    ax.set_xlim(0, entidades["casos"].max() * 1.12)
    ax.set_yticks(y, nombres, fontsize=8.5)
    ax.set_ylim(-0.6, entidades.height - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_title("Casos de violencia de pareja por entidad (ponderados)")
    ax.set_xlabel("Mujeres que reportaron violencia de pareja (etiqueta: % del total de casos)")
    fig.tight_layout()
    return fig


def lorenz(series: dict[str, np.ndarray], titulo: str, etiqueta_unidades: str = "Proporción acumulada de entidades") -> plt.Figure:
    """Curvas de Lorenz de una o más cantidades sobre las mismas unidades, con su Gini en la leyenda."""
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.plot([0, 1], [0, 1], color=GRIS_DIAGONAL, linewidth=1.2, linestyle="--", label="Igualdad perfecta (G = 0)")
    for (nombre, x), color in zip(series.items(), [AZUL, NARANJA, AQUA]):
        u, acum = curva_lorenz(x)
        ax.plot(u, acum, color=color, linewidth=2, marker="o", markersize=3.5, label=f"{nombre} (G = {coeficiente_gini(x):.3f})")
    primera = next(iter(series.values()))
    u, acum = curva_lorenz(primera)
    ax.fill_between(u, acum, u, color=AZUL, alpha=0.08)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel(etiqueta_unidades)
    ax.set_ylabel("Proporción acumulada del total")
    ax.set_title(titulo)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig
