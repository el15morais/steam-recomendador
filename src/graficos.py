"""
Padrão visual dos gráficos do artigo (Matplotlib + Seaborn).

Paleta categórica validada para daltonismo (ordem fixa, nunca reciclada),
marcas finas, grade discreta e exportação em 200 dpi para o PDF do artigo.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from . import config

# Ordem fixa: série 1 = azul, série 2 = laranja, série 3 = verde-água ...
CORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
AZUL, LARANJA = CORES[0], CORES[1]
TEXTO = "#0b0b0b"
TEXTO_SECUNDARIO = "#52514e"
GRADE = "#e4e3df"


def aplicar_estilo() -> None:
    sns.set_theme(style="whitegrid", palette=CORES)
    plt.rcParams.update({
        "figure.figsize": (8, 4.5),
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelcolor": TEXTO_SECUNDARIO,
        "axes.edgecolor": GRADE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRADE,
        "grid.linewidth": 0.8,
        "xtick.color": TEXTO_SECUNDARIO,
        "ytick.color": TEXTO_SECUNDARIO,
        "lines.linewidth": 2,
        "legend.frameon": False,
    })


def salvar(fig, nome: str) -> None:
    """Salva em reports/figures/<nome>.png (nome com prefixo numérico = ordem no artigo)."""
    config.FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURAS / f"{nome}.png")


def br(valor: float, formato: str = "{:,.0f}") -> str:
    """Formata número no padrão brasileiro: 12,345.6 -> 12.345,6"""
    return formato.format(valor).translate(str.maketrans({",": ".", ".": ","}))


def eixo_br(ax, eixo: str = "x", formato: str = "{:,.0f}") -> None:
    """Rótulos do eixo no padrão brasileiro (120.000 em vez de 120000)."""
    from matplotlib.ticker import FuncFormatter
    alvo = ax.xaxis if eixo == "x" else ax.yaxis
    alvo.set_major_formatter(FuncFormatter(lambda v, _: br(v, formato)))


def rotular_barras(ax, formato: str = "{:,.0f}", horizontal: bool = False) -> None:
    """Rótulo de valor na ponta de cada barra (texto em tinta neutra, não na cor da série)."""
    for container in ax.containers:
        rotulos = [br(v, formato) for v in container.datavalues]
        ax.bar_label(container, labels=rotulos, padding=3, color=TEXTO_SECUNDARIO, fontsize=9)
    if horizontal:
        ax.margins(x=0.12)
    else:
        ax.margins(y=0.12)
