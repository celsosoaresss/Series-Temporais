"""Estilo visual dos gráficos (paleta categórica validada para CVD, modo claro)."""

import matplotlib as mpl

# ordem fixa das séries — nunca reordenar/reciclar (segurança para daltonismo)
PALETA = {
    "azul": "#2a78d6",
    "verde_agua": "#1baf7a",
    "amarelo": "#eda100",
    "verde": "#008300",
    "violeta": "#4a3aa7",
    "vermelho": "#e34948",
    "magenta": "#e87ba4",
    "laranja": "#eb6834",
}
SERIES = list(PALETA.values())

TINTA = "#0b0b0b"          # texto principal
TINTA_2 = "#52514e"        # texto secundário
SUAVE = "#898781"          # eixos e rótulos
GRADE = "#e1e0d9"          # linhas de grade
EIXO = "#c3c2b7"           # linha do eixo
SUPERFICIE = "#fcfcfb"     # fundo do gráfico
DESTAQUE_MASCARA = "#e7e6e0"  # faixa neutra p/ trechos mascarados


def aplicar_estilo():
    mpl.rcParams.update({
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "axes.edgecolor": EIXO,
        "axes.labelcolor": TINTA_2,
        "axes.titlecolor": TINTA,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRADE,
        "grid.linewidth": 0.8,
        "xtick.color": SUAVE,
        "ytick.color": SUAVE,
        "text.color": TINTA,
        "lines.linewidth": 1.8,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "font.family": "sans-serif",
        "figure.dpi": 110,
        "legend.frameon": False,
    })
