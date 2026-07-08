# -*- coding: utf-8 -*-
"""Fluxograma da metodologia do BERTImputador (estilo Fig. 3 do SAITS).

Gera figuras/fluxograma_bert.{png,svg,pdf}. Reproduzível:
    python figuras/fluxograma_bert.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.estilo import aplicar_estilo, PALETA, TINTA, TINTA_2, EIXO, SUPERFICIE

aplicar_estilo()

AZUL = PALETA["azul"]
VERDE_AGUA = PALETA["verde_agua"]
VERMELHO = PALETA["vermelho"]

fig, ax = plt.subplots(figsize=(8.2, 10.5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 15.2)
ax.axis("off")


def caixa(x, y, w, h, texto, borda=EIXO, fundo=SUPERFICIE, lw=1.2, fs=10.5,
          cor_texto=TINTA, ls="-", peso="normal"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                       linewidth=lw, edgecolor=borda, facecolor=fundo,
                       linestyle=ls, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center",
            fontsize=fs, color=cor_texto, zorder=3, weight=peso, linespacing=1.35)
    return p


def seta(x0, y0, x1, y1, cor=TINTA_2, ls="-", lw=1.6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=13, linewidth=lw, color=cor,
                                 linestyle=ls, zorder=1))


CX, W = 2.1, 5.6          # coluna central (esquerda) e largura das caixas
XM = CX + W / 2           # eixo central das setas

# ---------------------------------------------------------------- entradas
caixa(0.7, 0.35, 4.0, 1.0,
      "Janela com lacunas\n$x \\in \\mathbb{R}^{L}$  ($L$ = 168 h)")
caixa(5.3, 0.35, 4.0, 1.0,
      "Máscara de imputação\n$M \\in \\{0,1\\}^{L}$  (1 = lacuna)")
ax.text(0.7, 0.06, "no treino: máscaras artificiais pontuais + blocos geométricos (razão 25 %)",
        fontsize=8.2, color=TINTA_2, ha="left")

seta(2.7, 1.55, XM - 0.9, 2.20)
seta(7.3, 1.55, XM + 0.9, 2.20)

# ------------------------------------------------- concatenação + projeção
caixa(CX, 2.25, W, 1.05,
      "$x_{\\mathrm{vis}} = x \\odot (1-M)$   (lacunas zeradas)\n"
      "concatenação $[\\,x_{\\mathrm{vis}} \\;\\|\\; M\\,] \\in \\mathbb{R}^{L\\times 2}$")
seta(XM, 3.42, XM, 3.95)

caixa(CX, 4.00, W, 0.80, "Projeção linear   $2 \\rightarrow d_{\\mathrm{model}}$")
seta(XM, 4.92, XM, 5.45)

# ------------------------------------------------ token [MASK] + pos. enc.
caixa(CX, 5.50, W, 0.90,
      "substituição pelo token $[\\mathrm{MASK}]$ aprendível ($\\in \\mathbb{R}^{d}$)\n"
      "nas posições com $M=1$",
      borda=VERDE_AGUA, lw=1.6)
seta(XM, 6.52, XM, 7.05)

caixa(CX, 7.10, W, 0.90,
      "soma com codificação posicional senoidal\n(calculada dinamicamente: qualquer $L$)")
seta(XM, 8.12, XM, 8.65)

# ------------------------------------------------------- encoder (destaque)
enc = caixa(CX - 0.35, 8.70, W + 0.70, 2.90, "", borda=AZUL, lw=1.8)
ax.text(XM, 11.32, "Encoder Transformer bidirecional  ($\\times N$ camadas, pré-norm)",
        ha="center", va="center", fontsize=10.5, color=AZUL, weight="bold")
caixa(CX + 0.25, 8.90, W - 0.50, 0.80,
      "Multi-Head Self-Attention\n(bidirecional: passado E futuro)", fs=9.5)
caixa(CX + 0.25, 9.95, W - 0.50, 0.85, "Feed-Forward (GELU)\n+ residual e LayerNorm", fs=9.5)
seta(XM, 9.82, XM, 9.95)  # MHSA -> FFN (fluxo ascendente dentro da camada)
seta(XM, 11.62, XM, 12.10)

# --------------------------------------------------- cabeça + reconstrução
caixa(CX, 12.15, W, 0.80,
      "LayerNorm final  +  cabeça linear  $d_{\\mathrm{model}} \\rightarrow 1$")
seta(XM, 13.07, XM, 13.55)

caixa(CX, 13.60, W, 0.80, "série reconstruída  $\\tilde{x} \\in \\mathbb{R}^{L}$",
      peso="bold")

# perda (só treino), à direita, tracejada
caixa(7.55, 12.35, 2.25, 1.75,
      "PERDA (treino)\n$\\mathcal{L} = \\mathrm{MSE}(\\tilde{x},\\,x)$\n"
      "APENAS nas posições\ncom $M=1$\n(alvo: $x$ original)",
      borda=VERMELHO, ls="--", fs=8.6, cor_texto=TINTA_2, lw=1.4)
seta(CX + W + 0.12, 14.00, 8.65, 14.10, cor=VERMELHO, ls="--", lw=1.2)

# imputação final
seta(XM, 14.52, XM, 14.90)
caixa(CX, 14.90, W, 0.28, "", borda="none", fundo="none")
ax.text(XM, 15.02, "imputação final:  $\\hat{x} = M \\odot \\tilde{x} + (1-M)\\odot x$"
        "   (observados preservados)", ha="center", fontsize=10.5, color=TINTA,
        weight="bold")

ax.set_title("BERTImputador — reconstrução mascarada para imputação de séries temporais",
             loc="left", fontsize=12, pad=14)

fig.tight_layout()
saida = Path(__file__).resolve().parent
for ext in ("png", "svg", "pdf"):
    fig.savefig(saida / f"fluxograma_bert.{ext}", dpi=300, bbox_inches="tight")
print("salvo: figuras/fluxograma_bert.{png,svg,pdf}")
