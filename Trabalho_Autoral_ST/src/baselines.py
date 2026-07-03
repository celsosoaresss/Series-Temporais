"""Baselines clássicos de imputação, para comparação com o BERT."""

import numpy as np
import pandas as pd
import torch


def _aplicar_por_janela(x: torch.Tensor, mascara: torch.Tensor, fn) -> torch.Tensor:
    """Aplica `fn(Series com NaN) -> Series imputada` janela a janela."""
    x_np = x.cpu().numpy().astype("float64")
    m_np = mascara.cpu().numpy()
    saida = x_np.copy()
    for i in range(len(x_np)):
        s = pd.Series(x_np[i])
        s[m_np[i]] = np.nan
        saida[i] = fn(s).to_numpy()
    return torch.from_numpy(saida.astype("float32"))


def imputar_media(x, mascara):
    """Média dos pontos observados da própria janela."""
    return _aplicar_por_janela(x, mascara, lambda s: s.fillna(s.mean()))


def imputar_ffill(x, mascara):
    """Repete o último valor observado (e bfill para o início)."""
    return _aplicar_por_janela(x, mascara, lambda s: s.ffill().bfill())


def imputar_linear(x, mascara):
    """Interpolação linear entre pontos observados."""
    return _aplicar_por_janela(
        x, mascara, lambda s: s.interpolate(method="linear", limit_direction="both")
    )


BASELINES = {
    "média da janela": imputar_media,
    "último valor (ffill)": imputar_ffill,
    "interpolação linear": imputar_linear,
}
