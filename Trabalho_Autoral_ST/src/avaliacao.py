"""Protocolo de avaliação reprodutível: cenários fixos × métodos de imputação.

Este módulo é o que se reutiliza para avaliar o método em OUTROS datasets:
basta janelar o novo conjunto (src.dados) e chamar `avaliar_metodos`.
As máscaras de avaliação são geradas com semente fixa, então qualquer pessoa
que rode o protocolo obtém exatamente os mesmos números.
"""

import numpy as np
import pandas as pd
import torch

from .baselines import BASELINES
from .mascaramento import mascaras_avaliacao


def imputar_bert_lotes(modelo, x: torch.Tensor, mascara: torch.Tensor, device, batch: int = 256):
    """Reconstrói janelas com o modelo, em lotes (para caber na GPU)."""
    preds = []
    modelo.eval()
    for i in range(0, len(x), batch):
        xb = x[i : i + batch].to(device)
        mb = mascara[i : i + batch].to(device)
        with torch.no_grad():
            preds.append(modelo(xb, mb).cpu())
    return torch.cat(preds)


def _metricas(pred, alvo, masc, desvios=None, medias=None):
    """MAE/RMSE na escala normalizada; se desvios/medias forem dados,
    também na escala original (kW), incluindo o MRE."""
    dif = (pred - alvo)[masc]
    out = {
        "MAE": dif.abs().mean().item(),
        "RMSE": dif.pow(2).mean().sqrt().item(),
    }
    if desvios is not None:
        d = torch.as_tensor(desvios, dtype=torch.float32)[:, None].expand_as(alvo)
        dif_kw = ((pred - alvo) * d)[masc]
        out["MAE_kW"] = dif_kw.abs().mean().item()
        out["RMSE_kW"] = dif_kw.pow(2).mean().sqrt().item()
        if medias is not None:
            m = torch.as_tensor(medias, dtype=torch.float32)[:, None].expand_as(alvo)
            alvo_kw = (alvo * d + m)[masc]
            out["MRE"] = (dif_kw.abs().sum() / alvo_kw.abs().sum().clamp(min=1e-8)).item()
    return out


def avaliar_metodos(
    janelas: np.ndarray,
    modelo,
    device,
    cenarios,
    semente: int = 123,
    desvios: np.ndarray | None = None,
    medias: np.ndarray | None = None,
    batch: int = 256,
) -> pd.DataFrame:
    """Avalia o modelo e os baselines nos mesmos cenários de máscara.

    janelas  – array (N, L), já normalizado
    cenarios – lista de tuplas (nome, tipo, razao); tipo em {'pontual','bloco'}
    desvios/medias – arrays (N,) com o desvio/média da série de origem de
      cada janela, para reportar métricas também na escala original.
    Retorna DataFrame longo: cenário × método × métricas.
    """
    x = torch.from_numpy(np.ascontiguousarray(janelas, dtype="float32"))
    linhas = []
    for nome_c, tipo, razao in cenarios:
        masc = mascaras_avaliacao(len(x), x.shape[1], tipo, razao, semente=semente)
        metodos = {
            "BERT (proposto)": lambda xx, mm: imputar_bert_lotes(modelo, xx, mm, device, batch)
        }
        metodos.update(BASELINES)
        for nome_m, fn in metodos.items():
            pred = fn(x, masc)
            linhas.append(
                {"cenário": nome_c, "método": nome_m, **_metricas(pred, x, masc, desvios, medias)}
            )
    return pd.DataFrame(linhas)
