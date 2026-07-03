"""Estratégias de mascaramento para o pré-treino auto-supervisionado.

O objetivo BERT adaptado a séries temporais: esconder uma fração dos passos
de tempo e treinar o modelo para reconstruí-los. Duas estratégias:

* pontual    – cada passo é mascarado independentemente (falhas isoladas);
* geométrica – blocos contíguos com comprimento médio `lm`, como no TST
  (Zerveas et al., 2021); simula falhas longas de sensor, o caso mais
  difícil e realista de dados faltantes.

Implementação vetorizada por lote (a cadeia de Markov de 2 estados é
simulada para todas as sequências ao mesmo tempo, passo a passo).
"""

import numpy as np
import torch


def _mascaras_pontuais(n: int, seq_len: int, razao: float, rng) -> np.ndarray:
    return rng.random((n, seq_len)) < razao


def _mascaras_geometricas(n: int, seq_len: int, razao: float, lm: float, rng) -> np.ndarray:
    """Blocos mascarados com comprimento ~ Geométrica(1/lm).

    Cadeia de 2 estados (mascarado/observado) com transições calibradas para
    que a fração esperada mascarada seja `razao`.
    """
    p_sair = 1.0 / lm                            # sair do estado mascarado
    p_entrar = p_sair * razao / (1.0 - razao)    # entrar no estado mascarado
    m = np.zeros((n, seq_len), dtype=bool)
    estado = rng.random(n) < razao
    for t in range(seq_len):
        m[:, t] = estado
        u = rng.random(n)
        estado = np.where(estado, u >= p_sair, u < p_entrar)
    return m


def _garantir_validas(m: np.ndarray, rng) -> np.ndarray:
    """Garante ao menos 1 posição mascarada e 1 observada por sequência."""
    seq_len = m.shape[1]
    for i in np.flatnonzero(~m.any(axis=1)):
        m[i, rng.integers(seq_len)] = True
    for i in np.flatnonzero(m.all(axis=1)):
        m[i, rng.integers(seq_len)] = False
    return m


def gerar_mascaras_lote(
    n: int,
    seq_len: int,
    razao: float = 0.25,
    prob_bloco: float = 0.5,
    lm: float = 6.0,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Gera máscaras para um lote, misturando as duas estratégias.

    Cada sequência usa mascaramento em blocos com probabilidade `prob_bloco`,
    caso contrário pontual. Retorna tensor bool (n, seq_len); True = posição
    mascarada (a imputar).
    """
    if rng is None:
        rng = np.random.default_rng()
    usa_bloco = rng.random(n) < prob_bloco
    pontual = _mascaras_pontuais(n, seq_len, razao, rng)
    blocos = _mascaras_geometricas(n, seq_len, razao, lm, rng)
    m = np.where(usa_bloco[:, None], blocos, pontual)
    return torch.from_numpy(_garantir_validas(m, rng))


def trechos_mascarados(m: np.ndarray) -> np.ndarray:
    """Trechos contíguos mascarados de uma máscara 1-D: array (k, 2) [início, fim)."""
    bordas = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
    return bordas.reshape(-1, 2)


def mascaras_avaliacao(
    n: int, seq_len: int, cenario: str, razao: float, semente: int = 123, lm: float = 6.0
) -> torch.Tensor:
    """Máscaras FIXAS (semente determinística) para avaliação reprodutível.

    cenario: 'pontual' ou 'bloco'.
    """
    rng = np.random.default_rng(semente)
    if cenario == "pontual":
        m = _mascaras_pontuais(n, seq_len, razao, rng)
    elif cenario == "bloco":
        m = _mascaras_geometricas(n, seq_len, razao, lm, rng)
    else:
        raise ValueError(f"cenário desconhecido: {cenario}")
    return torch.from_numpy(_garantir_validas(m, rng))
