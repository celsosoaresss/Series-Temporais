"""Carregamento, limpeza, divisão temporal, normalização e janelamento.

Todas as funções são genéricas: recebem um DataFrame com índice datetime e
uma coluna por série (cliente/sensor). Assim, o mesmo pipeline se aplica a
qualquer outro dataset de séries temporais (requisito de reprodutibilidade).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Carregamento específico do LD2011_2014 (UCI ElectricityLoadDiagrams)
# ---------------------------------------------------------------------------

def carregar_ld2011(caminho: str, freq: str = "1h") -> pd.DataFrame:
    """Lê o arquivo original (separador ';', decimal ',') e reamostra.

    O dado original é kW médio a cada 15 min; a média horária preserva a
    unidade (kW). Retorna DataFrame indexado por datetime, uma coluna por
    cliente (MT_001..MT_370).
    """
    df = pd.read_csv(caminho, sep=";", decimal=",", index_col=0, parse_dates=True)
    df = df.astype("float32")
    if freq is not None:
        df = df.resample(freq).mean()
    return df


def selecionar_clientes(
    df: pd.DataFrame,
    inicio: str,
    fim: str,
    max_frac_zeros: float = 0.05,
    n_clientes: int | None = None,
    semente: int = 42,
) -> pd.DataFrame:
    """Recorta o período [inicio, fim] e mantém clientes 'completos'.

    No LD2011_2014, zeros no início da série significam que o cliente ainda
    não existia (ausência de medição, não consumo nulo). Clientes com mais de
    `max_frac_zeros` de zeros no período são descartados. Se `n_clientes` for
    dado, sorteia (com semente fixa) um subconjunto reprodutível.
    """
    df = df.loc[inicio:fim]
    frac_zeros = (df == 0).mean()
    elegiveis = frac_zeros[frac_zeros <= max_frac_zeros].index.tolist()
    if n_clientes is not None and n_clientes < len(elegiveis):
        rng = np.random.default_rng(semente)
        elegiveis = sorted(rng.choice(elegiveis, size=n_clientes, replace=False))
    return df[elegiveis]


# ---------------------------------------------------------------------------
# Divisão temporal e normalização (genéricas)
# ---------------------------------------------------------------------------

def dividir_temporal(df: pd.DataFrame, fim_treino: str, fim_val: str):
    """Divide em treino/validação/teste por tempo (sem vazamento).

    treino: início..fim_treino | validação: ..fim_val | teste: ..final.
    """
    treino = df.loc[:fim_treino]
    val = df.loc[fim_treino:fim_val].iloc[1:]
    teste = df.loc[fim_val:].iloc[1:]
    return treino, val, teste


def ajustar_normalizador(treino: pd.DataFrame) -> pd.DataFrame:
    """Calcula média e desvio por série USANDO APENAS O TREINO."""
    stats = pd.DataFrame({"media": treino.mean(), "desvio": treino.std().replace(0, 1.0)})
    return stats


def normalizar(df: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    return (df - stats["media"]) / stats["desvio"]


def desnormalizar(valores: np.ndarray, media: float, desvio: float) -> np.ndarray:
    return valores * desvio + media


# ---------------------------------------------------------------------------
# Janelamento: cada série vira janelas univariadas de comprimento fixo
# ---------------------------------------------------------------------------

def janelar(df: pd.DataFrame, comprimento: int, passo: int):
    """Converte cada coluna em janelas (canal-independente / univariado).

    Retorna:
      janelas  – array (N, comprimento) float32
      id_serie – array (N,) int, índice da coluna de origem de cada janela
    A abordagem univariada torna o modelo agnóstico ao número de séries do
    dataset, permitindo aplicá-lo diretamente a outros conjuntos de dados.
    """
    todas, ids = [], []
    valores = df.to_numpy(dtype="float32")
    T = len(df)
    inicios = range(0, T - comprimento + 1, passo)
    for j in range(df.shape[1]):
        for i in inicios:
            todas.append(valores[i : i + comprimento, j])
            ids.append(j)
    return np.stack(todas), np.asarray(ids, dtype="int64")


def salvar_processado(pasta: str, config: dict, stats: pd.DataFrame, **arrays):
    """Salva janelas (.npz), estatísticas do normalizador e config (.json)."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pasta / "janelas.npz", **arrays)
    stats.to_json(pasta / "normalizador.json", orient="index")
    with open(pasta / "config_dados.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def carregar_processado(pasta: str):
    """Carrega o que `salvar_processado` gravou."""
    pasta = Path(pasta)
    arrays = dict(np.load(pasta / "janelas.npz"))
    stats = pd.read_json(pasta / "normalizador.json", orient="index")
    with open(pasta / "config_dados.json", encoding="utf-8") as f:
        config = json.load(f)
    if "clientes" in config:  # garante a mesma ordem de colunas usada no janelamento
        stats = stats.reindex(config["clientes"])
    return arrays, stats, config


class JanelasDataset(Dataset):
    """Dataset PyTorch de janelas univariadas já normalizadas."""

    def __init__(self, janelas: np.ndarray):
        self.janelas = torch.from_numpy(np.ascontiguousarray(janelas, dtype="float32"))

    def __len__(self):
        return len(self.janelas)

    def __getitem__(self, i):
        return self.janelas[i]
