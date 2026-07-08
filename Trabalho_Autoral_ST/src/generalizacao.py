"""Generalização a outros datasets sob o PROTOCOLO DO ARTIGO do SAITS.

Reproduz, para Air-Quality (Beijing), ETT e PhysioNet-2012, o protocolo da
Seção 4.1 de Du et al. (2023): mesmos períodos de divisão (teste PRIMEIRO,
depois validação, depois treino), mesmos comprimentos de amostra, padronização
por variável (estatísticas SÓ do treino) e avaliação eliminando 10 % dos
valores OBSERVADOS de validação/teste como verdade (MAE/RMSE/MRE na escala
padronizada, somas globais como nas eqs. do artigo). Assim os números do nosso
modelo podem ser postos ao lado da Tabela 2 do artigo — com a ressalva de que
os 10 % eliminados são um sorteio nosso (semente fixa), não o deles.

Diferença metodológica mantida de propósito: nosso método é univariado
(canal-independente — cada coluna/variável vira janelas próprias), enquanto o
SAITS é multivariado. Faltantes NATIVOS são suportados aqui: a máscara
artificial de treino é sorteada apenas sobre posições observadas; a entrada do
modelo marca nativos ∪ artificiais; perda e métricas usam só os artificiais.
"""

import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .mascaramento import gerar_mascaras_lote
from .treino import fixar_semente, perda_mascarada

# 12 registros descartados pelo artigo (sem série temporal alguma)
PHYSIONET_DESCARTADOS = {
    "147514", "142731", "145611", "140501", "155655", "143656",
    "156254", "150309", "140936", "141264", "150649", "142998",
}
# descritores estáticos do PhysioNet (não são séries temporais)
PHYSIONET_ESTATICOS = {"RecordID", "Age", "Gender", "Height", "ICUType"}

VARS_AIRQUALITY = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3",
                   "TEMP", "PRES", "DEWP", "RAIN", "WSPM"]


# ---------------------------------------------------------------------------
# Carregamento dos datasets
# ---------------------------------------------------------------------------

def carregar_airquality(pasta: str) -> pd.DataFrame:
    """12 estações × 11 variáveis contínuas → DataFrame (35064 h, 132 colunas)."""
    arquivos = sorted(Path(pasta).glob("PRSA_Data_*.csv"))
    if not arquivos:
        raise FileNotFoundError(f"nenhum PRSA_Data_*.csv em {pasta}")
    colunas = {}
    for arq in arquivos:
        df = pd.read_csv(arq)
        estacao = df["station"].iloc[0]
        idx = pd.to_datetime(df[["year", "month", "day", "hour"]])
        for var in VARS_AIRQUALITY:
            colunas[f"{estacao}_{var}"] = pd.Series(df[var].to_numpy("float32"), index=idx)
    return pd.DataFrame(colunas).sort_index()


def carregar_ett(caminho: str) -> pd.DataFrame:
    """ETTm1 (15 min): DataFrame (69680, 7) indexado por datetime."""
    df = pd.read_csv(caminho, parse_dates=["date"], index_col="date")
    return df.astype("float32").sort_index()


def carregar_physionet(pasta: str, cache: str | None = None):
    """Grade horária dos 3 conjuntos (a/b/c): (N, 48, 37) + ids + variáveis.

    Cada paciente vira uma grade 48 h × variáveis (média dentro da hora);
    NaN = não medido. Usa/gera um cache .npz porque a leitura dos ~12 mil
    arquivos leva minutos.
    """
    if cache and Path(cache).exists():
        d = np.load(cache, allow_pickle=True)
        return d["dados"], list(d["ids"]), list(d["variaveis"])

    registros = {}
    for tar_nome in ("physionet_set-a.tar.gz", "physionet_set-b.tar.gz",
                     "physionet_set-c.tar.gz"):
        with tarfile.open(Path(pasta) / tar_nome, "r:gz") as tar:
            for membro in tar.getmembers():
                if not membro.name.endswith(".txt"):
                    continue
                rid = Path(membro.name).stem
                if rid in PHYSIONET_DESCARTADOS:
                    continue
                df = pd.read_csv(tar.extractfile(membro))
                df = df[~df["Parameter"].isin(PHYSIONET_ESTATICOS)]
                df = df.dropna(subset=["Parameter"])
                if len(df) == 0:
                    continue
                horas = df["Time"].str.split(":").str[0].astype(int).clip(0, 47)
                df = df.assign(hora=horas)
                registros[rid] = df.groupby(["hora", "Parameter"])["Value"].mean()

    variaveis = sorted({v for r in registros.values() for v in r.index.get_level_values(1)})
    ids = sorted(registros)
    dados = np.full((len(ids), 48, len(variaveis)), np.nan, dtype="float32")
    pos_var = {v: j for j, v in enumerate(variaveis)}
    for i, rid in enumerate(ids):
        r = registros[rid]
        h = r.index.get_level_values(0).to_numpy()
        v = np.array([pos_var[x] for x in r.index.get_level_values(1)])
        dados[i, h, v] = r.to_numpy("float32")

    if cache:
        np.savez_compressed(cache, dados=dados, ids=np.array(ids),
                            variaveis=np.array(variaveis))
    return dados, ids, variaveis


# ---------------------------------------------------------------------------
# Janelamento univariado com faltantes nativos + protocolo do artigo
# ---------------------------------------------------------------------------

def _janelar_faltantes(valores: np.ndarray, comprimento: int, passo: int):
    """(T, D) com NaN → janelas (N, L), máscara nativa (N, L) e id da série."""
    T, D = valores.shape
    inicios = range(0, T - comprimento + 1, passo)
    xs, ids = [], []
    for j in range(D):
        for i in inicios:
            xs.append(valores[i : i + comprimento, j])
            ids.append(j)
    x = np.stack(xs).astype("float32")
    m_nat = np.isnan(x)
    return np.nan_to_num(x, nan=0.0), m_nat, np.asarray(ids, dtype="int64")


def preparar_grade_artigo(df: pd.DataFrame, fim_teste: str, fim_val: str,
                          comprimento: int, passo: int, min_observados: int = 2):
    """Divide como no artigo (TESTE primeiro), padroniza com stats do treino e
    janela cada coluna. Retorna dict split → (x, m_nat, id_serie) + stats."""
    teste = df.loc[:fim_teste]
    val = df.loc[fim_teste:fim_val].iloc[1:]
    treino = df.loc[fim_val:].iloc[1:]

    media = treino.mean()
    desvio = treino.std().replace(0, 1.0)
    saida = {}
    for nome, parte in (("treino", treino), ("val", val), ("teste", teste)):
        z = ((parte - media) / desvio).to_numpy("float32")
        x, m_nat, ids = _janelar_faltantes(z, comprimento, passo)
        validas = (~m_nat).sum(axis=1) >= min_observados
        saida[nome] = (x[validas], m_nat[validas], ids[validas])
    return saida, pd.DataFrame({"media": media, "desvio": desvio})


def preparar_physionet_artigo(dados: np.ndarray, semente: int = 42,
                              min_observados: int = 2):
    """Split 80/20 (teste) e 20 % do treino como validação, por paciente;
    padronização por variável com stats do treino; janelas univariadas
    (paciente × variável) de 48 passos."""
    rng = np.random.default_rng(semente)
    n = len(dados)
    ordem = rng.permutation(n)
    n_teste = int(round(0.2 * n))
    idx_teste = ordem[:n_teste]
    resto = ordem[n_teste:]
    n_val = int(round(0.2 * len(resto)))
    idx_val, idx_treino = resto[:n_val], resto[n_val:]

    treino = dados[idx_treino]
    media = np.nanmean(treino.reshape(-1, treino.shape[2]), axis=0)
    desvio = np.nanstd(treino.reshape(-1, treino.shape[2]), axis=0)
    desvio[desvio == 0] = 1.0

    saida = {}
    for nome, idx in (("treino", idx_treino), ("val", idx_val), ("teste", idx_teste)):
        z = (dados[idx] - media) / desvio                       # (n, 48, D)
        n_amostras, L, D = z.shape
        x = z.transpose(0, 2, 1).reshape(-1, L)                  # univariado
        m_nat = np.isnan(x)
        x = np.nan_to_num(x, nan=0.0).astype("float32")
        ids = np.tile(np.arange(D), n_amostras)
        validas = (~m_nat).sum(axis=1) >= min_observados
        saida[nome] = (x[validas], m_nat[validas], ids[validas])
    return saida, media, desvio


def eliminar_observados(m_nat: np.ndarray, frac: float = 0.10, semente: int = 123):
    """Sorteia ~frac das posições OBSERVADAS como verdade de avaliação
    (equivalente ao 'randomly eliminate 10% of observed values' do artigo)."""
    rng = np.random.default_rng(semente)
    return (rng.random(m_nat.shape) < frac) & ~m_nat


# ---------------------------------------------------------------------------
# Treino e avaliação com faltantes nativos
# ---------------------------------------------------------------------------

def treinar_faltantes(modelo, dados, cfg, device, caminho_saida,
                      fn_perda=perda_mascarada):
    """Treina com faltantes nativos; early stopping no MAE de validação sobre
    os 10 % eliminados FIXOS (critério do artigo).

    dados = {"treino": (x, m_nat), "val": (x, m_nat, m_elim)}.
    A máscara artificial de treino é sorteada só sobre observados; a entrada
    do modelo marca nativos ∪ artificiais; a perda usa só os artificiais.
    """
    import json

    fixar_semente(cfg["semente"])
    saida = Path(caminho_saida)
    saida.mkdir(parents=True, exist_ok=True)

    x_tr, mn_tr = dados["treino"]
    ds_tr = TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(mn_tr))
    loader = DataLoader(ds_tr, batch_size=cfg["batch"], shuffle=True, drop_last=True)

    x_v, mn_v, me_v = dados["val"]
    x_val = torch.from_numpy(x_v)
    m_in_val = torch.from_numpy(mn_v | me_v)
    m_elim_val = torch.from_numpy(me_v)

    modelo = modelo.to(device)
    otim = torch.optim.AdamW(modelo.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    agenda = torch.optim.lr_scheduler.CosineAnnealingLR(otim, T_max=cfg["epocas"])
    rng = np.random.default_rng(cfg["semente"])
    cfg_masc = cfg["mascara"]

    historico, melhor_val, sem_melhora = [], float("inf"), 0
    for ep in range(1, cfg["epocas"] + 1):
        modelo.train()
        total, n = 0.0, 0
        for xb, mnb in loader:
            xb, mnb = xb.to(device), mnb.to(device)
            m_art = gerar_mascaras_lote(
                xb.size(0), xb.size(1), razao=cfg_masc["razao"],
                prob_bloco=cfg_masc["prob_bloco"], lm=cfg_masc["lm"], rng=rng,
            ).to(device) & ~mnb
            pred = modelo(xb, mnb | m_art)
            if not m_art.any():
                continue
            perda = fn_perda(pred, xb, m_art)
            otim.zero_grad(set_to_none=True)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            otim.step()
            total += perda.item() * xb.size(0)
            n += xb.size(0)

        modelo.eval()
        soma_abs, n_elim = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(x_val), cfg["batch"]):
                xb = x_val[i : i + cfg["batch"]].to(device)
                pred = modelo(xb, m_in_val[i : i + cfg["batch"]].to(device))
                me = m_elim_val[i : i + cfg["batch"]].to(device)
                soma_abs += (pred - xb)[me].abs().sum().item()
                n_elim += int(me.sum())
        mae_val = soma_abs / n_elim
        agenda.step()

        historico.append({"epoca": ep, "perda_treino": total / max(n, 1),
                          "mae_val": mae_val})
        marcador = ""
        if mae_val < melhor_val - 1e-5:
            melhor_val, sem_melhora, marcador = mae_val, 0, "  <- melhor"
            torch.save({"estado": modelo.state_dict(), "hparams": modelo.hparams,
                        "cfg": cfg}, saida / "melhor_modelo.pt")
        else:
            sem_melhora += 1
        print(f"época {ep:3d} | treino {total/max(n,1):.4f} | MAE val {mae_val:.4f}{marcador}")
        if sem_melhora >= cfg["paciencia"]:
            print(f"early stopping (sem melhora há {cfg['paciencia']} épocas)")
            break

    with open(saida / "historico.json", "w", encoding="utf-8") as f:
        json.dump(historico, f, indent=2)
    return historico


def metricas_artigo(pred: torch.Tensor, alvo: torch.Tensor, m_elim: torch.Tensor):
    """MAE/RMSE/MRE com somas globais, como nas equações do artigo."""
    dif = (pred - alvo)[m_elim]
    alvo_e = alvo[m_elim]
    return {
        "MAE": dif.abs().mean().item(),
        "RMSE": dif.pow(2).mean().sqrt().item(),
        "MRE": (dif.abs().sum() / alvo_e.abs().sum().clamp(min=1e-8)).item(),
    }


def avaliar_faltantes(modelo, x: np.ndarray, m_nat: np.ndarray, m_elim: np.ndarray,
                      device, batch: int = 256):
    """Avalia no teste do protocolo do artigo (entrada: nativos ∪ eliminados)."""
    xt = torch.from_numpy(x)
    m_in = torch.from_numpy(m_nat | m_elim)
    preds = []
    modelo.eval()
    with torch.no_grad():
        for i in range(0, len(xt), batch):
            preds.append(modelo(xt[i : i + batch].to(device),
                                m_in[i : i + batch].to(device)).cpu())
    return metricas_artigo(torch.cat(preds), xt, torch.from_numpy(m_elim))


# ---------------------------------------------------------------------------
# Baselines do artigo (calibração do protocolo)
# ---------------------------------------------------------------------------

def baseline_mediana(x: np.ndarray, m_nat: np.ndarray, m_elim: np.ndarray,
                     ids: np.ndarray, medianas_treino: np.ndarray):
    """'Median' do artigo: preenche com a mediana da variável no TREINO."""
    pred = torch.from_numpy(x).clone()
    med = torch.from_numpy(medianas_treino[ids].astype("float32"))[:, None]
    m = torch.from_numpy(m_elim)
    pred[m] = med.expand_as(pred)[m]
    return metricas_artigo(pred, torch.from_numpy(x), m)


def baseline_last(x: np.ndarray, m_nat: np.ndarray, m_elim: np.ndarray):
    """'Last' do artigo: última observação anterior VISÍVEL; 0 se não houver."""
    visivel = ~(m_nat | m_elim)
    idx = np.where(visivel, np.arange(x.shape[1])[None, :], -1)
    idx = np.maximum.accumulate(idx, axis=1)
    linhas = np.arange(x.shape[0])[:, None]
    pred = np.where(idx >= 0, x[linhas, np.clip(idx, 0, None)], 0.0).astype("float32")
    pred = np.where(visivel, x, pred)
    return metricas_artigo(torch.from_numpy(pred), torch.from_numpy(x),
                           torch.from_numpy(m_elim))


def medianas_do_treino(x: np.ndarray, m_nat: np.ndarray, ids: np.ndarray,
                       n_series: int) -> np.ndarray:
    """Mediana por série calculada só nos observados do treino."""
    med = np.zeros(n_series, dtype="float32")
    for j in range(n_series):
        sel = ids == j
        vals = x[sel][~m_nat[sel]]
        med[j] = np.median(vals) if len(vals) else 0.0
    return med
