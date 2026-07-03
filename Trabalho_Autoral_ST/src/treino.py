"""Laço de treino auto-supervisionado, métricas e utilitários de reprodutibilidade."""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .mascaramento import gerar_mascaras_lote
from .modelo import BERTImputador


def fixar_semente(semente: int = 42):
    """Fixa todas as fontes de aleatoriedade (reprodutibilidade)."""
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.cuda.manual_seed_all(semente)


def perda_mascarada(pred: torch.Tensor, alvo: torch.Tensor, mascara: torch.Tensor):
    """MSE calculado APENAS nas posições mascaradas (objetivo tipo BERT)."""
    erro = (pred - alvo) ** 2
    return erro[mascara].mean()


def metricas_mascaradas(pred: torch.Tensor, alvo: torch.Tensor, mascara: torch.Tensor):
    """MAE, RMSE e MRE nas posições mascaradas."""
    dif = (pred - alvo)[mascara]
    alvo_m = alvo[mascara]
    mae = dif.abs().mean().item()
    rmse = dif.pow(2).mean().sqrt().item()
    mre = (dif.abs().sum() / alvo_m.abs().sum().clamp(min=1e-8)).item()
    return {"MAE": mae, "RMSE": rmse, "MRE": mre}


def _epoca(modelo, loader, cfg_masc, device, rng, otimizador=None):
    treinar_flag = otimizador is not None
    modelo.train(treinar_flag)
    total, n = 0.0, 0
    for lote in loader:
        lote = lote.to(device, non_blocking=True)
        mascara = gerar_mascaras_lote(
            lote.size(0), lote.size(1),
            razao=cfg_masc["razao"], prob_bloco=cfg_masc["prob_bloco"],
            lm=cfg_masc["lm"], rng=rng,
        ).to(device)
        with torch.set_grad_enabled(treinar_flag):
            pred = modelo(lote, mascara)
            perda = perda_mascarada(pred, lote, mascara)
        if treinar_flag:
            otimizador.zero_grad(set_to_none=True)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            otimizador.step()
        total += perda.item() * lote.size(0)
        n += lote.size(0)
    return total / n


def treinar(
    modelo: BERTImputador,
    ds_treino,
    ds_val,
    cfg: dict,
    device: str,
    caminho_saida: str,
):
    """Treina com early stopping; salva o melhor checkpoint + histórico.

    cfg precisa de: epocas, batch, lr, paciencia, semente,
    e cfg["mascara"] = {razao, prob_bloco, lm}.
    """
    fixar_semente(cfg["semente"])
    saida = Path(caminho_saida)
    saida.mkdir(parents=True, exist_ok=True)

    loader_tr = DataLoader(ds_treino, batch_size=cfg["batch"], shuffle=True, drop_last=True)
    loader_val = DataLoader(ds_val, batch_size=cfg["batch"], shuffle=False)

    modelo = modelo.to(device)
    otim = torch.optim.AdamW(modelo.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    agenda = torch.optim.lr_scheduler.CosineAnnealingLR(otim, T_max=cfg["epocas"])

    rng_treino = np.random.default_rng(cfg["semente"])
    rng_val_base = cfg["semente"] + 10_000  # máscaras de validação fixas por época

    historico, melhor_val, sem_melhora = [], float("inf"), 0
    for ep in range(1, cfg["epocas"] + 1):
        perda_tr = _epoca(modelo, loader_tr, cfg["mascara"], device, rng_treino, otim)
        perda_val = _epoca(
            modelo, loader_val, cfg["mascara"], device,
            np.random.default_rng(rng_val_base),  # mesmas máscaras toda época
        )
        agenda.step()
        historico.append({"epoca": ep, "perda_treino": perda_tr, "perda_val": perda_val})
        marcador = ""
        if perda_val < melhor_val - 1e-5:
            melhor_val, sem_melhora, marcador = perda_val, 0, "  <- melhor"
            torch.save(
                {"estado": modelo.state_dict(), "hparams": modelo.hparams, "cfg": cfg},
                saida / "melhor_modelo.pt",
            )
        else:
            sem_melhora += 1
        print(f"época {ep:3d} | treino {perda_tr:.4f} | val {perda_val:.4f}{marcador}")
        if sem_melhora >= cfg["paciencia"]:
            print(f"early stopping (sem melhora há {cfg['paciencia']} épocas)")
            break

    with open(saida / "historico.json", "w", encoding="utf-8") as f:
        json.dump(historico, f, indent=2)
    return historico


def carregar_modelo(caminho: str, device: str = "cpu") -> BERTImputador:
    ckpt = torch.load(caminho, map_location=device, weights_only=False)
    modelo = BERTImputador(**ckpt["hparams"]).to(device)
    modelo.load_state_dict(ckpt["estado"])
    modelo.eval()
    return modelo
