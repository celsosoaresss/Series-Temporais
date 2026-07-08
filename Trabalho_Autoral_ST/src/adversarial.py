"""Treinamento adversário para imputação, no estilo GAIN (Yoon et al., 2018).

Formalização da ideia "tempo + frequência com treino adversário":

* o GERADOR é um imputador da família espectral (`saits_fft.BERTFFT`), que já
  consome as duas entradas — a janela pré-processada no domínio do tempo e as
  features de frequência (FFT) via token [FREQ];
* o DISCRIMINADOR recebe a série COMPLETADA (observados + imputações) e prediz,
  posição a posição, quais valores foram imputados (a máscara). Como no GAIN,
  discriminar por posição dá sinal denso e evita o real/falso por sequência
  inteira dos GANs clássicos (GRUI-GAN/E²GAN), mais instável;
* a perda do gerador é a reconstrução de sempre (MSE apenas nas posições
  mascaradas, comparável aos demais notebooks) + λ_adv · termo adversarial
  (enganar o discriminador nas posições imputadas). Simplificação em relação
  ao GAIN original: sem vetor de "hint".

Estabilidade e comparabilidade: a validação/early stopping usa SOMENTE o MSE
mascarado do gerador (máscaras fixas por época, semente+10000), como em
`treino.treinar` — se o termo adversarial atrapalhar a imputação, o critério
de parada seleciona um checkpoint anterior em vez de deixar o GAN divergir.
O checkpoint salvo (`melhor_modelo.pt`) é o gerador, no mesmo formato dos
demais (carregável por `saits_fft.carregar_bert_fft`).
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .mascaramento import gerar_mascaras_lote
from .modelo import codificacao_posicional
from .saits_fft import (caracteristicas_espectrais,
                        caracteristicas_espectrais_segmentada,
                        comprimento_segmento)
from .treino import fixar_semente, perda_mascarada


class DiscriminadorMascara(nn.Module):
    """Prediz, por posição, a probabilidade de o valor ter sido imputado.

    Entrada: série completada (B, L) — sem acesso à máscara, obviamente.
    Também recebe o token espectral [FREQ] da série completada (as duas
    entradas do experimento valem para os dois lados do jogo adversarial).
    Saída: logits (B, L); sigmoide ≈ P(posição foi imputada).
    """

    def __init__(
        self,
        seq_len: int = 168,
        d_model: int = 64,
        n_camadas: int = 2,
        n_cabecas: int = 4,
        d_ff: int = 128,
        dropout: float = 0.1,
        modo_fft: str = "segmentada",
        n_segmentos: int = 6,
    ):
        super().__init__()
        self.hparams = dict(
            seq_len=seq_len, d_model=d_model, n_camadas=n_camadas,
            n_cabecas=n_cabecas, d_ff=d_ff, dropout=dropout,
            modo_fft=modo_fft, n_segmentos=n_segmentos,
        )
        self.modo_fft = modo_fft
        self.n_segmentos = n_segmentos
        self.proj_entrada = nn.Linear(1, d_model)
        if modo_fft == "segmentada":
            n_bins = comprimento_segmento(seq_len, n_segmentos) // 2 + 1
        else:
            n_bins = seq_len // 2 + 1
        self.proj_freq = nn.Sequential(
            nn.Linear(3 * n_bins, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        camada = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_cabecas, dim_feedforward=d_ff,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(camada, num_layers=n_camadas)
        self.norma_final = nn.LayerNorm(d_model)
        self.cabeca = nn.Linear(d_model, 1)

    def forward(self, x_completo: torch.Tensor) -> torch.Tensor:
        L = x_completo.size(1)
        h = self.proj_entrada(x_completo.unsqueeze(-1))
        h = h + codificacao_posicional(L, h.size(-1), device=h.device)
        if self.modo_fft == "segmentada":
            feats = caracteristicas_espectrais_segmentada(x_completo, self.n_segmentos)
        else:
            feats = caracteristicas_espectrais(x_completo)
        token = self.proj_freq(feats)
        h = torch.cat([token.unsqueeze(1), h], dim=1)
        h = self.norma_final(self.encoder(h))
        return self.cabeca(h[:, 1:]).squeeze(-1)


def treinar_adversarial(
    gerador: nn.Module,
    discriminador: DiscriminadorMascara,
    ds_treino,
    ds_val,
    cfg: dict,
    device: str,
    caminho_saida: str,
):
    """Laço adversarial com early stopping no MSE mascarado de validação.

    cfg precisa de: epocas, batch, lr, lr_d, lambda_adv, paciencia, semente,
    e cfg["mascara"] = {razao, prob_bloco, lm}. Salva `melhor_modelo.pt`
    (gerador, formato padrão), `discriminador.pt` e `historico.json` com as
    perdas por época (rec, adv, D e val).
    """
    fixar_semente(cfg["semente"])
    saida = Path(caminho_saida)
    saida.mkdir(parents=True, exist_ok=True)

    loader_tr = DataLoader(ds_treino, batch_size=cfg["batch"], shuffle=True, drop_last=True)
    loader_val = DataLoader(ds_val, batch_size=cfg["batch"], shuffle=False)

    gerador = gerador.to(device)
    discriminador = discriminador.to(device)
    otim_g = torch.optim.AdamW(gerador.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    otim_d = torch.optim.AdamW(discriminador.parameters(), lr=cfg["lr_d"], weight_decay=1e-4)
    agenda_g = torch.optim.lr_scheduler.CosineAnnealingLR(otim_g, T_max=cfg["epocas"])
    agenda_d = torch.optim.lr_scheduler.CosineAnnealingLR(otim_d, T_max=cfg["epocas"])
    bce = nn.BCEWithLogitsLoss()

    rng_treino = np.random.default_rng(cfg["semente"])
    rng_val_base = cfg["semente"] + 10_000  # mesmas máscaras de val dos outros treinos
    cfg_masc = cfg["mascara"]

    historico, melhor_val, sem_melhora = [], float("inf"), 0
    for ep in range(1, cfg["epocas"] + 1):
        gerador.train()
        discriminador.train()
        soma = {"rec": 0.0, "adv": 0.0, "D": 0.0}
        n = 0
        for lote in loader_tr:
            lote = lote.to(device, non_blocking=True)
            mascara = gerar_mascaras_lote(
                lote.size(0), lote.size(1),
                razao=cfg_masc["razao"], prob_bloco=cfg_masc["prob_bloco"],
                lm=cfg_masc["lm"], rng=rng_treino,
            ).to(device)
            alvo_masc = mascara.float()

            rec = gerador(lote, mascara)
            x_completo = torch.where(mascara, rec, lote)

            # --- discriminador: acertar quais posições foram imputadas
            logits_d = discriminador(x_completo.detach())
            perda_d = bce(logits_d, alvo_masc)
            otim_d.zero_grad(set_to_none=True)
            perda_d.backward()
            torch.nn.utils.clip_grad_norm_(discriminador.parameters(), 1.0)
            otim_d.step()

            # --- gerador: reconstruir bem E fazer as imputações parecerem observadas
            logits_g = discriminador(x_completo)
            perda_adv = bce(logits_g[mascara], torch.zeros_like(logits_g[mascara]))
            perda_rec = perda_mascarada(rec, lote, mascara)
            perda_g = perda_rec + cfg["lambda_adv"] * perda_adv
            otim_g.zero_grad(set_to_none=True)
            perda_g.backward()
            torch.nn.utils.clip_grad_norm_(gerador.parameters(), 1.0)
            otim_g.step()

            b = lote.size(0)
            soma["rec"] += perda_rec.item() * b
            soma["adv"] += perda_adv.item() * b
            soma["D"] += perda_d.item() * b
            n += b

        # --- validação: SÓ o gerador, MSE mascarado, máscaras fixas por época
        gerador.eval()
        rng_val = np.random.default_rng(rng_val_base)
        soma_val, n_val = 0.0, 0
        with torch.no_grad():
            for lote in loader_val:
                lote = lote.to(device, non_blocking=True)
                mascara = gerar_mascaras_lote(
                    lote.size(0), lote.size(1),
                    razao=cfg_masc["razao"], prob_bloco=cfg_masc["prob_bloco"],
                    lm=cfg_masc["lm"], rng=rng_val,
                ).to(device)
                soma_val += perda_mascarada(gerador(lote, mascara), lote, mascara).item() * lote.size(0)
                n_val += lote.size(0)
        perda_val = soma_val / n_val
        agenda_g.step()
        agenda_d.step()

        historico.append({
            "epoca": ep,
            "perda_treino": soma["rec"] / n + cfg["lambda_adv"] * soma["adv"] / n,
            "perda_rec": soma["rec"] / n,
            "perda_adv": soma["adv"] / n,
            "perda_D": soma["D"] / n,
            "perda_val": perda_val,
        })
        marcador = ""
        if perda_val < melhor_val - 1e-5:
            melhor_val, sem_melhora, marcador = perda_val, 0, "  <- melhor"
            torch.save(
                {"estado": gerador.state_dict(), "hparams": gerador.hparams, "cfg": cfg},
                saida / "melhor_modelo.pt",
            )
            torch.save(
                {"estado": discriminador.state_dict(), "hparams": discriminador.hparams},
                saida / "discriminador.pt",
            )
        else:
            sem_melhora += 1
        print(f"época {ep:3d} | rec {soma['rec']/n:.4f} | adv {soma['adv']/n:.4f} "
              f"| D {soma['D']/n:.4f} | val {perda_val:.4f}{marcador}")
        if sem_melhora >= cfg["paciencia"]:
            print(f"early stopping (sem melhora há {cfg['paciencia']} épocas)")
            break

    with open(saida / "historico.json", "w", encoding="utf-8") as f:
        json.dump(historico, f, indent=2)
    return historico
