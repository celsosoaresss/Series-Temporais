"""BERT para imputação de séries temporais (masked reconstruction).

Arquitetura no estilo TST (Zerveas et al., 2021, "A Transformer-based
Framework for Multivariate Time Series Representation Learning"):

  valor + indicador de máscara ──Linear──► d_model
                                   + token [MASK] aprendível (nas posições mascaradas)
                                   + codificação posicional senoidal
                     ──► Encoder Transformer (multi-head self-attention bidirecional)
                     ──► Linear ──► valor reconstruído por passo de tempo

Como no BERT, a atenção é bidirecional (não causal): para imputar o passo t o
modelo olha passado E futuro — exatamente o que se quer em imputação. A
codificação posicional senoidal é calculada dinamicamente, então o modelo
aceita janelas de qualquer comprimento (importante para transferir a outros
datasets).
"""

import math

import torch
import torch.nn as nn


def codificacao_posicional(seq_len: int, d_model: int, device=None) -> torch.Tensor:
    """Codificação senoidal clássica (Vaswani et al., 2017), shape (1, L, d)."""
    pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


class BERTImputador(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        n_camadas: int = 3,
        n_cabecas: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hparams = dict(
            d_model=d_model, n_camadas=n_camadas, n_cabecas=n_cabecas,
            d_ff=d_ff, dropout=dropout,
        )
        # entrada: [valor observado (0 onde mascarado), indicador de máscara]
        self.proj_entrada = nn.Linear(2, d_model)
        self.token_mascara = nn.Parameter(torch.zeros(d_model))
        camada = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_cabecas,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(camada, num_layers=n_camadas)
        self.norma_final = nn.LayerNorm(d_model)
        self.cabeca = nn.Linear(d_model, 1)
        nn.init.normal_(self.token_mascara, std=0.02)

    def forward(self, x: torch.Tensor, mascara: torch.Tensor) -> torch.Tensor:
        """x: (B, L) valores normalizados; mascara: (B, L) bool, True = imputar.

        Retorna (B, L): série reconstruída (só as posições mascaradas são
        usadas na perda, mas o modelo reconstrói tudo).
        """
        x_visivel = x.masked_fill(mascara, 0.0)
        entrada = torch.stack([x_visivel, mascara.float()], dim=-1)  # (B, L, 2)
        h = self.proj_entrada(entrada)
        # substitui a representação das posições mascaradas pelo token [MASK]
        h = torch.where(mascara.unsqueeze(-1), self.token_mascara.expand_as(h), h)
        h = h + codificacao_posicional(x.size(1), h.size(-1), device=h.device)
        h = self.encoder(h)
        h = self.norma_final(h)
        return self.cabeca(h).squeeze(-1)

    def imputar(self, x: torch.Tensor, mascara: torch.Tensor) -> torch.Tensor:
        """Série completa: valores observados mantidos, mascarados imputados."""
        self.eval()
        with torch.no_grad():
            rec = self(x, mascara)
        return torch.where(mascara, rec, x)
