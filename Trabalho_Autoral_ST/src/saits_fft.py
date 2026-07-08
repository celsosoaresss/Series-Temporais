"""SAITS-FFT: adaptação do SAITS (Du et al., 2023) com condicionamento espectral.

O SAITS original ("SAITS: Self-attention-based imputation for time series",
Expert Systems with Applications, 2023) tem três ingredientes:

  1. DMSA — self-attention com a diagonal mascarada: o passo t é reconstruído
     apenas a partir dos DEMAIS passos (remove o atalho de copiar a si mesmo);
  2. treinamento conjunto MIT + ORT — perda de imputação nos valores
     artificialmente mascarados (MIT) somada à perda de reconstrução dos
     valores observados (ORT), que só é não-trivial graças à DMSA;
  3. dois blocos DMSA encadeados + um bloco de combinação ponderada.

Adaptação proposta aqui (a contribuição deste trabalho):

  * mantemos (1) e (2);
  * substituímos (3) — o segundo bloco e a combinação ponderada, cujo ganho o
    próprio artigo admite ser marginal — por um ÚNICO bloco DMSA condicionado
    a um token espectral [FREQ]: a rFFT da parte visível da janela (posições
    mascaradas = 0, que é a média na escala z-score) é projetada em um token
    extra prefixado à sequência, a que todas as posições podem atender.

Motivação: na imputação de BLOCOS longos o contexto local desaparece e o que
resta é a estrutura periódica global (ciclos diário/semanal, dominantes em
carga elétrica). A atenção teria de aprender essa periodicidade
implicitamente; a FFT a entrega explicitamente, com custo O(L log L) e uma
única projeção linear de parâmetros extras.

Diferenças de treino em relação ao SAITS original, para manter a comparação
com o `BERTImputador` controlada: perda MSE (não MAE) e sem segunda
representação intermediária. `usar_fft=False` dá a variante de ablação
(DMSA + MIT/ORT, sem o token espectral).

Segunda contribuição (experimento do notebook 05): **refinamento iterativo com
pesos compartilhados** (`n_passadas=2`). O mesmo encoder roda duas vezes: a
1ª passada imputa; as lacunas são preenchidas com essas imputações (eq. 13 do
SAITS) e a 2ª passada refina, vendo valores plausíveis como âncoras em vez de
zeros — o canal de máscara continua marcando o que era lacuna original. É o
mecanismo do 2º bloco DMSA do SAITS com ZERO parâmetros novos: profundidade
por iteração, não por parâmetros. Em treino o forward devolve a tupla com as
reconstruções de todas as passadas (supervisão profunda via
`treino.perda_mit_ort_passadas`); em avaliação devolve só a final, mantendo a
interface de `avaliar_metodos` e dos checkpoints antigos (que não têm
`n_passadas` nos hparams e caem no padrão 1).

Nota: como no SAITS original, a conexão residual do Transformer ainda carrega
a própria embedding do passo t; a DMSA bloqueia apenas o caminho da atenção.
"""

import torch
import torch.nn as nn

from .modelo import codificacao_posicional


def caracteristicas_espectrais(x_visivel: torch.Tensor) -> torch.Tensor:
    """Espectro da janela visível: [log1p(amplitude), Re, Im] concatenados.

    x_visivel: (B, L) com posições mascaradas já zeradas (0 = média, pois a
    série é z-score — o preenchimento neutro). Retorna (B, 3*(L//2+1)).
    A rFFT é normalizada por L/2 para que os coeficientes fiquem O(1) em
    séries de variância unitária.
    """
    L = x_visivel.size(1)
    espectro = torch.fft.rfft(x_visivel, dim=1) / (L / 2.0)
    amp = espectro.abs()
    return torch.cat([torch.log1p(amp), espectro.real, espectro.imag], dim=1)


def comprimento_segmento(seq_len: int, n_segmentos: int) -> int:
    """Comprimento de segmento p/ K segmentos com 50% de sobreposição.

    L = lseg + (K-1)·lseg/2  ⇒  lseg = 2L/(K+1). Para L=168 e K=6: lseg=48
    (2 dias por segmento, passo de 24 h — o ciclo diário cai num bin exato).
    """
    return 2 * seq_len // (n_segmentos + 1)


def caracteristicas_espectrais_segmentada(
    x_visivel: torch.Tensor, n_segmentos: int = 6
) -> torch.Tensor:
    """Espectro segmentado (método de Welch): a janela é dividida em
    `n_segmentos` segmentos com 50% de sobreposição, cada um passa por uma
    rFFT e as componentes de MESMO período são promediadas entre os
    segmentos; as médias de [log1p(amplitude média), Re médio, Im médio] são
    concatenadas como os dados de frequência. Retorna (B, 3*(lseg//2+1)).

    Em relação à FFT única da janela inteira: (1) a média entre segmentos
    reduz a variância do estimador espectral; (2) um bloco mascarado corrompe
    apenas os segmentos que o contêm, e a média dilui esse efeito — na FFT
    única o vazamento espectral da lacuna contamina todos os bins. O passo
    igual a lseg/2 mantém harmônicos alinhados ao passo somando em fase.
    Se L não for múltiplo do passo, o resto ao final é descartado (unfold).
    """
    lseg = comprimento_segmento(x_visivel.size(1), n_segmentos)
    segmentos = x_visivel.unfold(1, lseg, lseg // 2)      # (B, K, lseg)
    espectro = torch.fft.rfft(segmentos, dim=2) / (lseg / 2.0)
    amp = espectro.abs().mean(dim=1)
    re = espectro.real.mean(dim=1)
    im = espectro.imag.mean(dim=1)
    return torch.cat([torch.log1p(amp), re, im], dim=1)


class SAITSFFT(nn.Module):
    """Bloco DMSA único + token espectral [FREQ] (usar_fft=False = ablação)."""

    def __init__(
        self,
        seq_len: int = 168,
        d_model: int = 64,
        n_camadas: int = 3,
        n_cabecas: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        usar_fft: bool = True,
        n_passadas: int = 1,
        modo_fft: str = "padrao",
        n_segmentos: int = 6,
    ):
        super().__init__()
        if n_passadas < 1:
            raise ValueError(f"n_passadas deve ser >= 1, recebido {n_passadas}")
        if modo_fft not in ("padrao", "segmentada"):
            raise ValueError(f"modo_fft deve ser 'padrao' ou 'segmentada', recebido {modo_fft!r}")
        self.hparams = dict(
            seq_len=seq_len, d_model=d_model, n_camadas=n_camadas,
            n_cabecas=n_cabecas, d_ff=d_ff, dropout=dropout, usar_fft=usar_fft,
            n_passadas=n_passadas, modo_fft=modo_fft, n_segmentos=n_segmentos,
        )
        self.usar_fft = usar_fft
        self.n_passadas = n_passadas
        self.modo_fft = modo_fft
        self.n_segmentos = n_segmentos
        # entrada: [valor observado (0 onde mascarado), indicador de máscara],
        # como no SAITS (concat de X̂ e M̂) — sem token [MASK] aprendível
        self.proj_entrada = nn.Linear(2, d_model)
        if usar_fft:
            if modo_fft == "segmentada":
                n_bins = comprimento_segmento(seq_len, n_segmentos) // 2 + 1
            else:
                n_bins = seq_len // 2 + 1
            self.proj_freq = nn.Sequential(
                nn.Linear(3 * n_bins, d_model), nn.GELU(), nn.Linear(d_model, d_model)
            )
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

    def _mascara_dmsa(self, seq_len: int, device) -> torch.Tensor:
        """Máscara de atenção (S, S) bool: True = atenção proibida.

        Diagonal proibida apenas nas posições temporais (DMSA, eq. 5 do
        SAITS); o token [FREQ], se existir, atende e é atendido livremente.
        """
        extra = 1 if self.usar_fft else 0
        S = seq_len + extra
        m = torch.zeros(S, S, dtype=torch.bool, device=device)
        idx = torch.arange(extra, S, device=device)
        m[idx, idx] = True
        return m

    def _extrair_espectro(self, valores: torch.Tensor) -> torch.Tensor:
        if self.modo_fft == "segmentada":
            return caracteristicas_espectrais_segmentada(valores, self.n_segmentos)
        return caracteristicas_espectrais(valores)

    def _passada(self, valores: torch.Tensor, mascara: torch.Tensor) -> torch.Tensor:
        """Uma passada do encoder sobre `valores` (B, L) já preenchidos
        (zeros na 1ª passada, imputações da passada anterior nas seguintes).
        O canal de máscara sempre marca as lacunas ORIGINAIS."""
        L = valores.size(1)
        entrada = torch.stack([valores, mascara.float()], dim=-1)  # (B, L, 2)
        h = self.proj_entrada(entrada)
        h = h + codificacao_posicional(L, h.size(-1), device=h.device)
        if self.usar_fft:
            token = self.proj_freq(self._extrair_espectro(valores))
            h = torch.cat([token.unsqueeze(1), h], dim=1)  # (B, L+1, d)
        h = self.encoder(h, mask=self._mascara_dmsa(L, valores.device))
        h = self.norma_final(h)
        if self.usar_fft:
            h = h[:, 1:]
        return self.cabeca(h).squeeze(-1)

    def forward(self, x: torch.Tensor, mascara: torch.Tensor):
        """x: (B, L) valores normalizados; mascara: (B, L) bool, True = imputar.

        Em avaliação retorna (B, L): a reconstrução da última passada. Em
        treino com n_passadas > 1 retorna a tupla com TODAS as passadas
        (supervisão profunda; o gradiente flui de uma passada à seguinte,
        como no SAITS end-to-end). A perda usa as posições mascaradas para
        MIT e as observadas para ORT.
        """
        recs = []
        valores = x.masked_fill(mascara, 0.0)
        for p in range(self.n_passadas):
            rec = self._passada(valores, mascara)
            recs.append(rec)
            if p + 1 < self.n_passadas:
                # próxima passada vê as lacunas preenchidas com as imputações
                valores = torch.where(mascara, rec, x)
        if self.training and len(recs) > 1:
            return tuple(recs)
        return recs[-1]

    def reconstruir_passadas(self, x: torch.Tensor, mascara: torch.Tensor) -> list[torch.Tensor]:
        """Reconstrução de CADA passada, em modo avaliação (p/ diagnóstico)."""
        self.eval()
        with torch.no_grad():
            recs = []
            valores = x.masked_fill(mascara, 0.0)
            for p in range(self.n_passadas):
                rec = self._passada(valores, mascara)
                recs.append(rec)
                if p + 1 < self.n_passadas:
                    valores = torch.where(mascara, rec, x)
        return recs

    def imputar(self, x: torch.Tensor, mascara: torch.Tensor) -> torch.Tensor:
        """Série completa: valores observados mantidos, mascarados imputados."""
        self.eval()
        with torch.no_grad():
            rec = self(x, mascara)
        return torch.where(mascara, rec, x)


def carregar_saits(caminho: str, device: str = "cpu") -> SAITSFFT:
    ckpt = torch.load(caminho, map_location=device, weights_only=False)
    modelo = SAITSFFT(**ckpt["hparams"]).to(device)
    modelo.load_state_dict(ckpt["estado"])
    modelo.eval()
    return modelo


class BERTFFT(nn.Module):
    """BERTImputador + token espectral [FREQ] — a contraparte BERT da família.

    Mantém o paradigma do BERT do notebook 02 (atenção completa, token [MASK]
    aprendível nas posições mascaradas, perda só nos mascarados/MIT) e
    acrescenta apenas o token espectral prefixado, com os mesmos modos de
    extração do SAITSFFT (`modo_fft` = 'padrao' | 'segmentada'). Serve para
    isolar o efeito das features de frequência sob o treino do BERT, sem
    DMSA nem ORT.
    """

    def __init__(
        self,
        seq_len: int = 168,
        d_model: int = 64,
        n_camadas: int = 3,
        n_cabecas: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        modo_fft: str = "padrao",
        n_segmentos: int = 6,
    ):
        super().__init__()
        if modo_fft not in ("padrao", "segmentada"):
            raise ValueError(f"modo_fft deve ser 'padrao' ou 'segmentada', recebido {modo_fft!r}")
        self.hparams = dict(
            seq_len=seq_len, d_model=d_model, n_camadas=n_camadas,
            n_cabecas=n_cabecas, d_ff=d_ff, dropout=dropout,
            modo_fft=modo_fft, n_segmentos=n_segmentos,
        )
        self.modo_fft = modo_fft
        self.n_segmentos = n_segmentos
        self.proj_entrada = nn.Linear(2, d_model)
        self.token_mascara = nn.Parameter(torch.zeros(d_model))
        if modo_fft == "segmentada":
            n_bins = comprimento_segmento(seq_len, n_segmentos) // 2 + 1
        else:
            n_bins = seq_len // 2 + 1
        self.proj_freq = nn.Sequential(
            nn.Linear(3 * n_bins, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
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

    def _extrair_espectro(self, valores: torch.Tensor) -> torch.Tensor:
        if self.modo_fft == "segmentada":
            return caracteristicas_espectrais_segmentada(valores, self.n_segmentos)
        return caracteristicas_espectrais(valores)

    def forward(self, x: torch.Tensor, mascara: torch.Tensor) -> torch.Tensor:
        """x: (B, L); mascara: (B, L) bool, True = imputar. Retorna (B, L)."""
        L = x.size(1)
        x_visivel = x.masked_fill(mascara, 0.0)
        entrada = torch.stack([x_visivel, mascara.float()], dim=-1)  # (B, L, 2)
        h = self.proj_entrada(entrada)
        h = torch.where(mascara.unsqueeze(-1), self.token_mascara.expand_as(h), h)
        h = h + codificacao_posicional(L, h.size(-1), device=h.device)
        token = self.proj_freq(self._extrair_espectro(x_visivel))
        h = torch.cat([token.unsqueeze(1), h], dim=1)  # (B, L+1, d)
        h = self.encoder(h)
        h = self.norma_final(h)
        return self.cabeca(h[:, 1:]).squeeze(-1)

    def imputar(self, x: torch.Tensor, mascara: torch.Tensor) -> torch.Tensor:
        """Série completa: valores observados mantidos, mascarados imputados."""
        self.eval()
        with torch.no_grad():
            rec = self(x, mascara)
        return torch.where(mascara, rec, x)


def carregar_bert_fft(caminho: str, device: str = "cpu") -> BERTFFT:
    ckpt = torch.load(caminho, map_location=device, weights_only=False)
    modelo = BERTFFT(**ckpt["hparams"]).to(device)
    modelo.load_state_dict(ckpt["estado"])
    modelo.eval()
    return modelo
