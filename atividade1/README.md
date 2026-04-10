# Análise de Séries Temporais: Exportação e Capacidade Estática de Café

Este repositório contém a atividade de observação e análise de séries temporais da capacidade estática e exportação de café no Brasil. O código presente no notebook `atividade.ipynb` realiza desde a leitura dos dados, análise exploratória e decomposição da série, até os rigorosos testes de estacionaridade utilizando o método de Dickey-Fuller Aumentado (ADF).

## Estrutura do Código

O arquivo principal `atividade.ipynb` executa os seguintes passos:

1.  **Importação e Limpeza de Dados:** Uso do `pandas` para carga e estruturação por Ano e Região/UF.
2.  **Análise Exploratória:** Cálculo de médias, desvios padrão e visualização temporal via `matplotlib`.
3.  **Decomposição:** Separação de Tendência, Sazonalidade e Resíduos via `statsmodels`.
4.  **Teste de Estacionaridade:** Aplicação do teste ADF e Diferenciação iterativa (ordens $d=1$ ou $d=2$) para estabilização da série.

---

## Fundamentação Teórica e Fórmulas

### 1. Decomposição de Séries Temporais

Uma série temporal $Y_t$ é decomposta em três componentes primários sob um modelo aditivo:

* **Tendência ($T_t$):** Direção de longo prazo.
* **Sazonalidade ($S_t$):** Flutuações periódicas (ex: safras).
* **Resíduos ($R_t$):** Variação aleatória (ruído).

**Modelo Aditivo:**
$$Y_t = T_t + S_t + R_t$$

**Extração de Resíduos:**
$$R_t = Y_t - T_t - S_t$$

> **Nota:** Se a variação aumenta proporcionalmente ao nível da série, utiliza-se o modelo **Multiplicativo**: $Y_t = T_t \times S_t \times R_t$.

### 2. Estacionaridade

Uma série é **estacionária** quando suas propriedades estatísticas (média e variância) são constantes ao longo do tempo, oscilando em torno de um equilíbrio sem tendências estruturais.

### 3. Teste de Dickey-Fuller Aumentado (ADF)

O Teste ADF avalia a presença de uma raiz unitária. O modelo de regressão utilizado é:

$$\Delta Y_t = \alpha + \beta t + \gamma Y_{t-1} + \sum_{j=1}^{p} \delta_j \Delta Y_{t-j} + \epsilon_t$$

**Onde:**
* $\Delta Y_t$: Primeira diferença ($Y_t - Y_{t-1}$)
* $\alpha$: Constante
* $\beta t$: Coeficiente de tendência temporal
* $\epsilon_t$: Erro aleatório (ruído branco)

**Hipóteses:**
* **$H_0$ (Nula):** $\gamma = 0$. A série possui raiz unitária (**Não-estacionária**).
* **$H_1$ (Alternativa):** $\gamma < 0$. A série **é estacionária**.

### 4. Diferenciação (Ordem $d$)

Para converter uma série não-estacionária em estacionária, aplica-se a diferenciação para estabilizar a média:

* **Ordem 1 ($d=1$):**
    $$Y'_t = Y_t - Y_{t-1}$$
* **Ordem 2 ($d=2$):**
    $$Y''_t = Y_t - 2Y_{t-1} + Y_{t-2}$$

---

## Como Executar

1.  Instale as dependências: `pandas`, `matplotlib` e `statsmodels`.
2.  Verifique os caminhos dos arquivos `.xls` no carregamento dos DataFrames.
3.  Execute as células do notebook sequencialmente para gerar os gráficos e resultados do teste ADF.
