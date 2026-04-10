# Análise de Séries Temporais: Exportação e Capacidade Estática de Café

Este repositório contém a atividade de observação e análise de séries temporais da capacidade estática e exportação de café no Brasil. O código presente no notebook `atividade.ipynb` realiza desde a leitura dos dados, análise exploratória e decomposição da série, até os rigorosos testes de estacionaridade utilizando o método de Dickey-Fuller Aumentado (ADF).

## Estrutura do Código

O arquivo principal `atividade.ipynb` executa os seguintes passos:

1. **Importação e Limpeza de Dados:** 
   O código carrega as planilhas usando o `pandas` (`pd.read_excel`), permitindo a visualização da Quantidade em função do Ano e da Região/UF.
2. **Análise Exploratória e Visualização:**
   - Para cada região (Nordeste, Centro-Oeste, etc.), calcula estatísticas chave como média e desvio padrão.
   - Apresenta através do `matplotlib` como a quantidade evolui anualmente no tempo para detectar à primeira vista o comportamento da série.
3. **Decomposição da Série Temporal:**
   Utiliza `seasonal_decompose` (da biblioteca `statsmodels`) para separar e analisar independentemente os componentes de Tendência, Sazonalidade e Resíduos.
4. **Teste de Estacionaridade (Dickey-Fuller):**
   Utiliza a função `adfuller` para descobrir de maneira estatística se uma região possui uma série temporal estacionária. Caso não tenha, ele aplica Diferenciação (ordens $d=1$ ou $d=2$) iterativamente para estabilizar a média e variância.

---

## Fundamentação Teórica e Fórmulas

Para um melhor entendimento das ferramentas aplicadas neste código, apresentamos abaixo o escopo teórico principal de uma base de Séries Temporais.

### 1. Decomposição de Séries Temporais

Uma série temporal $Y_t$ é comumente assumida como a soma (ou produto) de três componentes primários. O código adota um modelo clássico, onde para um instante $t$:
- **Tendência ($T_t$):** A direção a longo prazo em que a série está se movendo (pode ser contínua de crescimento ou queda).
- **Sazonalidade ($S_t$):** As flutuações periódicas curtas. Resultam de fatores que acontecem numa frequência específica (como safras anuais de café).
- **Resíduos / Ruído ($R_t$):** É a variação aleatória restante na série. O que o modelo de sazonalidade e tendência não consegue explicar.

**Formula do Modelo Aditivo:**
$$ Y_t = T_t + S_t + R_t $$

Isolando os comportamentos, os *Resíduos* ($R_t$) podem ser extraídos e analisados unicamente:
$$ R_t = Y_t - T_t - S_t $$
*(Obs: Se a variação sazonal aumentar proporcionalmente com o nível da série, um modelo **Multiplicativo** ($Y_t = T_t \times S_t \times R_t$) seria preferido).*

### 2. Estacionaridade

Uma premissa de diversos modelos preditivos (como ARIMA) é que os dados devem ser **Estacionários**. Uma série é dita estacionária quando as suas propriedades estatísticas (como média e variância) não mudam com o tempo. Ou seja, ela oscila em volta de uma média constante sem tendência evidente.

### 3. Teste de Dickey-Fuller Aumentado (ADF)

No código, foi importado e utilizado via `statsmodels.tsa.stattools.adfuller`. O Teste ADF avalia a presença de uma "raiz unitária" em uma amostra de série temporal. Uma raiz unitária sugere que a série é influenciada pela tendência ao longo do tempo (logo, Não-Estacionária).

O ADF encaixa o próximo modelo de regressão para investigar a diferenciação da série:
$$ \Delta Y_t = \alpha + \beta t + \gamma Y_{t-1} + \sum_{j=1}^{p} \delta_j \Delta Y_{t-j} + \epsilon_t $$

Onde:
* $\Delta Y_t$: Primeira diferença ($Y_t - Y_{t-1}$)
* $\alpha$: Constante
* $\beta t$: O coeficiente de uma tendência temporal.
* $\epsilon_t$: O erro aleatório branco.

**Hipóteses do Teste:**
* **$H_0$ (Hipótese Nula):** $\gamma = 0$. A série tem raiz unitária, o que a torna **não-estacionária**.
* **$H_1$ (Hipótese Alternativa):** $\gamma < 0$. A série **é estacionária**.

Assim, se o valor final da probabilidade (p-valor) que o código informa for **maior** do que um nível de significância (geralmente $0.05$), você aceita que a série **NÃO** é estacionária (ex: "Série NÃO estacionária (p=0.9975)"). 

### 4. Diferenciação (Ordem de integração, `d`)

No momento que o teste ADF aponta uma série *não-estacionária*, o raciocínio é eliminar a tendência fazendo com que a série passe a ter variância constante. Como visto nas saídas do Jupyter: `Estacionarizada com Ordem 2 (p=0.0000)`.
Esse processo se chama **Diferenciação**.

* **Diferenciação de Ordem 1 ($d=1$):** Substitui cada valor de $Y_t$ pela subtração em relação ao momento anterior:
  $$ Y'_t = Y_t - Y_{t-1} $$
* **Diferenciação de Ordem 2 ($d=2$):** Aplicada quando a ordem 1 falha. Faz-se a diferença das diferenças contínuas:
  $$ Y''_t = Y'_t - Y'_{t-1} = (Y_t - Y_{t-1}) - (Y_{t-1} - Y_{t-2}) = Y_t - 2Y_{t-1} + Y_{t-2} $$

Tornar a sua série estacionária por estas transformações abre caminho para que a Região identificada possa usar um modelo ARIMA(p,d,q) com hiperparâretro de integração `d` recém-encontrado.

---
## Como Executar

1. Certifique-se de que os pacotes essenciais `pandas`, `matplotlib` e `statsmodels` estão instalados em seu ambiente.
2. Atualize, se necessário, os diretórios de origem dos arquivos `.xls` no momento de instanciar o DataFrame no início do arquivo.
3. Rode todas as células sequencialmente para extrair os gráficos de Tendência e o *output* descritivo do teste ADF na sua console.
