# Predição de Resultados no Futebol com Machine Learning

Projeto de Machine Learning que busca prever o resultado de partidas de futebol (vitória do time da casa, empate ou vitória do visitante) a partir do histórico de desempenho das equipes.

## Sumário

1. [Identificação do Problema e Contextualização](#1-identificação-do-problema-e-contextualização)
2. [Aquisição e Análise de Dados](#2-aquisição-e-análise-de-dados)
3. [Pré-processamento de Dados](#3-pré-processamento-de-dados)
4. [Modelagem](#4-modelagem)
5. [Avaliação do Modelo](#5-avaliação-do-modelo)
6. [Conclusões e Recomendações](#6-conclusões-e-recomendações)

---

## 1. Identificação do Problema e Contextualização

Prever o resultado de uma partida de futebol é um problema de interesse constante, especialmente por conta do mercado de apostas esportivas. Casas de apostas licenciadas (como a Betano, no Brasil) oferecem retornos financeiros associados às probabilidades implícitas do resultado de um jogo — por exemplo, uma aposta de R$100 em determinado resultado pode retornar R$315, um lucro de R$215 caso o palpite esteja correto.

O volume de dinheiro movimentado nesse mercado é expressivo: segundo levantamento da [Klavi](https://klavi.ai/placar-das-bets), com uma amostra de 1 milhão de brasileiros, cerca de **R$800 milhões** foram apostados ao longo da Copa do Mundo.

Uma forma de tentar prever esses resultados é olhar para o histórico das equipes: número de finalizações, gols marcados, gols sofridos, entre outras estatísticas de jogos anteriores.

### Escopo do projeto

Em vez de trabalhar com confrontos de Copa do Mundo (baixa frequência de jogos, dificultando a identificação de padrões), o projeto foca na **Championship**, a segunda divisão do futebol inglês. A escolha se justifica por:

- Ocorrer todos os anos, de forma recorrente;
- Gerar um volume de dados consideravelmente maior;
- Facilitar a identificação de padrões estatísticos consistentes.

---

## 2. Aquisição e Análise de Dados

### Fonte dos dados

Os dados foram obtidos diretamente da plataforma [football.uk](http://football-data.co.uk/), que disponibiliza informações desde a temporada 2000/2001. O futebol inglês é organizado por temporadas, que começam em agosto de um ano e terminam em maio do ano seguinte. Cada temporada corresponde a um arquivo CSV com informações de partidas e estatísticas.

- **504 jogos** por temporada;
- **13.104 jogos** no total considerando todas as temporadas disponíveis.

### Estrutura dos dados

Para cada partida, são disponibilizadas informações como:

- Time da casa e time visitante;
- Gols marcados por cada equipe;
- Finalizações;
- Cartões;
- Faltas.

Nem todas as colunas puderam ser aproveitadas, já que a disponibilidade de dados varia bastante entre elas — colunas com baixa proporção de preenchimento ao longo dos jogos foram descartadas (ver seção de pré-processamento).

### Variável alvo

A classe alvo a ser predita é o resultado da partida:

- **H** — vitória do time da casa;
- **D** — empate;
- **A** — vitória do time visitante.

A distribuição das classes é relativamente equilibrada, com o **erro da classe majoritária em 43,7%** (ou seja, sempre prever "vitória do mandante" acertaria 56,3% dos casos).

---

## 3. Pré-processamento de Dados

### 3.1 Remoção de colunas com dados insuficientes

A primeira etapa consistiu em descartar colunas cuja proporção de preenchimento ao longo dos jogos era baixa demais para serem usadas de forma confiável.

### 3.2 Tratamento da data

A data de cada jogo foi convertida em duas novas variáveis: **dia do ano** e **semana**. Essa transformação evita a comparação direta de jogos entre anos diferentes sem contexto, e captura a natureza cíclica das temporadas, que pode carregar informação relevante (ex.: início x fim de temporada).

### 3.3 Unificação das odds

Existem diversas casas de apostas, cada uma com seus próprios valores de odds, nem sempre disponíveis para todos os jogos. Para lidar com isso, as odds foram unificadas em um único valor por partida: a **média das odds** entre as casas disponíveis.

### 3.4 Construção de histórico por equipe

Como não é possível utilizar estatísticas do próprio jogo antes de ele acontecer (vazamento de dados), foi construído um **histórico por equipe** anterior a cada confronto, considerando:

- **Médias móveis simples e exponencial** dos atributos, com janelas de tamanho **3, 5, 7 e 10** jogos;
- **Indicadores de performance** relacionados ao número de vitórias, derrotas e empates dentro da janela considerada.

Esse processo gerou **641 colunas**, restando **8.032 linhas válidas** de dados (após remoção de registros sem histórico suficiente).

### 3.5 Redução de dimensionalidade

Devido à alta correlação entre muitas das colunas geradas (natural, já que derivam do mesmo atributo em janelas diferentes), foram aplicadas duas estratégias complementares: **extração** e **seleção de atributos**.

#### Extração de atributos (PCA)

Foi aplicado PCA (Principal Component Analysis), calculando o número de componentes necessários para explicar:

| Variância explicada | Nº de atributos |
|---|---|
| 90% | 67 |
| 95% | 109 |
| 99% | 228 |

Isso gerou **3 novos datasets**, um para cada nível de variância.

#### Seleção de atributos — Wrapper

Estratégia baseada em um modelo capaz de ranquear atributos por importância. A cada iteração, os **5% menos relevantes** são descartados, repetindo o processo até atingir um limiar mínimo de atributos. Executada com dois modelos baseados em árvore de decisão:

- **XGBoost**
- **Random Forest**

Gerando **2 datasets** adicionais.

#### Seleção de atributos — Filter

Para viabilizar o uso de modelos não baseados em árvore de decisão, também foi aplicada uma seleção via **filter**, utilizando **mutual information** como métrica de ranqueamento (mede o quanto uma variável independente informa sobre a variável alvo). Foram mantidos os top:

- **25**
- **50**
- **100** atributos

Gerando **3 datasets** adicionais.

### 3.6 Resultado do pré-processamento

Ao final dessa etapa, foram obtidos **9 datasets distintos**, combinando o dataset original pré-processado com as diferentes técnicas de seleção e extração de atributos.

---

## 4. Modelagem

### Algoritmos utilizados

Foram escolhidos algoritmos de duas naturezas distintas:

- **Baseados em árvore de decisão**: XGBoost, Random Forest;
- **Geométricos**: KNN, SVC.

Cada algoritmo foi executado em cada um dos datasets gerados, com exceção dos datasets construídos via wrapper, que foram usados apenas com o modelo correspondente (o dataset do wrapper XGBoost apenas com XGBoost, e o do wrapper Random Forest apenas com Random Forest).

### Divisão dos dados

- **80%** treino;
- **20%** teste.

### Ajuste de hiperparâmetros

Para cada combinação (modelo + dataset):

1. Definido um conjunto de valores possíveis para cada hiperparâmetro do modelo;
2. Sorteada aleatoriamente uma combinação de valores, repetindo esse processo **20 vezes**;
3. Selecionada a combinação de hiperparâmetros que resultou na melhor acurácia;
4. O modelo final é treinado com **100% dos dados de treino (80%)** utilizando os melhores hiperparâmetros encontrados;
5. A acurácia final é avaliada sobre os **20% de teste** separados inicialmente.

---

## 5. Avaliação do Modelo

### Visão geral

O modelo escolhido para análise detalhada foi o **Random Forest**, por apresentar a melhor acurácia entre os candidatos. Como a diferença de acurácia entre os melhores modelos não é suficiente, por si só, para determinar superioridade real, e como o Random Forest é comparativamente mais simples, ele foi selecionado como modelo final — privilegiando simplicidade entre modelos com desempenho equivalente.

### Matriz de confusão e métricas

O modelo não consegue prever empates corretamente, concentrando seus acertos na classe majoritária (vitória do time da casa) — porém sem depender exclusivamente dela.

**Precisão** (leitura por linha da matriz de confusão): a maior precisão está na classe majoritária, enquanto empates apresentam o pior desempenho. Quando o modelo indica vitória do time da casa, ele acerta cerca de **50%** das vezes.

**Recall** (leitura por coluna da matriz de confusão): segue o mesmo padrão — melhor desempenho na classe majoritária e pior nos empates, indicando que o modelo praticamente nunca prevê empate como resultado.

---

## 6. Conclusões e Recomendações

A predição de resultados de futebol se mostra um problema difícil, influenciado por múltiplas variáveis que não podem ser plenamente capturadas pelos dados disponíveis, além de fatores randômicos inerentes ao esporte.

Como possíveis melhorias futuras:

- Utilizar uma base de dados mais completa, incorporando estatísticas mais modernas como **xG (Expected Goals)**;
- Explorar de forma mais sistemática o espaço de decisões de pré-processamento (janelas de médias móveis, estratégias de seleção/extração de atributos), já que ao longo do projeto não havia um caminho único e claro a seguir — exigindo testar múltiplas combinações e abordagens (ex.: janelas de 3, 5, 7 e 10 jogos).

---

## Estrutura resumida do pipeline

```
Dados brutos (football.uk)
        │
        ▼
Limpeza e remoção de colunas incompletas
        │
        ▼
Engenharia de atributos (data → dia do ano / semana; odds → média)
        │
        ▼
Construção de histórico por equipe (médias móveis + performance)
        │
        ▼
Redução de dimensionalidade (PCA / Wrapper / Filter) → 9 datasets
        │
        ▼
Modelagem (KNN, SVC, Random Forest, XGBoost) + tuning de hiperparâmetros
        │
        ▼
Avaliação (acurácia, matriz de confusão, precisão, recall)
        │
        ▼
Modelo final: Random Forest
```