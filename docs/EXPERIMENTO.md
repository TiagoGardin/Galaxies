# Relatório de Experimento

## Hipótese

<!-- O que você espera encontrar, antes de rodar? (ex.: sobre o baseline, sobre o ganho do conteúdo) -->

## Setup

<!--
- Alvo e baseline(s).
- Splits: datas de corte do temporal (treino/validação/teste) e como foi o split por canal.
- Métricas e por que cada uma.
- Onde os runs estão registrados (MLflow).
-->

## Resultados

### Baseline vs. modelo

| Modelo | MAE-log | MAPE | Spearman | Spearman intra-canal | Acurácia par-a-par | Cobertura do intervalo | F1 faixas |
|---|---|---|---|---|---|---|---|
| Baseline | | | | | | | |
| | | | | | | | |

### Ablação por bloco

| Features | Spearman | MAE-log | Ganho vs. anterior |
|---|---|---|---|
| A | | | |
| A+B | | | |
| A+B+C | | | |
| A+B+C+D | | | |

### Split por canal (canais não vistos)

<!-- Mesmas métricas. Como o modelo generaliza para um cliente novo? -->

## Análise de erros

<!-- Onde o modelo mais erra? Que tipo de vídeo/canal? Exemplos concretos. -->

## Conclusão

<!-- Responda à pergunta norteadora com os números. A hipótese se confirmou? -->
