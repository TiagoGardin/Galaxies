# Relatório de Experimento

## Hipótese

Registrada **antes** de qualquer treino (11/10/2026), para não ser ajustada aos resultados.

**Pergunta norteadora (TAPI):** é possível prever, antes da publicação e a partir do conteúdo do criativo e do histórico do canal, como um público real vai reagir a ele, e com que margem de erro?

| # | Hipótese | Como verificar |
|---|---|---|
| H1 | O modelo completo bate o baseline (mediana do canal) no teste temporal | MAE-log menor que o do baseline e acurácia par-a-par intra-canal acima de 50% (o baseline empata todos os pares do mesmo canal) |
| H2 | A ordenação entre criativos do mesmo canal acerta em torno de 60% dos pares, o patamar que a Galaxies considera prova de conceito | Acurácia par-a-par intra-canal no teste |
| H3 | O conteúdo ajuda, mas pouco: os blocos C e D somados adicionam entre +0,02 e +0,04 de Spearman sobre A+B, na linha da literatura (SMP Challenge, SMTPD) | Tabela de ablação |
| H4 | O desempenho cai em canais nunca vistos e em canais de marca | Split por canal e métricas por tipo de canal |
| H5 | Os campos binários do VLM ajudam mais que as notas subjetivas 0–10 | Ablação dentro do bloco D |

Como o alvo é **relativo ao canal**, o Spearman global é calculado sobre o alvo relativo e deve ficar **abaixo** dos 0,55–0,85 da literatura, que usa curtidas absolutas (onde o tamanho do canal explica a maior parte da variação). Para comparação com a literatura, também é reportado o Spearman em `log1p(curtidas)` absoluto.

## Setup

### Alvo e baselines
- **Alvo:** `log1p(curtidas) − log1p(mediana dos últimos (até) 20 Shorts do canal publicados ≥ 30 dias antes)`. Definição completa no [MODEL_CARD.md](MODEL_CARD.md).
- **Baseline 1:** mediana do canal (prever alvo = 0).
- **Baseline 2:** regressão linear de `log1p(curtidas)` sobre `log_mediana_likes_ult20`.

### Splits
Validação cruzada aleatória **não é usada** para as métricas finais.

Implementação: `python -m galaxies_tool.splits` → `data/processed/splits_v1.json` (datas de corte, semente e listas de canais). Só entram vídeos com `elegivel_treino` (alvo com pelo menos 5 Shorts de histórico).

1. **Canais de demonstração:** os 4 canais com `demo = sim` em `data/canais.csv` (Eudora, Alice Salazar, Luciane Cachinski, Liza Block) ficam fora de treino, validação e teste. Servem só para o "de-para" da apresentação.
2. **Canais novos:** 20% dos demais canais, sorteados dentro de cada estrato setor × tipo (semente 42), ficam fora do treino e da validação. Seus vídeos do período de teste formam o conjunto **teste em canais novos**; os vídeos anteriores ao corte não são usados.
3. **Temporal (principal):** os vídeos elegíveis dos canais restantes são cortados por `published_at` nos quantis de 70% e 85% (arredondados para o início do dia, UTC): **treino (~70%) / validação (~15%) / teste temporal (~15%)**. As datas de corte ficam registradas aqui e não mudam depois de olhar o teste. Se o treino cobrir menos de 12 meses, a feature `mes` sai do modelo.
   - Corte treino → validação: **23/04/2026**
   - Corte validação → teste: **02/07/2026**
   - O treino cobre 12,7 meses (01/04/2025 a 22/04/2026), então a feature `mes` **entra** no modelo.

Tamanho dos conjuntos (`base_v1`, coleta de 12/10/2026):

| Conjunto | Vídeos | Criadores / marcas | Período |
|---|---|---|---|
| Treino | 22.430 | 18.354 / 4.076 | 01/04/2025 – 22/04/2026 |
| Validação | 4.774 | 4.008 / 766 | 23/04/2026 – 01/07/2026 |
| Teste temporal | 4.861 | 3.953 / 908 | 02/07/2026 – 12/09/2026 |
| Teste em canais novos (51 canais) | 1.146 | 950 / 196 | 02/07/2026 – 12/09/2026 |
| Não usados (vídeos dos canais novos antes do corte) | 6.966 | 6.130 / 836 | — |
| Canais de demonstração (4) | 688 | — | — |

Canais novos por estrato: beleza/criador 30, beleza/marca 3, moda/criador 11, moda/marca 7. A lista está em `splits_v1.json`.

Hiperparâmetros e calibração do intervalo usam só a validação. O teste é avaliado **uma única vez**, no modelo final.

### Amostra dos blocos C (vídeo) e D (VLM)
A parte de vídeo do bloco C (download, frames e áudio) e o VLM do bloco D não cobrem a base inteira: o download precisa ir devagar para não ser bloqueado pelo YouTube, e a cota gratuita do Gemini é de cerca de 500 vídeos por dia. Os dois usam a mesma fila (`data/features/amostra_vlm.txt`, semente 42, `galaxies_tool/amostra_vlm.py`): 2.500 vídeos sorteados do treino, 1.000 da validação e depois todo o teste temporal, em ordem aleatória. Se o prazo terminar antes da fila, a parte coberta do teste continua sendo uma amostra aleatória dele. O uso de uma amostra foi aprovado pelo mentor em 14/10/2026.

Consequências:
- A ablação com os blocos C (vídeo) e D é feita só nos vídeos da amostra, comparando todos os modelos nos mesmos vídeos.
- O teste em canais novos fica fora da fila, então os blocos C (vídeo) e D **não são avaliados em canais novos**. Nesse conjunto, só os blocos A, B e C (thumbnail) são avaliados.

### Métricas
| Métrica | Definição | Por que |
|---|---|---|
| MAE-log | Erro absoluto médio no alvo (escala log) | Erro típico em "quantas vezes" o vídeo fica acima/abaixo do canal |
| MAPE | Erro percentual médio sobre `1 + curtidas`, após converter a previsão de volta. O `+1` é necessário porque 372 vídeos-alvo têm 0 curtidas (divisão por zero) e é coerente com o `log1p` do alvo | Erro relativo legível para o cliente |
| Spearman global | Correlação de postos entre previsão e real em todo o teste | Ordenação geral |
| Spearman intra-canal | Média do Spearman calculado dentro de cada canal (canais com ≥ 5 vídeos no teste). Indefinido para previsões constantes, como a do baseline 1 | Ordenação entre criativos da mesma marca, que é a decisão real |
| Acurácia par-a-par intra-canal | Fração de pares de vídeos do mesmo canal em que o modelo ordena igual ao real. Pares empatados no real são ignorados; empate na previsão conta meio acerto | "Qual de dois criativos rende mais", a pergunta da Galaxies |
| Acurácia par-a-par na mesma época (7 dias) | Igual à anterior, mas só com pares do mesmo canal publicados a até 7 dias de distância | É a decisão real da Galaxies: escolher entre criativos que sairiam na mesma época. Ver "Ajuste de protocolo" abaixo |
| Cobertura do intervalo | Fração de vídeos com curtidas reais dentro do intervalo de 80% | Calibração da incerteza (alvo: ~80%) |
| F1 macro das faixas | F1 nas faixas abaixo / na média / acima / viral | Qualidade da comunicação em categorias |

Todas as métricas são reportadas também por setor (beleza × moda) e por tipo de canal (marca × criador).

### Ajuste de protocolo (13/10/2026, antes de qualquer contato com o teste)
Na validação, uma previsão que **não olha o conteúdo** ("se a mediana recente do canal está alta, o próximo vídeo fica abaixo dela", ou seja, regressão à média) já acerta **62,0%** dos pares na acurácia par-a-par com todos os pares do canal. A razão: pares de vídeos do mesmo canal publicados em épocas diferentes são ordenados pelo momento do canal, não pelo criativo. Restrita a pares publicados a até 7 dias de distância, a mesma previsão cai para **54,2%**.

Por isso foi acrescentada a **acurácia par-a-par na mesma época (7 dias)**, que mede o que a Galaxies quer saber (qual de dois criativos publicar agora). A métrica original continua reportada, porque está nas hipóteses registradas; H1 e H2 são avaliadas nas duas. O teste não foi aberto antes desta mudança.

### Validação das flags de formato
Cerca de 100 vídeos sorteados do treino são rotulados à mão (assistindo) com o formato real. Mede-se a precisão e o recall das flags de texto (bloco B) e do campo `formato` do VLM (bloco D) contra esses rótulos.

### Registro
Cada execução é registrada no MLflow (`mlruns/`, local): parâmetros, versão da base, datas de corte, métricas e importâncias SHAP. O run final é referenciado no README.

## Resultados

### Desenvolvimento (validação, 14/10/2026)

Números da **validação** (4.774 vídeos), usados só para decidir o que entra no modelo; os resultados oficiais são os do teste, abaixo. Treino: 22.430 vídeos; early stopping nos últimos 10% do treino. Bloco C ainda sem a parte de vídeo (amostra). As duas últimas linhas acrescentam o OCR da thumbnail.

Reprodutibilidade: as linhas de baselines, A, A+B (manual), A+B e as duas com OCR saem de `uv run python -m galaxies_tool.modelo`. As linhas "cor e rosto" e "+ SigLIP da thumb" (sem OCR), a validação cruzada por canal e o IC por bootstrap abaixo vieram de execuções exploratórias anteriores ao OCR, com os mesmos dados, splits e hiperparâmetros, e não têm script próprio no repositório.

| Modelo | MAE-log | MAPE | Spearman | Spearman intra-canal | Par-a-par | Par-a-par mesma época | F1 faixas |
|---|---|---|---|---|---|---|---|
| Baseline 1 (mediana do canal) | 0,875 | 1,722 | indefinido | indefinido | 50,0% | 50,0% | 0,130 |
| Baseline 2 (linear) | 0,879 | 1,757 | 0,078 | 0,224 | 62,0% | 54,2% | 0,130 |
| LightGBM A | 0,864 | 1,909 | 0,152 | 0,177 | 56,6% | 53,8% | 0,242 |
| LightGBM A+B (manual) | 0,839 | 1,629 | 0,236 | 0,242 | 58,8% | 55,4% | 0,271 |
| LightGBM A+B | 0,835 | 1,514 | 0,230 | 0,257 | 59,9% | 55,5% | 0,268 |
| LightGBM A+B+C (cor e rosto) | 0,829 | 1,436 | 0,257 | 0,280 | 60,5% | 56,2% | 0,263 |
| LightGBM A+B+C (+ SigLIP da thumb) | **0,815** | 1,355 | **0,305** | **0,334** | 61,7% | 57,4% | 0,285 |
| LightGBM A+B+C (cor, rosto e OCR) | 0,832 | 1,434 | 0,250 | 0,282 | 60,4% | 56,3% | 0,268 |
| LightGBM A+B+C (cor, rosto, OCR e SigLIP) | 0,819 | **1,354** | 0,296 | 0,327 | 61,6% | **57,8%** | **0,289** |

Verificação do ganho da thumbnail. Risco: o embedding SigLIP funcionar como "identidade visual do canal" (um `channel_id` disfarçado). Teste: validação cruzada agrupada por canal (GroupKFold, 5 folds) dentro do treino, em que os canais avaliados nunca foram vistos:

| Canais não vistos (média de 5 folds) | MAE-log | Spearman intra-canal | Par-a-par mesma época |
|---|---|---|---|
| A+B | 0,848 | 0,219 | 54,9% |
| A+B+C (cor, rosto e SigLIP) | 0,838 | 0,264 | 56,3% |

O ganho diminui, mas continua lá: a thumbnail carrega sinal sobre o vídeo, não só sobre o canal. Na validação, o ganho de MAE-log é de 0,020, com IC 95% por bootstrap pareado reamostrando canais (2.000 réplicas) de [0,009; 0,032]. (Verificação feita antes do OCR, com cor, rosto e SigLIP.)

O OCR (texto na thumbnail em 60% dos vídeos) não muda o resultado além do ruído: o SigLIP provavelmente já capta a presença de texto. Ele segue na lista e a decisão final sai do SHAP por bloco.

### Baseline vs. modelo

| Modelo | MAE-log | MAPE | Spearman | Spearman intra-canal | Acurácia par-a-par | Par-a-par mesma época | Cobertura do intervalo | F1 faixas |
|---|---|---|---|---|---|---|---|---|
| Baseline 1 (mediana do canal) | _pendente_ | _pendente_ | _pendente_ | indefinido | 50% (todos empatados) | 50% (todos empatados) | — | _pendente_ |
| Baseline 2 (linear) | _pendente_ | | | | | | — | |
| LightGBM A+B+C+D | _pendente_ | | | | | | | |

### Ablação por bloco

| Features | Spearman | MAE-log | Ganho vs. anterior |
|---|---|---|---|
| A | | | |
| A+B | | | |
| A+B+C | | | |
| A+B+C+D | | | |

### Split por canal (canais não vistos)

_Pendente: mesmas métricas da tabela principal no conjunto "canais novos"._

## Análise de erros

Roteiro definido antes dos resultados:
1. Erro por faixa real (o modelo subestima virais?).
2. Erro por setor, tipo de canal, porte e tamanho do histórico.
3. Os 20 maiores erros, com vídeo, previsão, real e top drivers do SHAP, classificados manualmente (trend, collab, impulso pago, edição posterior do texto etc.).
4. Erro em função de `std_log_likes_ult20` (o `confidence_flag` está ligando nos casos certos?).

_Pendente: preencher após o teste._

## Conclusão

_Pendente: responder H1–H5 e a pergunta norteadora com os números do teste._
