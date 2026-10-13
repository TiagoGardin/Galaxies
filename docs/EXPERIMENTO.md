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

### Métricas
| Métrica | Definição | Por que |
|---|---|---|
| MAE-log | Erro absoluto médio no alvo (escala log) | Erro típico em "quantas vezes" o vídeo fica acima/abaixo do canal |
| MAPE | Erro percentual médio sobre `1 + curtidas`, após converter a previsão de volta. O `+1` é necessário porque 372 vídeos-alvo têm 0 curtidas (divisão por zero) e é coerente com o `log1p` do alvo | Erro relativo legível para o cliente |
| Spearman global | Correlação de postos entre previsão e real em todo o teste | Ordenação geral |
| Spearman intra-canal | Média do Spearman calculado dentro de cada canal (canais com ≥ 5 vídeos no teste). Indefinido para previsões constantes, como a do baseline 1 | Ordenação entre criativos da mesma marca, que é a decisão real |
| Acurácia par-a-par intra-canal | Fração de pares de vídeos do mesmo canal em que o modelo ordena igual ao real. Pares empatados no real são ignorados; empate na previsão conta meio acerto | "Qual de dois criativos rende mais", a pergunta da Galaxies |
| Cobertura do intervalo | Fração de vídeos com curtidas reais dentro do intervalo de 80% | Calibração da incerteza (alvo: ~80%) |
| F1 macro das faixas | F1 nas faixas abaixo / na média / acima / viral | Qualidade da comunicação em categorias |

Todas as métricas são reportadas também por setor (beleza × moda) e por tipo de canal (marca × criador).

### Validação das flags de formato
Cerca de 100 vídeos sorteados do treino são rotulados à mão (assistindo) com o formato real. Mede-se a precisão e o recall das flags de texto (bloco B) e do campo `formato` do VLM (bloco D) contra esses rótulos.

### Registro
Cada execução é registrada no MLflow (`mlruns/`, local): parâmetros, versão da base, datas de corte, métricas e importâncias SHAP. O run final é referenciado no README.

## Resultados

### Baseline vs. modelo

| Modelo | MAE-log | MAPE | Spearman | Spearman intra-canal | Acurácia par-a-par | Cobertura do intervalo | F1 faixas |
|---|---|---|---|---|---|---|---|
| Baseline 1 (mediana do canal) | _pendente_ | _pendente_ | _pendente_ | indefinido | 50% (todos empatados) | — | _pendente_ |
| Baseline 2 (linear) | _pendente_ | | | | | — | |
| LightGBM A+B+C+D | _pendente_ | | | | | | |

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
