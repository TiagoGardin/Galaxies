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

**Ajuste de 15/10/2026 (antes de qualquer contato com o teste):** o prazo de entrega foi antecipado para 21/10 e o VLM fica limitado à cota gratuita, usada até a manhã de 19/10 (~1.950 vídeos a mais). Na ordem acima, isso terminaria sem nenhum vídeo de teste com bloco D, e o ganho do VLM não poderia ser medido no teste. A fila do VLM (`data/features/fila_vlm.txt`) passa a ser a mesma amostra reordenada: os 1.419 primeiros do treino (os já processados), 400 da validação, todo o teste e só então o restante, mantendo a ordem aleatória dentro de cada split. Resultado esperado: ~1.420 vídeos de treino, ~400 de validação e ~1.550 de teste com bloco D. Resultado obtido (última rodada em 19/10/2026): 1.419 de treino, 400 de validação e 1.086 de teste (a cota rendeu menos que o previsto, entre 360 e 460 vídeos por dia, por causa de novas tentativas após erros 503). A parte de vídeo do bloco C continua na ordem original, que já cobre treino, validação e parte do teste.

Consequências:
- A ablação com os blocos C (vídeo) e D é feita só nos vídeos da amostra, comparando todos os modelos nos mesmos vídeos.
- O teste em canais novos fica fora da fila, então os blocos C (vídeo) e D **não são avaliados em canais novos**. Nesse conjunto, só os blocos A, B e C (thumbnail) são avaliados.
- Antes de cada rodada, o VLM consulta a YouTube Data API (~1 unidade a cada 50 vídeos) e tira da fila os vídeos removidos ou privados, para não gastar a cota do Gemini. Em 15/10/2026, nenhum dos 8.361 estava indisponível.
- A parte de vídeo do bloco C não cobre os vídeos "made for kids" (1,6% da amostra, 11% em moda/marca; ver [FEATURES.md](FEATURES.md)). Por isso o ganho da parte de vídeo em canais de marca é medido com menos vídeos.

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
100 vídeos sorteados do treino (semente 42), entre os que têm rubrica do VLM, para comparar as duas fontes nos mesmos vídeos, são rotulados à mão (assistindo, sem olhar as flags nem a rubrica) com o formato principal. Mede-se a precisão e o recall das flags de texto (bloco B) e do campo `formato` do VLM (bloco D) contra esses rótulos (`python -m galaxies_tool.rotulos sortear` / `avaliar`).

**Situação (18/10/2026):** a amostra foi sorteada (`data/rotulos_formato.csv`), mas a rotulagem manual ainda não foi feita. As flags e a rubrica entram no modelo sem essa validação; o efeito delas é medido pela ablação.

### Registro
Cada execução é registrada no MLflow (`mlflow.db` + `mlruns/`, locais e fora do git): hiperparâmetros, lista de features, versão e hash da base e das features, datas de corte, métricas e a média do |SHAP| por feature do modelo completo. Experimentos `galaxies_validacao` (desenvolvimento, ajuste e artefato) e `galaxies_teste` (avaliação final, única). As tabelas também ficam em `resultados/` (CSV versionado no git), para quem não tem o MLflow local.

## Resultados

### Desenvolvimento (validação, 14/10/2026)

Números da **validação** (4.774 vídeos), usados só para decidir o que entra no modelo; os resultados oficiais são os do teste, abaixo. Treino: 22.430 vídeos; early stopping nos últimos 10% do treino. Bloco C ainda sem a parte de vídeo (amostra). As duas últimas linhas acrescentam o OCR da thumbnail.

Tabela histórica: as linhas de baselines, A, A+B (manual), A+B e as duas com OCR saíram de `galaxies_tool.modelo` na versão do commit `9040fcd` (hiperparâmetros padrão, sem vídeo nem VLM); o `modelo.py` atual gera a tabela de 18/10, abaixo. As linhas "cor e rosto" e "+ SigLIP da thumb" (sem OCR), a validação cruzada por canal e o IC por bootstrap abaixo vieram de execuções exploratórias anteriores ao OCR, com os mesmos dados, splits e hiperparâmetros, e não têm script próprio no repositório.

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

O OCR (texto na thumbnail em 60% dos vídeos) não muda o resultado além do ruído: o SigLIP provavelmente já capta a presença de texto. Ele segue no modelo; a ablação de 18/10 mede o bloco da thumbnail inteiro.

**Escolha da família de modelo (15/10/2026).** Mesmas 76 features (A + B + C da thumbnail), mesmo early stopping e perda L1 nos boostings; hiperparâmetros próximos do padrão de cada biblioteca, sem busca. Tabela histórica, gerada por `galaxies_tool.comparar_modelos` na versão do commit `5d5ab1c`; o script atual gera a comparação com as 106 features, abaixo.

| Modelo | MAE-log | MAPE | Spearman intra-canal | Par-a-par | Par-a-par mesma época | F1 faixas |
|---|---|---|---|---|---|---|
| LightGBM | **0,819** | 1,354 | 0,327 | 61,6% | **57,8%** | 0,289 |
| CatBoost | 0,820 | **1,196** | **0,332** | **61,7%** | 57,7% | 0,274 |
| XGBoost | 0,829 | 1,404 | 0,307 | 61,1% | 56,5% | **0,309** |
| Ridge (one-hot + imputação) | 0,857 | 1,674 | 0,218 | 58,7% | 55,4% | 0,224 |

LightGBM e CatBoost empatam dentro do ruído (exceto no MAPE, melhor no CatBoost); o XGBoost fica um pouco atrás e o modelo linear perde em todas as métricas, o que indica que as interações não lineares importam. O LightGBM é mantido por empatar com o melhor e por ter quantis nativos (intervalo) e TreeSHAP nativo (explicação do agente). Uma média LightGBM + CatBoost será testada junto com a busca de hiperparâmetros e só entra se o ganho superar o ruído (resultado em "Desenvolvimento final", abaixo: entrou).

### Desenvolvimento final: ablação com vídeo e VLM (validação, 18/10/2026)

Hiperparâmetros do Optuna (40 tentativas, `configs/lgbm.yaml`): MAE-log de 0,8166 com os valores anteriores para 0,8147 com os escolhidos, diferença dentro do ruído e otimista por ser a melhor de 40 (aula 3.4). Mesmos hiperparâmetros e semente em todos os degraus; os blocos que só existem na amostra são comparados nos vídeos da amostra (983 com vídeo, 400 com VLM). Reprodutível com `uv run python -m galaxies_tool.modelo` (tabelas completas em `resultados/validacao_metricas.csv` e `resultados/validacao_ablacao.csv`).

| Degrau (validação inteira, 4.774 vídeos) | MAE-log | Spearman | Spearman intra-canal | Par-a-par | Par-a-par mesma época |
|---|---|---|---|---|---|
| Baseline 1 (mediana do canal) | 0,875 | indefinido | indefinido | 50,0% | 50,0% |
| Baseline 2 (linear) | 0,879 | 0,078 | 0,224 | 62,0% | 54,2% |
| A | 0,863 | 0,162 | 0,165 | 56,5% | 54,3% |
| A+B (manual) | 0,842 | 0,226 | 0,233 | 58,4% | 55,2% |
| A+B | 0,832 | 0,243 | 0,249 | 59,7% | 55,8% |
| A+B+C (thumb) | 0,817 | 0,301 | 0,328 | 61,4% | 57,1% |
| A+B+C (thumb + vídeo) | 0,819 | 0,291 | 0,317 | 61,1% | 57,1% |
| **A+B+C+D (só LightGBM)** | **0,816** | 0,301 | 0,309 | 61,4% | 57,0% |
| Média LightGBM + CatBoost, A+B+C+D (modelo final) | 0,817 | — | 0,331 | — | 57,7% |

Diferenças com IC 95% (bootstrap pareado de canais, 2.000 réplicas; negativo no MAE = o novo erra menos). "Completo" é o LightGBM A+B+C+D:

| Comparação | Vídeos | Δ MAE-log | Δ par-a-par mesma época | Δ Spearman intra-canal |
|---|---|---|---|---|
| A+B (manual) − A | 4.774 | −0,021 [−0,033; −0,011] | +0,9 p.p. [−0,4; +2,1] | +0,068 [+0,034; +0,102] |
| A+B − A+B (manual): embedding de texto | 4.774 | −0,010 [−0,020; +0,001] | +0,6 p.p. [−0,8; +2,2] | +0,016 [−0,022; +0,052] |
| A+B+C (thumb) − A+B | 4.774 | −0,015 [−0,024; −0,006] | +1,3 p.p. [−0,4; +3,4] | +0,079 [+0,045; +0,113] |
| + vídeo e gancho − A+B+C (thumb) | 983 | +0,007 [−0,001; +0,014] | +1,2 p.p. [−2,5; +4,8] | −0,010 [−0,057; +0,041] |
| + VLM − A+B+C (thumb + vídeo) | 400 | −0,000 [−0,011; +0,010] | −1,4 p.p. [−9,0; +6,0] | −0,037 [−0,123; +0,062] |
| Controle negativo: + 17 colunas de ruído − A+B+C (thumb + vídeo) | 400 | +0,007 [−0,007; +0,020] | −0,7 p.p. [−7,0; +5,5] | −0,010 [−0,075; +0,052] |
| VLM: binárias − notas (H5) | 400 | −0,004 [−0,015; +0,007] | +3,6 p.p. [−2,7; +9,9] | +0,066 [−0,015; +0,152] |
| Completo − completo sem B | 4.774 | −0,012 [−0,026; +0,001] | +1,0 p.p. [−0,1; +2,0] | +0,028 [+0,001; +0,056] |
| Completo − completo sem C (thumb) | 4.774 | −0,017 [−0,025; −0,009] | +1,6 p.p. [+0,3; +3,1] | +0,069 [+0,038; +0,099] |
| Completo − completo sem C (vídeo) | 983 | +0,001 [−0,006; +0,010] | −0,1 p.p. [−2,4; +2,3] | +0,035 [−0,006; +0,077] |
| Completo − A+B+C (thumb) | 4.774 | −0,001 [−0,005; +0,003] | −0,1 p.p. [−1,1; +0,8] | −0,019 [−0,042; +0,002] |
| **Completo − baseline 1 (mediana do canal)** | 4.774 | **−0,059 [−0,093; −0,031]** | **+7,0 p.p. [+5,3; +8,9]** | indefinido no baseline |
| Completo − baseline 2 (linear) | 4.774 | −0,063 [−0,096; −0,033] | +2,8 p.p. [+0,2; +5,4] | +0,085 [+0,026; +0,142] |

Leitura:
- **O que comprovadamente ajuda:** as features manuais de texto e a thumbnail (SigLIP, cor, rosto, OCR), nas duas análises (escada e "todos menos um"). O embedding de texto fica no limite.
- **Vídeo, gancho e VLM não mostram ganho detectável** na validação, em nenhuma das duas análises: os intervalos cruzam o zero e o VLM fica indistinguível do controle negativo de ruído. Com 400 vídeos, um ganho do tamanho esperado na literatura (+0,02 a +0,04 de Spearman) não seria detectável; o teste, com ~1.150 vídeos com VLM e ~4.800 com vídeo, mede de novo.
- **O modelo completo bate os dois baselines** no MAE e na ordenação entre criativos da mesma época (+7,0 p.p. sobre a mediana do canal, IC acima de zero).

**Família de modelo com as 106 features (18/10/2026)**, mesmo protocolo de early stopping; reprodutível com `uv run python -m galaxies_tool.comparar_modelos` (`resultados/validacao_familias.csv`):

| Modelo | MAE-log | MAPE | Spearman intra-canal | Par-a-par | Par-a-par mesma época | F1 faixas |
|---|---|---|---|---|---|---|
| LightGBM | **0,816** | 1,346 | 0,309 | 61,4% | 57,0% | 0,286 |
| CatBoost | 0,822 | **1,227** | 0,325 | 61,6% | **58,0%** | 0,266 |
| XGBoost | 0,831 | 1,416 | 0,313 | 60,9% | 56,3% | **0,309** |
| Ridge | 0,861 | 1,642 | 0,217 | 58,2% | 55,1% | 0,224 |
| Média LightGBM + CatBoost | 0,816 | 1,275 | **0,333** | **61,9%** | 57,7% | 0,271 |

A média contra o LightGBM sozinho (IC 95%, bootstrap de canais): MAE-log +0,001 [−0,004; +0,006], par-a-par mesma época +0,7 p.p. [+0,1; +1,3], Spearman intra-canal +0,024 [+0,009; +0,040]. Pela regra registrada em 15/10 ("a média só entra se o ganho superar o ruído"), **a média entra**: ela empata no erro e ganha nas duas métricas de ordenação, que são as da decisão. O intervalo continua vindo dos quantis do LightGBM, e o SHAP da média é a média dos SHAPs dos dois modelos (aditivo).

### Protocolo congelado antes do teste (18/10/2026)

Decidido só com a validação, antes de abrir o teste:
1. **Modelo avaliado e servido:** média de LightGBM e CatBoost (L1) com todos os blocos (A+B+C+D, 106 colunas), como pré-registrado na tabela "Baseline vs. modelo". A validação não mostrou prejuízo mensurável dos blocos sem ganho (completo − A+B+C (thumb): −0,001 de MAE, −0,1 p.p. no par-a-par da mesma época), o TAPI exige a camada multimodal, e a ablação no teste responde à H3. O modelo enxuto A+B+C (thumb) é uma alternativa equivalente na validação; se o teste confirmar que vídeo e VLM não ajudam, ele é a recomendação para produção (mais barato e sem a cota do Gemini).
2. **Hiperparâmetros:** LightGBM com `configs/lgbm.yaml`; CatBoost com os valores fixos de `modelo.PARAMS_CATBOOST`; número de árvores de cada um escolhido por early stopping no treino (últimos 10%, em ordem de tempo).
3. **Intervalo:** quantis de 10% e 90% + CQR medido na validação com o modelo que não a viu (cobertura crua 71,4% → correção de +0,225 em log).
4. **Refit:** o artefato final é reajustado em treino + validação (27.204 vídeos), com os mesmos hiperparâmetros e o mesmo número de árvores, e mantém a correção e os limiares medidos na validação. A escada de ablação do teste é reajustada do mesmo jeito.
5. **`confidence_flag`:** as 7 regras e limiares do MODEL_CARD, calculados na validação (16,7% dos vídeos da validação sinalizados).
   Na validação, o modelo de desenvolvimento (só treino) tem MAE-log 0,817, Spearman intra-canal 0,331, par-a-par na mesma época 57,7% e cobertura de 80,0% (por construção, porque a validação é a calibração).
6. **Teste:** uma única execução de `python -m galaxies_tool.avaliacao_final` depois da última rodada do VLM (19/10), com tabela principal, ablação, recortes, canais novos, avaliação do flag e os 20 maiores erros.
7. **Demonstração:** os 3 Shorts elegíveis mais recentes de cada um dos 4 canais de demonstração (regra fixada em `demo.py`).

### Baseline vs. modelo (teste temporal, avaliado uma vez em 19/10/2026)

4.861 vídeos de 02/07 a 12/09/2026, de canais vistos no treino; modelo final reajustado em treino + validação. Reprodução: `uv run python -m galaxies_tool.avaliacao_final --forcar` (tabelas em `resultados/teste_*.csv` e `resultados/teste_resumo.json`).

| Modelo | MAE-log | MAPE | Spearman | Spearman (curtidas absolutas) | Spearman intra-canal | Acurácia par-a-par | Par-a-par mesma época | Cobertura do intervalo | F1 faixas |
|---|---|---|---|---|---|---|---|---|---|
| Baseline 1 (mediana do canal) | 0,835 | 1,876 | indefinido | 0,858 | indefinido | 50,0% | 50,0% | — | 0,128 |
| Baseline 2 (linear) | 0,842 | 1,939 | 0,081 | 0,858 | 0,259 | 60,6% | 53,4% | — | 0,128 |
| **Modelo final (média LightGBM + CatBoost, A+B+C+D)** | **0,760** | **1,200** | **0,405** | **0,882** | **0,357** | **61,2%** | **59,1%** | **82,7%** (largura média 2,68 em log) | **0,278** |

Diferenças do modelo final com IC 95% (bootstrap pareado de canais):
- contra a mediana do canal: MAE-log −0,076 [−0,097; −0,055]; par-a-par +11,2 p.p. [+9,6; +12,8]; par-a-par na mesma época +9,1 p.p. [+7,3; +10,9];
- contra a regressão linear: MAE-log −0,082 [−0,102; −0,063]; par-a-par +0,6 p.p. [−1,6; +3,1] (empate: a regressão à média já ordena bem pares de épocas diferentes); par-a-par na mesma época +5,7 p.p. [+3,7; +8,0]; Spearman intra-canal +0,098 [+0,041; +0,151].

Cobertura do intervalo de 80% por grupo: beleza 81,8%, moda 84,8%; criadores 82,7%, marcas 82,6%. Por faixa do resultado real, a cobertura cai para 73% nos vídeos "abaixo" e para **43% nos virais** (≥ 3× a mediana do canal): o intervalo é calibrado na média, não na cauda.

### Ablação por bloco (teste)

Mesma escada da validação, com os mesmos hiperparâmetros e número de árvores, cada degrau reajustado em treino + validação (ver "Protocolo congelado").

| Features (teste inteiro, 4.861 vídeos) | Spearman | MAE-log | Spearman intra-canal | Par-a-par mesma época |
|---|---|---|---|---|
| A | 0,249 | 0,806 | 0,208 | 55,6% |
| A+B | 0,365 | 0,773 | 0,322 | 59,5% |
| A+B+C (thumb) | 0,406 | 0,761 | 0,336 | 59,4% |
| A+B+C (thumb + vídeo) | 0,392 | 0,764 | 0,349 | 58,8% |
| A+B+C+D (só LightGBM) | 0,405 | 0,760 | 0,366 | 59,2% |

| Comparação (IC 95%) | Vídeos | Δ MAE-log | Δ par-a-par mesma época | Δ Spearman intra-canal |
|---|---|---|---|---|
| A+B (manual) − A | 4.861 | −0,021 [−0,033; −0,009] | +2,7 p.p. [+1,0; +4,2] | +0,070 [+0,039; +0,103] |
| Embedding de texto: A+B − A+B (manual) | 4.861 | −0,012 [−0,021; −0,003] | +1,2 p.p. [−0,2; +2,7] | +0,044 [+0,011; +0,078] |
| Thumbnail: A+B+C (thumb) − A+B | 4.861 | −0,012 [−0,021; −0,003] | −0,1 p.p. [−1,4; +1,4] | +0,015 [−0,018; +0,050] |
| Vídeo e gancho − A+B+C (thumb) | 4.783 | +0,003 [−0,003; +0,009] | −0,6 p.p. [−1,4; +0,2] | +0,014 [−0,003; +0,034] |
| **VLM − A+B+C (thumb + vídeo)** | 1.086 | **−0,009 [−0,014; −0,003]** | **+2,3 p.p. [+0,4; +4,4]** | +0,038 [−0,004; +0,080] |
| Controle negativo: ruído − A+B+C (thumb + vídeo) | 1.086 | +0,007 [−0,000; +0,015] | +0,3 p.p. [−2,1; +2,3] | +0,002 [−0,040; +0,045] |
| VLM binárias − notas (H5) | 1.086 | +0,003 [−0,005; +0,010] | −0,5 p.p. [−3,5; +1,8] | −0,021 [−0,060; +0,016] |
| Completo − completo sem B | 4.861 | −0,011 [−0,019; −0,004] | +1,7 p.p. [−0,2; +3,4] | +0,027 [+0,004; +0,051] |
| Completo − completo sem C (thumb) | 4.861 | −0,010 [−0,021; +0,001] | −0,4 p.p. [−2,1; +1,3] | +0,031 [+0,003; +0,059] |
| Completo − completo sem C (vídeo) | 4.783 | −0,002 [−0,007; +0,002] | −0,3 p.p. [−0,8; +0,3] | +0,027 [+0,008; +0,045] |
| Completo − A+B+C (thumb) | 4.861 | −0,001 [−0,007; +0,005] | −0,2 p.p. [−0,9; +0,5] | +0,029 [+0,012; +0,046] |

Leitura:
- **Texto e thumbnail ajudam** no erro e, em alguma métrica de ordenação, nas duas análises.
- **O VLM ajudou no teste**, ao contrário da validação: nos 1.086 vídeos com rubrica, menos erro e +2,3 p.p. no par-a-par da mesma época, com IC acima de zero, enquanto o controle negativo (ruído no lugar da rubrica) não ganha nada. Na validação, com 400 vídeos, o mesmo efeito não era detectável. O efeito é pequeno e foi medido numa única janela; trate como evidência moderada.
- **A parte de vídeo (cortes, movimento, voz, energia, gancho) não ajuda no erro nem no par-a-par**; só o Spearman intra-canal da ablação "todos menos um" sobe (+0,027), um sinal fraco e isolado.
- **O ensemble** (média com o CatBoost) teve no teste o mesmo MAE-log do LightGBM sozinho (0,760) e Spearman intra-canal um pouco menor (0,357 contra 0,366); o ganho de ordenação visto na validação não se repetiu.

### Split por canal (canais não vistos)

1.146 vídeos de 51 canais que nunca entraram no treino, no mesmo período do teste. Só os blocos A, B e C (thumbnail) existem para eles (fora da amostra de vídeo e VLM).

| Modelo | MAE-log | Spearman | Spearman intra-canal | Par-a-par | Par-a-par mesma época | F1 faixas |
|---|---|---|---|---|---|---|
| Baseline 1 (mediana do canal) | **0,830** | indefinido | indefinido | 50,0% | 50,0% | 0,129 |
| Baseline 2 (linear) | 0,859 | 0,065 | 0,139 | 57,3% | 53,4% | 0,129 |
| Modelo final | 0,843 | 0,200 | 0,145 | 54,4% | 52,3% | 0,237 |

Contra a mediana do canal: MAE-log +0,013 [−0,030; +0,057] (empate), par-a-par +4,4 p.p. [+2,9; +7,0], par-a-par na mesma época +2,3 p.p. [+0,4; +5,8]. **Em canais nunca vistos, o modelo praticamente perde o valor:** o erro empata com a mediana do canal e a ordenação fica pouco acima do acaso.

## Análise de erros

Roteiro definido antes dos resultados:
1. Erro por faixa real (o modelo subestima virais?).
2. Erro por setor, tipo de canal, porte e tamanho do histórico.
3. Os 20 maiores erros, com vídeo, previsão, real e top drivers do SHAP, classificados manualmente (trend, collab, impulso pago, edição posterior do texto etc.).
4. Erro em função de `std_log_likes_ult20` (o `confidence_flag` está ligando nos casos certos?).

**1. Por faixa real** (`resultados/teste_recortes.csv`): o modelo acerta bem o "na média" (MAE-log 0,28) e erra muito os extremos. Nos **virais** (≥ 3× a mediana, 14% do teste), MAE-log 1,79 e cobertura do intervalo de 43%: o modelo encolhe para o centro e quase nunca prevê viral. Nos vídeos "abaixo", MAE-log 0,91 e cobertura de 73%.

**2. Por grupo** (MAE-log do modelo × da mediana do canal): beleza 0,81 × 0,88; moda 0,65 × 0,73; criadores 0,75 × 0,82; **marcas 0,81 × 0,90** (marcas erram mais, mas o ganho sobre o baseline é maior); canais com menos de 20 mil inscritos 0,90 × 0,96 (o pior porte); canais de 100–500 mil 0,63 × 0,67. Ordenação na mesma época: moda 63,7% contra beleza 57,8%.

**3. Os 20 maiores erros** (`resultados/teste_maiores_erros.csv`), classificados pelo título, pelas visualizações e pelo canal (a classificação assistindo aos vídeos fica registrada na coluna `classificacao_manual`):
- **13 virais** (resultado 85× a 2.400× a mediana do canal): 8 em canais pequenos (mediana de 44 a 106 curtidas), 6 deles com `#youtubeshopping` e temas de "achadinho" ou ASMR, e 5 em canais médios (mediana de 320 a 540). É o caso que nenhuma feature antecipa: o algoritmo levou o vídeo para fora da audiência do canal. O mesmo título ("Gloss stick Vult ou Batom Garoto?") aparece duas vezes, em vídeos diferentes.
- **3 anúncios com mídia paga** (Avon e Época Cosméticos em canais de marca, NIVEA numa publi de criador): milhões de visualizações com poucas curtidas por visualização, sinal de impulso pago, que o modelo não observa (ver Limitações).
- **4 publis de cupom da SHEIN em canais grandes**, com 2 a 18 curtidas contra medianas de 1.271 a 4.758: posts patrocinados que o público quase ignora. O modelo os previu abaixo da média, mas não tanto.

**4. Instabilidade do canal e `confidence_flag`:** o MAE-log cresce com a instabilidade do canal (0,53 no quartil mais estável, 1,03 no mais instável). O flag ligou em 16% do teste (777 vídeos), e os sinalizados erram bem mais: MAE-log 1,10 contra 0,70, cobertura 77% contra 84%. Por motivo: `canal_instavel` 1,13 (426 vídeos), `intervalo_largo` 1,12 (535), `extracao_incompleta` 1,22 (78, os vídeos que não puderam ser baixados), `canal_pouco_historico` 1,53 (9) e `fora_da_distribuicao` 0,81 (47, a regra mais fraca, mas ainda acima dos não sinalizados). Todas as regras separam erro, então todas ficam.

**Extensão (LLM zero-shot × modelo):** nos 1.086 vídeos com rubrica, a nota `potencial_viral_0a10` que o próprio VLM deu, sem ver o histórico, ordena pouco melhor que o acaso (par-a-par na mesma época 52,4%, Spearman intra-canal 0,08), contra 60,2% e 0,32 do modelo nos mesmos vídeos. A opinião do LLM sobre o vídeo não substitui um modelo treinado com o histórico.

## Conclusão

| Hipótese | Resultado no teste | Veredito |
|---|---|---|
| H1: o modelo completo bate a mediana do canal | MAE-log 0,760 × 0,835 (−0,076 [−0,097; −0,055]); par-a-par 61,2% e 59,1% na mesma época, ambos acima de 50% com IC | **Confirmada** |
| H2: ~60% de acerto na ordenação entre criativos do mesmo canal | 61,2% com todos os pares; 59,1% [57,3%; 60,9%] em pares da mesma época | **Confirmada** (no limite, na métrica mais exigente) |
| H3: C e D somam +0,02 a +0,04 de Spearman sobre A+B | Spearman 0,365 (A+B) → 0,405 (A+B+C+D): +0,040, quase todo da thumbnail; o VLM soma na sua amostra, o vídeo não | **Confirmada no total**, com a ressalva sobre a parte de vídeo |
| H4: o desempenho cai em canais nunca vistos e em marcas | Canais novos: empate com a mediana no erro e 52,3% na mesma época. Marcas: MAE-log 0,81 contra 0,75 dos criadores, mesma ordenação | **Confirmada** (canais novos); **parcial** (marcas: mais erro, ordenação igual) |
| H5: os campos binários do VLM ajudam mais que as notas | Binárias × notas: sem diferença (−0,5 p.p. [−3,5; +1,8]) | **Não confirmada** |

**Pergunta norteadora.** Sim, em parte: antes da publicação, a partir do criativo e do histórico do canal, o modelo ordena dois criativos do mesmo canal publicados na mesma semana corretamente em cerca de 59% dos casos (a mediana do canal acerta 50%) e erra o número de curtidas por um fator típico de e^0,76 ≈ 2,1×, com um intervalo de 80% que cobriu 83% dos casos. O valor vem principalmente do histórico do canal, do texto e da imagem de capa; a rubrica do VLM soma um pouco. O modelo não antecipa virais (o maior erro) e não serve para canais sem histórico na base: nesses casos, o `confidence_flag` e a mediana do canal são a resposta honesta.

**Recomendação para produção** (o "Protocolo congelado" deixou a escolha para depois do teste): manter o VLM, que somou ordenação no teste, e tornar opcional a parte de vídeo (download, Whisper, cortes, movimento, gancho), que não somou em nenhuma métrica principal e é a etapa mais cara e frágil do agente (download bloqueável, vídeos "made for kids"). Retreinar sem ela é uma decisão para a próxima versão, com nova validação; o artefato avaliado aqui continua sendo o servido.
