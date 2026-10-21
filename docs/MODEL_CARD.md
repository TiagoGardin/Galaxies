# Model Card

**Versão servida:** `lgbm_v2` (`models/lgbm_v2.joblib` e `.json`), treinada em 21/10/2026 sobre a `base_v1` (coleta de 12/10/2026), com 27.204 vídeos de treino + validação (01/04/2025 a 01/07/2026), 142 features. Escolhida na validação depois do teste temporal e confirmada no teste prospectivo pela regra pré-registrada (EXPERIMENTO.md, "Depois do teste"). Registrada no MLflow (run `artefato_lgbm_v2` em `galaxies_validacao`; avaliação em `galaxies_prospectivo`).

**Versão anterior:** `lgbm_final` (v1, `models/lgbm_final.joblib`), treinada em 18/10/2026 com os mesmos dados e 106 features: é a avaliada no teste temporal (experimento `galaxies_teste`). A v2 é a v1 com duas mudanças: PCA de 32 componentes (em vez de 16) nos embeddings de texto e de thumbnail, e 4 features de comparáveis (bloco R, abaixo). Hiperparâmetros, protocolo de treino, intervalo e `confidence_flag` são os mesmos. A tool usa a v1 se os embeddings da base não estiverem presentes (as features de comparáveis dependem deles) ou com `GALAXIES_MODELO=models/lgbm_final.joblib`.

## Tarefa

**Alvo (decidido):**

```
alvo = log1p(curtidas) − log1p(mediana de curtidas dos últimos (até) 20 Shorts do canal
                                publicados ≥ 30 dias antes do vídeo-alvo)
```

- **Curtidas, não views:** é o alvo primário do TAPI, e a forma de contar views mudou em 31/03/2025 e em 24/08/2026.
- **Relativo ao canal:** isola o efeito do conteúdo, que é o que interessa para decidir entre criativos. O baseline (mediana do canal) equivale a prever `alvo = 0`.
- **Conversão para o contrato da tool:** `predicted_metric = (1 + mediana_canal) × e^(previsão) − 1` (inversa exata do `log1p`); `relative_to_channel` é a própria previsão; os quantis do intervalo são convertidos da mesma forma.
- **Mínimo de histórico:** pelo menos 5 Shorts maduros anteriores para entrar no treino. Na tool, o canal precisa de pelo menos 1, e abaixo de 10 o `confidence_flag` liga (`canal_pouco_historico`).

**Alvo secundário (comunicação):** a previsão relativa é convertida em faixas pela razão `(1 + curtidas previstas) / (1 + mediana do canal)`, que é exatamente `e^alvo` (para contagens altas, praticamente igual a curtidas / mediana):

| Faixa | Razão prevista | Alvo relativo |
|---|---|---|
| Abaixo | < 0,67× | < −0,40 |
| Na média | 0,67× a 1,5× | −0,40 a 0,41 |
| Acima | 1,5× a 3× | 0,41 a 1,10 |
| Viral | ≥ 3× | ≥ 1,10 |

As faixas são simétricas em escala log (0,67 ≈ 1/1,5) e são avaliadas com F1 macro.

## Algoritmo e configuração

- **Modelo principal:** média de um LightGBM e de um CatBoost, os dois com perda L1 (o alvo tem cauda longa mesmo em log). O LightGBM lida bem com features heterogêneas, trata ausentes e tem objetivo de quantil; a média com o CatBoost entrou por melhorar a ordenação na validação com IC acima de zero (Spearman intra-canal +0,023, par-a-par na mesma época +0,6 p.p.), sem mudar o erro; no teste, empatou com o LightGBM sozinho. XGBoost e Ridge ficaram atrás (ver EXPERIMENTO.md).
- **Hiperparâmetros:** busca com Optuna (TPE, semente 42, 40 tentativas) **só na validação**, minimizando o MAE-log, com early stopping nos últimos 10% do treino (em ordem de tempo). Espaço de busca: `num_leaves`, `learning_rate`, `min_child_samples`, `feature_fraction`, `bagging_fraction`, `lambda_l1`, `lambda_l2`. Valores escolhidos em [`configs/lgbm.yaml`](../configs/lgbm.yaml) (`python -m galaxies_tool.ajustar`); cada execução é registrada no MLflow.
- **Intervalo de predição de 80%:** dois modelos LightGBM com `objective="quantile"` (`alpha` 0,10 e 0,90, folhas de pelo menos 100 vídeos, early stopping pela pinball loss), calibrados por **predição conformal sobre os quantis (CQR, Romano et al. 2019)**: a correção é medida na validação, a fatia mais recente antes do teste, com o modelo que não a viu. Os limites são ordenados para nunca cruzarem a previsão.
- **Treino final (refit):** escolhidos os hiperparâmetros, o número de árvores (por early stopping), a correção do CQR e os limiares do `confidence_flag` com o modelo treinado só no treino, o artefato final é reajustado em **treino + validação**, para não perder os meses mais recentes, e mantém a correção e os limiares medidos fora da amostra. É esse artefato que o teste avalia uma vez e que o agente serve (`models/lgbm_final.joblib`, `python -m galaxies_tool.artefato`).
- **Baselines:** (1) mediana do canal (prever alvo = 0); (2) regressão linear de `log1p(curtidas)` sobre `log_mediana_likes_ult20`.
- **Explicabilidade:** TreeSHAP exato de cada modelo (`pred_contrib=True` no LightGBM, `ShapValues` no CatBoost), com a média dos dois (aditiva como a previsão) por previsão, convertido em efeito multiplicativo (`e^φ`); os componentes de cada embedding são somados num único fator, e os valores também são agregados por bloco. O código confere a aditividade (valor base + soma = previsão) a cada chamada.
- **`confidence_flag` (baixa confiança):** liga se qualquer regra disparar; cada regra tem um código estável e uma frase (contrato da tool). Limiares calculados uma vez, fora do teste, e versionados no artefato:

  | Código | Regra |
  |---|---|
  | `canal_pouco_historico` | menos de 10 Shorts maduros (≥ 30 dias) no histórico do canal |
  | `canal_instavel` | `std_log_likes_ult20` acima do percentil 90 do treino |
  | `intervalo_largo` | intervalo de 80% mais largo que o percentil 90 das larguras na calibração |
  | `fora_da_distribuicao` | distância ao 10º vizinho do treino, no espaço padronizado dos PCAs de texto e thumbnail, acima do percentil 99 da calibração |
  | `setor_fora_do_treino` | setor diferente de moda e beleza (ex.: automotivo) |
  | `extracao_incompleta` | parte do criativo não pôde ser analisada (imagem, vídeo ou rubrica do VLM) |
  | `data_fora_do_periodo` | publicação planejada mais de 180 dias depois (ou 365 antes) do fim dos dados |

  A regra só se justifica se os casos sinalizados errarem mais. No teste, todas separam erro: MAE-log de 1,10 nos sinalizados contra 0,70 nos demais, de 0,81 (`fora_da_distribuicao`, a mais fraca) a 1,53 (`canal_pouco_historico`) por motivo.

## Features

| Bloco | Conteúdo | Fonte |
|---|---|---|
| A | Histórico do canal (sempre com Shorts ≥ 30 dias antes), tempo da publicação, metadados | API do YouTube |
| B | Features manuais do título/descrição + embedding de texto (PCA 16) | Texto |
| C | Thumbnail (base inteira): SigLIP 2 (PCA 16), cor, rosto, OCR. Vídeo (amostra): cortes, movimento, voz, energia de áudio, SigLIP 2 do gancho (PCA 8) | Vídeo e thumbnail (offline) |
| D | Rubrica JSON respondida pelo VLM (amostra): 8 sim/não, 5 notas 0–10 e 4 categorias | Gemini 3.1 Flash-Lite |
| R (só v2) | Desempenho real de Shorts parecidos publicados pelo menos 30 dias antes: do próprio canal (média ponderada dos 5 mais parecidos e similaridade do mais parecido), dos 10 mais recentes do canal e de 20 parecidos de outros canais do setor | Base coletada + embeddings de texto e thumbnail |

As partes de vídeo do bloco C e o bloco D existem só numa amostra da base (cota e ritmo de download; ver EXPERIMENTO.md); fora dela, as colunas são ausentes e o LightGBM as trata nativamente. No agente, todo criativo em vídeo passa pelas duas etapas.

Lista completa, regras anti-vazamento e features descartadas: [FEATURES.md](FEATURES.md).

## Métricas

Métricas usadas (definições em [EXPERIMENTO.md](EXPERIMENTO.md)): MAE em log, MAPE em curtidas, Spearman global e intra-canal, acurácia par-a-par intra-canal (todos os pares e pares da mesma época, a até 7 dias), cobertura do intervalo de 80% e F1 macro das faixas.

**Teste temporal** (4.861 vídeos, 02/07 a 12/09/2026, avaliado uma única vez, com a v1):

| | MAE-log | Spearman intra-canal | Par-a-par | Par-a-par mesma época | Cobertura 80% | F1 faixas |
|---|---|---|---|---|---|---|
| Mediana do canal (baseline) | 0,835 | indefinido | 50,0% | 50,0% | — | 0,128 |
| **Este modelo** | **0,760** | **0,357** | **61,2%** | **59,1%** | **82,7%** | **0,278** |

| Recorte | MAE-log (modelo × baseline) | Par-a-par mesma época | Cobertura 80% |
|---|---|---|---|
| Beleza (3.344) | 0,81 × 0,88 | 57,8% | 81,8% |
| Moda (1.517) | 0,65 × 0,73 | 63,7% | 84,8% |
| Criadores (3.953) | 0,75 × 0,82 | 59,2% | 82,7% |
| Marcas (908) | 0,81 × 0,90 | 59,0% | 82,6% |
| Canais nunca vistos (1.146) | 0,84 × 0,83 | 52,3% | — |
| Vídeos virais (≥ 3× a mediana) | 1,79 × 2,01 | — | 42,6% |

`confidence_flag`: ligou em 16% do teste; MAE-log 1,10 nos sinalizados contra 0,70 nos demais. Detalhes e IC em [EXPERIMENTO.md](EXPERIMENTO.md).

**Teste prospectivo** (Shorts publicados de 12/09 a 21/09/2026, depois de toda a base; curtidas de 21/10; extração sem vídeo nem VLM; avaliado uma única vez, com v1 e v2):

| Canais vistos (852 vídeos) | MAE-log | Spearman intra-canal | Par-a-par mesma época | Cobertura 80% | Interval score | Viés mediano |
|---|---|---|---|---|---|---|
| Mediana do canal (baseline) | 0,786 | indefinido | 50,0% | — | — | — |
| v1 | 0,772 | 0,232 | 59,9% | 80,2% | 3,76 | +0,16 |
| **v2 (servida)** | **0,745** | 0,250 | **61,7%** | 81,9% | 3,67 | +0,13 |

v2 − v1: MAE-log −0,027 [−0,046; −0,008]; par-a-par na mesma época +1,8 p.p. [−0,4; +4,0]. Canais novos (262 vídeos): MAE-log 0,845 (v2) × 0,899 (v1) × 0,913 (mediana). Na parte com vídeo e VLM (454 vídeos de canais vistos), v2 − v1 no MAE-log −0,019 [−0,041; +0,003] e no par-a-par +0,7 p.p.; nos 132 de canais novos, a v2 ordenou pior (54,1% × 63,1%, IC largo). Na validação, o modelo de desenvolvimento da v2 tem MAE-log 0,811 (v1: 0,817) e par-a-par na mesma época de 58,9% (v1: 57,7%). O viés positivo vem da idade das curtidas (30 a 39 dias, contra 30 dias a 18 meses no treino): em vídeos recém-maduros, o modelo superestima o nível em ~15%.

## Usos pretendidos

- **Usuário:** a persona sintética da Galaxies, que chama o modelo como tool (função Python ou servidor MCP) para complementar a avaliação qualitativa com um número.
- **Decisão:** **comparar criativos do mesmo canal antes da publicação** ("qual destes 3 vídeos tende a render mais curtidas para esta marca?") e sinalizar quando um criativo está muito abaixo ou acima do padrão do canal.
- **Contexto válido:** YouTube Shorts de moda e beleza, canais brasileiros, canal com histórico de pelo menos 5 Shorts.

## Usos NÃO pretendidos

- **Prever o número absoluto exato de curtidas** como promessa a um cliente: o valor absoluto é mal previsível; o modelo é desenhado para ordenar e comparar.
- **Comparar canais diferentes entre si** ("o criador X é melhor que o Y"): o alvo é relativo a cada canal.
- **Avaliar criadores individualmente** (contratação, remuneração, ranking de pessoas).
- **Outras plataformas** (TikTok, Instagram Reels), outros setores ou outros países sem retreino.
- **Prever views, comentários ou vendas:** o modelo foi treinado só para curtidas.
- **Canais sem histórico:** sem histórico, o modelo cai no `confidence_flag` e a previsão não deve guiar decisão.

## Limitações

- **Marcas são minoria na base** (52 de 260 canais e menos vídeos por canal): o desempenho em canais de marca, o perfil de cliente da Galaxies, é menos garantido e é reportado à parte.
- **Fatores fora do criativo não observados:** impulso pago, distribuição do algoritmo, collabs não declaradas, trends do momento.
- **Curtidas medidas no dia da coleta**, não num horizonte fixo; vídeos mais antigos tiveram mais tempo de acumular (mitigado pela idade mínima de 30 dias).
- **Texto e thumbnail podem ter sido editados depois da publicação.**
- **Notas subjetivas do VLM** podem refletir o reconhecimento de criadores famosos.
- **Deriva:** a plataforma muda a distribuição dos Shorts; o modelo precisa de retreino periódico com coletas novas.
- Condições do `confidence_flag`: ver "Algoritmo e configuração".
- **Onde mais erra** (teste): vídeos que viralizam muito acima do canal (não os antecipa; o intervalo cobre só 43% deles), conteúdo patrocinado (14 dos 20 maiores erros, assistidos um a um, são publis: com link de compra, vão muito acima do canal; de criadores grandes, às vezes têm engajamento quase nulo), mídia paga, canais pequenos (< 20 mil inscritos) e canais instáveis.
- **Canais que o modelo nunca viu no treino:** no teste temporal (v1), o erro empata com a mediana do canal e a ordenação fica pouco acima do acaso (52,3% na mesma época); use a previsão como ordem de grandeza. No teste prospectivo, os canais novos da base ficaram melhores (v2: 58,0% na mesma época), mas eles têm os vídeos antigos na base; um canal de fora, informado só por `channel_stats`, não tem os comparáveis do próprio canal.
- **A v2 foi escolhida depois do teste temporal** (validação, 13 variantes) e julgada num único período prospectivo de 9 dias; o ganho de ordenação sobre a v1 ainda tem IC cruzando zero.
