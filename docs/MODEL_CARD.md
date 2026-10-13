# Model Card

## Tarefa

**Alvo (decidido):**

```
alvo = log1p(curtidas) − log1p(mediana de curtidas dos últimos (até) 20 Shorts do canal
                                publicados ≥ 30 dias antes do vídeo-alvo)
```

- **Curtidas, não views:** é o alvo primário do TAPI, e a forma de contar views mudou em 31/03/2025 e em 24/08/2026.
- **Relativo ao canal:** isola o efeito do conteúdo, que é o que interessa para decidir entre criativos. O baseline (mediana do canal) equivale a prever `alvo = 0`.
- **Conversão para o contrato da tool:** `predicted_metric = (1 + mediana_canal) × e^(previsão) − 1` (inversa exata do `log1p`); `relative_to_channel` é a própria previsão; os quantis do intervalo são convertidos da mesma forma.
- **Mínimo de histórico:** pelo menos 5 Shorts anteriores para entrar no treino. Abaixo disso, a tool liga o `confidence_flag`.

**Alvo secundário (comunicação):** a previsão relativa é convertida em faixas pela razão `(1 + curtidas previstas) / (1 + mediana do canal)`, que é exatamente `e^alvo` (para contagens altas, praticamente igual a curtidas / mediana):

| Faixa | Razão prevista | Alvo relativo |
|---|---|---|
| Abaixo | < 0,67× | < −0,40 |
| Na média | 0,67× a 1,5× | −0,40 a 0,41 |
| Acima | 1,5× a 3× | 0,41 a 1,10 |
| Viral | ≥ 3× | ≥ 1,10 |

As faixas são simétricas em escala log (0,67 ≈ 1/1,5) e são avaliadas com F1 macro.

## Algoritmo e configuração

- **Modelo principal:** LightGBM (`objective="regression_l1"`), escolhido por lidar bem com features tabulares heterogêneas, valores ausentes nativos e por ter objetivo de quantil. O L1 é usado porque o alvo tem cauda longa mesmo em log.
- **Intervalo de predição de 80%:** dois modelos LightGBM adicionais com `objective="quantile"` e `alpha=0.10` e `alpha=0.90`. Se a cobertura na validação ficar longe de 80%, os limites são recalibrados por predição conformal (MAPIE) na validação.
- **Hiperparâmetros:** ajustados com Optuna **só no split de validação**, com early stopping. Espaço de busca: `num_leaves`, `learning_rate`, `min_child_samples`, `feature_fraction`, `bagging_fraction`, `lambda_l1`, `lambda_l2`. Configuração final em `configs/lgbm.yaml`; cada execução é registrada no MLflow.
- **Baselines:** (1) mediana do canal (prever alvo = 0); (2) regressão linear de `log1p(curtidas)` sobre `log_mediana_likes_ult20`.
- **Explicabilidade:** SHAP (`TreeExplainer`) por previsão; os valores são agregados por bloco de features para a explicação em linguagem natural.
- **`confidence_flag` (baixa confiança)** liga se qualquer condição for verdadeira:
  - menos de 5 Shorts no histórico do canal;
  - intervalo de 80% mais largo que o percentil 90 das larguras na validação;
  - features multimodais ausentes (download ou VLM falhou);
  - canal com `std_log_likes_ult20` acima do percentil 90 do treino.

## Features

| Bloco | Conteúdo | Fonte |
|---|---|---|
| A | Histórico do canal (sempre com Shorts ≥ 30 dias antes), tempo da publicação, metadados | API do YouTube |
| B | Features manuais do título/descrição + embedding de texto (PCA 16) | Texto |
| C | Embeddings SigLIP 2 da thumbnail e do gancho, estética, rosto, OCR, cortes, movimento, voz, energia de áudio | Vídeo e thumbnail (offline) |
| D | Rubrica JSON respondida pelo VLM | Gemini Flash-Lite |

Lista completa, regras anti-vazamento e features descartadas: [FEATURES.md](FEATURES.md).

## Métricas

Métricas usadas (definições em [EXPERIMENTO.md](EXPERIMENTO.md)): MAE em log, MAPE em curtidas, Spearman global e intra-canal, acurácia par-a-par intra-canal, cobertura do intervalo de 80% e F1 macro das faixas.

_Pendente: preencher com os números do conjunto de teste após o treino final, incluindo os baselines, o recorte por setor e por tipo de canal (marca × criador)._

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
- _Pendente: completar com a análise de erros do experimento (onde o modelo mais erra)._
