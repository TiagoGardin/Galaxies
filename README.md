# Galaxies — Agente Preditivo de Performance em Mídias Sociais

Um agente que prevê, **antes da publicação**, quantas curtidas um YouTube Short de moda ou beleza deve receber em relação ao padrão do canal, com intervalo de 80%, ordenação entre criativos e explicação dos fatores. Entregue como tool (função Python e servidor MCP) para a persona sintética da Galaxies.

## 1. Problema e recorte

Hoje, quando um time de marketing envia criativos para a Galaxies, as personas sintéticas assistem e dão uma opinião qualitativa (nota de 0 a 5). Não há um componente estatístico nem um número real para comparar depois. Este projeto entrega essa peça: um modelo treinado em dados observados, cujo erro pode ser medido, embrulhado num agente que a persona consulta como ferramenta.

**Pergunta norteadora:** é possível prever, antes da publicação e a partir do conteúdo do criativo e do histórico do canal, como um público real vai reagir a ele, e com que margem de erro?

| Recorte | Escolha | Por quê |
|---|---|---|
| Plataforma | YouTube Shorts | API oficial, gratuita e legalmente segura (TikTok não tem API gratuita) |
| Setores | Moda e beleza | Clientes mais ativos da Galaxies; setores próximos, com criadores e marcas em comum |
| Geografia | Canais brasileiros (pt-BR) | Público da Galaxies; o filtro é por canal, então vídeos com música em outro idioma entram |
| Métrica | Curtidas, **relativas ao canal** | Curtidas medem reação do público; a forma de contar views mudou duas vezes (03/2025 e 08/2026). O alvo relativo isola o efeito do conteúdo, que é o que decide entre criativos de uma mesma marca |
| Objetivo | Ordenar criativos > acertar o valor absoluto | O valor absoluto é mal previsível; a Galaxies considera ~60% de acerto na ordenação uma prova de conceito |

## 2. Base de dados

- **Origem:** YouTube Data API v3, coleta própria (o caminho `channels` → playlist de uploads → `videos`, sem `search.list` para listar vídeos).
- **Canais:** 260 canais brasileiros (169 de beleza e 91 de moda; 208 criadores e 52 marcas), selecionados a partir de sementes e de uma descoberta por busca, com verificação automática e revisão manual de cada canal, inclusive das thumbnails, para excluir repostagens, compilações e conteúdo gerado por IA.
- **Volume (`base_v1`, coleta de 12/10/2026):** 79.402 Shorts, dos quais **40.865 são vídeos-alvo elegíveis** (01/04/2025 a 12/09/2026) em 258 canais e o restante serve de histórico. Split temporal: 22.430 treino, 4.774 validação, 4.861 teste, mais 1.146 vídeos de 51 canais nunca vistos.
- **Teste prospectivo (coleta de 21/10/2026):** 1.114 Shorts publicados entre 12/09 e 21/09/2026, depois de todo o treino, com as curtidas de 21/10 (`data/prospectivo/`).
- **Licença e termos:** dados públicos coletados dentro dos [Termos de Serviço da YouTube API](https://developers.google.com/youtube/terms/api-services-terms-of-service); a mídia (vídeos, áudios, thumbnails) nunca é redistribuída. Uso acadêmico; ver "Termos de uso e ética" no data card.

Detalhes, filtros, esquema de colunas, termos de uso e vieses: [docs/DATA_CARD.md](docs/DATA_CARD.md)

## 3. Arquitetura

O número sempre vem do modelo treinado. O LLM só redige a explicação a partir de um JSON fechado de fatos, e um validador confere que ele não inventou números nem fatores.

```
criativo (URL do YouTube, vídeo/imagem local ou texto) + legenda + data planejada + canal + setor
        │
        ▼
 Agente orquestrador (LangGraph, grafo fixo)                                  galaxies_tool/agente.py
   ├─ Tool 1  extrair_atributos → texto (regras + embedding), thumbnail (cor, rosto, OCR, SigLIP 2),
   │                              vídeo em 360p (cortes, movimento, Whisper, energia, gancho SigLIP 2)
   │                              e rubrica do VLM (Gemini)                     extracao.py, video.py, vlm.py
   ├─ Tool 2  historico_canal   → bloco A (histórico só com Shorts ≥ 30 dias antes) + comparáveis, cujo
   │                              desempenho real vira feature na v2 (RAG)        canal.py, rag.py
   ├─ Tool 3  prever            → LightGBM + CatBoost: previsão + intervalo de 80% (quantis + CQR) + faixa
   │                              + confidence_flag com motivos codificados       previsao.py, artefato.py
   ├─ Tool 4  explicar          → SHAP por fator e por bloco + subagente explicador com validador
   └─ Tool 5  ordenar           → ranking de vários criativos com P(A > B) e what-if de horário e de legenda
        │
        ▼
 Saída validada com Pydantic (contrato.py) → função Python e servidor MCP (servidor_mcp.py)

 Treino dinâmico (treino_cliente.py): base de um cliente no esquema mínimo do TAPI → modelo especialista,
 avaliado com o mesmo protocolo e servido pelo mesmo agente ("modelo": "<nome>")
```

- **Extração com as mesmas funções do treino:** a Tool 1 chama exatamente o código que gerou as features da base, inclusive o download em 360p; um teste confere que, para vídeos da validação, o agente reproduz as features e a previsão do modelo avaliado. A extração com redes neurais (PyTorch) roda num subprocesso persistente, porque o PyTorch e o LightGBM no mesmo processo derrubam o Python no macOS (dois OpenMP).
- **Explicador ancorado (aula 5.7):** o LLM recebe só os fatos calculados (previsão, intervalo, fatores do SHAP, motivos de baixa confiança, comparáveis) e devolve uma explicação estruturada. Um validador determinístico confere que todo número citado está no JSON e todo fator citado está entre os do modelo; se reprovar, o LLM recebe os problemas e tenta uma vez mais; se reprovar de novo, ou se não houver chave de API, a resposta usa um texto padrão montado só com os fatos. O campo `atributos_extraidos.explicacao_fonte` diz qual foi usado (`llm`, `llm_corrigido` ou `padrao`), e `explicacao_reprovacoes` registra o que o validador reprovou.

**Contrato da tool** (esquema agnóstico à base, conforme o TAPI; `galaxies_tool/contrato.py`):

| Entrada | Saída |
|---|---|
| `creative`: URL do YouTube, caminho de vídeo/imagem ou texto | `predicted_metric`: curtidas previstas |
| `caption` (+ `description`, `tags`): legenda planejada | `interval_low` / `interval_high`: intervalo de 80% calibrado |
| `planned_datetime`: data e hora previstas (sem fuso = Brasília) | `confidence_flag` + `confidence_reasons` (código e frase) |
| `channel_id` (canal da base) ou `channel_stats` (histórico do cliente) | `relative_to_channel`: mediana do canal, razão prevista e faixa |
| `sector`: `moda`, `beleza` ou `automotivo` (este último sinalizado) | `top_drivers`: fatores que mais pesaram, valor e efeito em % |
| `analisar_video`: baixa o vídeo e chama o VLM (padrão: sim) | `explanation`: justificativa em linguagem natural |
| `id`: rótulo do criativo no ranking | `comparables`: Shorts parecidos do histórico, com o desempenho real |
| `modelo`: modelo especialista de um cliente (vazio = principal) | `contribuicao_por_bloco`, `atributos_extraidos`, `modelo_versao` |

Tools do servidor MCP: `prever_criativo`, `ordenar_criativos` (ranking com a chance de cada criativo superar o seguinte), `what_if_horarios`, `what_if_legendas` e `treinar_modelo_cliente` (treino dinâmico com a base de um cliente); recurso `modelo://card` com a versão, a calibração e as métricas do modelo servido.

**Estrutura do repositório**

```
src/galaxies_tool/     código (um módulo por etapa; cada um tem "Uso:" na docstring)
docs/                  DATA_CARD, FEATURES, MODEL_CARD, EXPERIMENTO e TAPI_CONFORMIDADE
data/                  canais (CSVs), coleta bruta (raw/), base processada (processed/), features (features/),
                       teste prospectivo (prospectivo/), base de exemplo do treino dinâmico (exemplos/) e rótulos
models/                artefatos versionados: lgbm_v2 (servido) e lgbm_final (v1, a do teste temporal)
resultados/            tabelas de validação, teste, backtest, prospectivo, demonstração e rótulos
configs/lgbm.yaml      hiperparâmetros escolhidos pelo Optuna
tests/                 pytest (o marcador slow carrega os modelos de extração)
.streamlit/            tema da interface opcional
```

| Etapa | Módulos |
|---|---|
| Base de dados | `youtube_api` (cliente da API com contagem de cota), `descobrir_canais`, `verificar_canais`, `coletar`, `regras` (regras do data card), `processar`, `splits` |
| Features | `features` (blocos A e B), `embeddings_texto`, `thumbs` (thumbnail e gancho), `video`, `amostra_vlm`, `vlm`, `multimodal` (vídeo e VLM → colunas), `rag` (comparáveis, v2) |
| Modelo e experimento | `modelo` (blocos, treino, ablação), `metricas`, `ajustar` (Optuna), `comparar_modelos`, `artefato`, `avaliacao_final`, `backtest`, `por_setor`, `rotulos` |
| Teste prospectivo | `prospectivo` (coleta e extração), `avaliacao_prospectiva` (avaliação única) |
| Agente e integração | `contrato` (Pydantic), `extracao` (Tool 1), `canal` (Tool 2), `previsao` (Tools 3 e 4), `agente` (LangGraph), `servidor_mcp`, `api` (FastAPI), `app` (Streamlit), `demo`, `treino_cliente` (treino dinâmico) |

## 4. Instalação e execução

**Pré-requisitos:** [uv](https://docs.astral.sh/uv/) (instala o Python 3.12 e as versões exatas de `uv.lock`), `ffmpeg` e [Deno](https://deno.com/) (exigido pelo yt-dlp atual) para a parte de vídeo, uma chave da YouTube Data API v3 e uma chave do Gemini (Google AI Studio).

```bash
uv sync                      # cria o ambiente com as versões fixadas em uv.lock
cp .env.example .env         # preencha YOUTUBE_API_KEY e GEMINI_API_KEY
uv run pytest -m "not slow"  # testes rápidos (o marcador slow carrega os modelos de extração, ~1 min)
```

**O que já vem no repositório:** a lista de canais revisada (`data/canais.csv`), a coleta bruta e a base processada (`data/raw/`, `data/processed/`), as features tabulares (`data/features/*.parquet`), as medidas de vídeo e as rubricas do VLM por vídeo (`data/features/video/`, `data/features/vlm/`) e as filas da amostra. Também vêm os artefatos do modelo (`models/lgbm_v2.joblib`, o servido, e `models/lgbm_final.joblib`, a v1 avaliada no teste temporal), para o agente e o servidor MCP rodarem sem retreinar, o embedding do gancho (`data/features/emb_gancho.npy`, que exigiria baixar os vídeos de novo), a coleta e a tabela de atributos do teste prospectivo (`data/prospectivo/`), a base de exemplo do treino dinâmico (`data/exemplos/`) e as tabelas de resultado (`resultados/`). **O que não vem** (grande ou regenerável, ver `.gitignore`): os embeddings brutos de texto e thumbnail (`emb_texto.npy` e `emb_thumb.npy`, 60 MB cada; regenere com `embeddings_texto` e `thumbs siglip`). **Sem eles, o agente usa a v1** (a v2 precisa deles para as features de comparáveis), avisa isso em `atributos_extraidos.aviso_modelo` e no `modelo://card`, e não mostra comparáveis. Também não vêm a mídia baixada (`data/media/`) e o registro do MLflow (`mlruns/`, `mlflow.db`). Para reproduzir do zero, rode as etapas abaixo em ordem; cada uma tem cache e pode ser retomada.

```bash
# 1. Base de dados (já executada; ver docs/DATA_CARD.md)
uv run python -m galaxies_tool.descobrir_canais      # descoberta de canais via busca (~2.400 unidades de cota)
uv run python -m galaxies_tool.verificar_canais      # critérios automáticos → revisão manual em data/canais.csv
uv run python -m galaxies_tool.coletar               # coleta dos canais → data/raw/coleta=AAAA-MM-DD/
uv run python -m galaxies_tool.processar             # filtros + validação (pandera) → data/processed/base_v1.parquet
uv run python -m galaxies_tool.splits                # cortes do experimento → data/processed/splits_v1.json

# 2. Features
uv run python -m galaxies_tool.features              # blocos A e B (sem embedding) → features_ab.parquet
uv run python -m galaxies_tool.embeddings_texto      # embedding do texto (bloco B)
uv run python -m galaxies_tool.thumbs baixar         # thumbnails (bloco C, base inteira)
uv run python -m galaxies_tool.thumbs estatisticas   # cor e rosto
uv run python -m galaxies_tool.thumbs ocr            # texto na thumbnail
uv run python -m galaxies_tool.thumbs siglip         # embedding SigLIP 2 da thumbnail
uv run python -m galaxies_tool.amostra_vlm           # filas da amostra dos blocos C (vídeo) e D
uv run python -m galaxies_tool.video                 # vídeo em 360p: cortes, movimento, Whisper, energia, gancho
uv run python -m galaxies_tool.thumbs siglip_gancho  # embedding SigLIP 2 dos quadros do gancho
uv run python -m galaxies_tool.vlm                   # rubrica do VLM (~450 vídeos por dia no free tier)

# 3. Modelo (validação; o teste fica guardado)
uv run python -m galaxies_tool.ajustar               # Optuna → configs/lgbm.yaml
uv run python -m galaxies_tool.modelo                # baselines, ablação e "todos menos um", com IC → resultados/
uv run python -m galaxies_tool.comparar_modelos      # LightGBM × CatBoost × XGBoost × Ridge (+ média)
uv run python -m galaxies_tool.artefato              # artefato v1: models/lgbm_final.joblib (+ .json)
uv run python -m galaxies_tool.avaliacao_final       # avaliação ÚNICA no teste → resultados/teste_*
uv run python -m galaxies_tool.backtest              # backtest com origem móvel (treino + validação)
uv run python -m galaxies_tool.por_setor             # um modelo por setor × global (só validação)
uv run python -m galaxies_tool.artefato --versao lgbm_v2 --variante v2   # v2: PCA 32 + comparáveis (RAG)
uv run python -m galaxies_tool.prospectivo coletar   # teste prospectivo: Shorts publicados depois da base
uv run python -m galaxies_tool.prospectivo extrair [--video]             # Tool 1 do agente em cada vídeo
uv run python -m galaxies_tool.prospectivo avaliar --parte principal     # avaliação ÚNICA de v1 e v2
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001   # registro dos experimentos

# 4. Agente
uv run python -m galaxies_tool.demo                  # demonstração: canais fora do treino, previsto × real
uv run python -m galaxies_tool.demo --salvas         # plano B: reimprime as respostas salvas, sem rede
uv run --group app streamlit run src/galaxies_tool/app.py              # interface opcional (ao vivo ou demo salva)
uv run python -m galaxies_tool.servidor_mcp          # servidor MCP (stdio); --http para HTTP em 127.0.0.1:8000
uv run --group api uvicorn galaxies_tool.api:app --port 8001           # API REST opcional (docs em /docs)

# 5. Treino dinâmico (base de um cliente no esquema mínimo do TAPI)
uv run python -m galaxies_tool.treino_cliente exemplo                    # base de exemplo: canais de marca
uv run python -m galaxies_tool.treino_cliente treinar data/exemplos/base_cliente_marcas.parquet --nome marcas
```

**Como função Python:**

```python
from galaxies_tool.agente import prever_criativo, ordenar_criativos, what_if_legendas
entrada = {"creative": "https://www.youtube.com/shorts/<id>", "caption": "3 makes em 30 s",
           "planned_datetime": "2026-10-20T18:00", "channel_id": "UC...", "sector": "beleza"}
r = prever_criativo(entrada)
print(r.predicted_metric, r.interval_low, r.interval_high, r.confidence_flag, r.explanation)
what_if_legendas(entrada, ["Tutorial: make em 60 s", "Resenha sincera: vale a pena?"])  # mesma capa e vídeo
```

**Interface opcional (Streamlit):** mostra cada tool do grafo terminando em tempo real, o ranking com os intervalos e a mediana do canal, os fatores do SHAP (azul aumenta, vermelho reduz), os Shorts comparáveis com miniatura, o vídeo embutido e a inspeção dos atributos, em tema claro ou escuro; o modo "Demo salva" abre a demonstração sem rede. É uma vitrine sobre a mesma função Python, não um produto (o consumo previsto é pela tool).

**API REST opcional (FastAPI):** os mesmos endpoints da tool, com o mesmo contrato Pydantic: `POST /predict`, `POST /rank`, `POST /what-if/horarios`, `POST /what-if/legendas`, `POST /train` (treino dinâmico), `GET /modelo` (model card) e `GET /saude`. Entrada inválida volta como 422; erro de domínio (ex.: canal fora da base sem `channel_stats`), como 400.

**Testado com o [MCP Inspector](https://github.com/modelcontextprotocol/inspector)** (modo de linha de comando; lista as tools, lê o `modelo://card` e chama `prever_criativo`, devolvendo os mesmos números da função Python). O teste `tests/test_mcp.py` faz o mesmo com o cliente do SDK oficial.

```bash
npx @modelcontextprotocol/inspector@0.17.2 --cli uv run --directory "$PWD" python -m galaxies_tool.servidor_mcp --method tools/list
```

**Num cliente MCP** (Claude Desktop, Claude Code ou o agente da Galaxies), com caminho absoluto:

```json
{"mcpServers": {"galaxies-preditor": {"command": "uv", "args": ["run", "--directory",
  "/caminho/para/o/repositorio", "python", "-m", "galaxies_tool.servidor_mcp"]}}}
```

## 5. Modelos e configurações

| Componente | Escolha | Por quê |
|---|---|---|
| Modelo preditivo (servido: **v2**) | Média de LightGBM 4.7 e CatBoost 1.2 (`L1`) + quantis 10% e 90% do LightGBM, com as features de comparáveis (RAG) | Bons em dados tabulares heterogêneos e com ausentes; a média ganhou na ordenação na validação. A v2 (PCA 32 + comparáveis) foi escolhida na validação depois do teste e confirmada no teste prospectivo pela regra pré-registrada; a v1 (`lgbm_final`, PCA 16, sem comparáveis) é a avaliada no teste temporal |
| Hiperparâmetros | Optuna 5.0 (TPE, 40 tentativas) na validação, early stopping temporal | Em [`configs/lgbm.yaml`](configs/lgbm.yaml) |
| Intervalo | Quantis + conformal sobre quantis (CQR) calibrado na validação | Cobertura de 80% medida, não declarada |
| Treino final | Reajuste em treino + validação com a calibração medida fora da amostra | Não perde os meses mais recentes |
| Embedding de texto | `paraphrase-multilingual-mpnet-base-v2` (sentence-transformers 6.1) + PCA 32 (v1: 16) | Funciona bem em português |
| Embedding visual | `google/siglip2-base-patch16-224` (transformers 5.19): thumbnail (PCA 32; v1: 16) e gancho (PCA 8) | Encoder aberto, recomendado no TAPI |
| Comparáveis (RAG) | Busca exata por cosseno (numpy) na média dos embeddings de texto e thumbnail, só entre Shorts com 30 dias antes | 40 mil vetores dispensam índice aproximado (FAISS) |
| Transcrição | faster-whisper 1.2 (`small`, int8, VAD) | Gratuito e local |
| Detectores | MediaPipe 1.1 (rosto), EasyOCR 1.7 (texto), PySceneDetect 0.7 (cortes), OpenCV (cor e movimento) | Gratuitos, locais |
| VLM | `gemini-3.1-flash-lite` (google-genai 2.30), temperatura 0, saída estruturada | Aceita a URL do YouTube; o 3.5 vivia sobrecarregado |
| Explicabilidade | TreeSHAP exato dos dois modelos (média) + subagente explicador (`gemini-3.5-flash`, reserva `gemini-3.1-flash-lite`) com validador | Atribuição exata e explicação conferível |
| Agente | LangGraph 1.2 (grafo fixo) | Extração, contexto e previsão sempre na mesma ordem |
| Integração | Função Python + servidor MCP (SDK oficial `mcp` 2.3) + API REST opcional (FastAPI) | Consumo como tool pela Galaxies; a API atende quem não usa MCP |
| Treino dinâmico | pandera (esquema mínimo) + o mesmo pipeline de features, artefato e métricas | Um modelo especialista por cliente, sem código novo |
| Experimentos | MLflow 3.17 (versão e hash da base, cortes, parâmetros, métricas, SHAP) + pytest | Registro reprodutível |

Detalhes: [docs/MODEL_CARD.md](docs/MODEL_CARD.md) · [docs/FEATURES.md](docs/FEATURES.md) · [docs/TAPI_CONFORMIDADE.md](docs/TAPI_CONFORMIDADE.md) (cada exigência do TAPI e onde ela está)

## 6. Metodologia

- **Alvo:** `log1p(curtidas) − log1p(mediana dos últimos (até) 20 Shorts do canal publicados ≥ 30 dias antes)`; o baseline (mediana do canal) equivale a prever zero.
- **Features em 4 blocos**, medidos por ablação: A (canal, tempo, metadados), B (texto), C (thumbnail na base inteira; vídeo e gancho numa amostra), D (rubrica do VLM, na amostra); a v2 acrescenta o bloco R (desempenho de Shorts parecidos publicados pelo menos 30 dias antes).
- **Validação:** split temporal treino/validação/teste (~70/15/15, cortes em 23/04/2026 e 02/07/2026) como principal, mais um teste em 51 canais nunca vistos (20% dos canais). Sem validação cruzada aleatória. Hiperparâmetros, escolha de blocos e calibração usam só a validação; o teste é avaliado uma única vez (`avaliacao_final.py` se recusa a rodar duas vezes). Depois do teste: um **backtest com origem móvel** (4 origens, lacuna de 30 dias) e um **teste prospectivo** com Shorts publicados depois de toda a base, pré-registrado e avaliado uma vez, onde a v2 foi julgada.
- **Métricas:** MAE-log, MAPE, Spearman global e intra-canal, acurácia par-a-par intra-canal (com todos os pares e só com pares publicados a até 7 dias, que é a decisão real entre criativos), cobertura do intervalo de 80% e F1 das faixas, também por setor, tipo de canal e porte.
- **Ganho de cada bloco:** escada cumulativa A → A+B → A+B+C → A+B+C+D, ablação "todos menos um" e um controle negativo (colunas de ruído no lugar do VLM), todos com os mesmos hiperparâmetros e IC 95% por bootstrap pareado que reamostra canais. Os blocos que só existem na amostra são comparados nos vídeos da amostra.
- **Incerteza:** quantis do LightGBM calibrados por CQR; `confidence_flag` com sete regras codificadas (histórico curto, canal instável, intervalo largo, criativo fora da distribuição, setor fora do treino, extração incompleta, data fora do período), avaliado no teste pelo erro dos casos sinalizados.
- **Anti-vazamento:** só informação existente antes da publicação; histórico do canal só com Shorts maduros (≥ 30 dias antes); inscritos do dia da coleta nunca são feature; PCAs ajustados só no treino; `channel_id` nunca é feature; o VLM recebe só o vídeo (sem título, canal ou métricas). Lista completa e features descartadas em [docs/FEATURES.md](docs/FEATURES.md).

Hipóteses registradas antes do treino e protocolo completo: [docs/EXPERIMENTO.md](docs/EXPERIMENTO.md)

## 7. Resultados

**Teste temporal** (4.861 vídeos de 02/07 a 12/09/2026, avaliado uma única vez com a v1; `resultados/teste_*`):

| Modelo | MAE-log | Spearman (curtidas absolutas) | Spearman intra-canal | Par-a-par | Par-a-par mesma época | Cobertura 80% |
|---|---|---|---|---|---|---|
| Baseline: mediana do canal | 0,835 | 0,858 | indefinido | 50,0% | 50,0% | — |
| Baseline: regressão linear | 0,842 | 0,858 | 0,259 | 60,6% | 53,4% | — |
| **Modelo final (LightGBM + CatBoost, A+B+C+D)** | **0,760** | **0,882** | **0,357** | **61,2%** | **59,1%** | **82,7%** |

- **Bate o baseline** com IC 95% acima de zero (bootstrap de canais): MAE-log −0,076 [−0,097; −0,055] e +9,1 p.p. [+7,3; +10,9] no acerto de pares de criativos do mesmo canal publicados na mesma semana. O erro típico é um fator de ~2,1× nas curtidas; o intervalo de 80% cobriu 82,7% (81,8% em beleza, 84,8% em moda).
- **Ablação (camada multimodal):**

  | Degrau | Spearman | MAE-log | Ganho (IC 95%) |
  |---|---|---|---|
  | A (canal, tempo, metadados) | 0,249 | 0,806 | — |
  | + B (texto) | 0,365 | 0,773 | erro e ordenação, IC acima de zero |
  | + C thumbnail | 0,406 | 0,761 | MAE −0,012 [−0,021; −0,003] |
  | + C vídeo e gancho | 0,392 | 0,764 | sem ganho |
  | + D VLM | 0,405 | 0,760 | na amostra com rubrica: MAE −0,009 [−0,014; −0,003], par-a-par mesma época +2,3 p.p. [+0,4; +4,4]; controle de ruído sem ganho |

- **Canais nunca vistos** (1.146 vídeos de 51 canais): o modelo empata com a mediana do canal no erro (0,843 × 0,830) e ordena pouco acima do acaso (52,3% na mesma época).
- **Onde erra:** virais (cobertura do intervalo de 43%), conteúdo patrocinado (14 dos 20 maiores erros são publis: com link de compra, explodem muito acima do canal; de criadores grandes, às vezes são ignoradas), mídia paga; canais pequenos e instáveis. O `confidence_flag` ligou em 16% dos vídeos, que tiveram MAE-log de 1,10 contra 0,70 dos demais.
- **LLM zero-shot × modelo:** a nota de "potencial viral" do VLM acerta 52,4% dos pares da mesma época, contra 60,2% do modelo nos mesmos vídeos.
- **Hipóteses:** H1, H2 e H3 confirmadas; H4 confirmada para canais novos e parcial para marcas; H5 (binárias do VLM > notas) não confirmada. Detalhes, recortes e análise de erros em [docs/EXPERIMENTO.md](docs/EXPERIMENTO.md).

**Backtest com origem móvel** (4 origens de 11/2025 a 05/2026, cada uma avaliada nos 2 meses seguintes, com lacuna de 30 dias; `resultados/backtest*.csv`): o modelo bate a mediana do canal em todas, com IC acima de zero, e com ganhos parecidos (MAE-log −0,036 a −0,042; par-a-par na mesma época +7,4 a +8,4 p.p.). O resultado não depende de um período favorável.

**Teste prospectivo** (Shorts publicados de 12/09 a 21/09/2026, depois de toda a base, curtidas coletadas em 21/10; protocolo pré-registrado; avaliado uma vez; `resultados/prospectivo_*`). Parte principal, sem vídeo nem VLM, 852 vídeos de canais vistos:

| Modelo | MAE-log | Par-a-par mesma época | Cobertura 80% |
|---|---|---|---|
| Mediana do canal | 0,786 | 50,0% | — |
| v1 (a do teste temporal) | 0,772 | 59,9% | 80,2% |
| **v2 (servida)** | **0,745** | **61,7%** | 81,9% |

- A v1 repete a ordenação do teste (59,9% × 59,1%), mas o ganho no erro sobre a mediana fica dentro do ruído (−0,014 [−0,041; +0,012]), em parte por um viés previsto: essas curtidas têm só 30 a 39 dias.
- A v2 erra menos que a v1 (−0,027 [−0,046; −0,008]) e ordena um pouco melhor (+1,8 p.p. [−0,4; +4,0]); pela regra registrada antes de abrir os dados, passou a ser o modelo servido. Nos 262 vídeos de canais novos, também erra menos (0,845 × 0,899).
- Parte complementar, com vídeo e VLM (586 vídeos extraídos até 09h00 de 21/10): as duas versões ordenam melhor que a mediana (v1 57,4%, v2 58,1% na mesma época, IC acima de zero), e a v2 segue errando menos (−0,019 [−0,041; +0,003]). Nos 132 vídeos de canais novos dessa amostra, a v2 ordenou pior que a v1 (IC largo): ponto a acompanhar.

**Demonstração** (`python -m galaxies_tool.demo`; 12 Shorts de 4 canais fora do treino, escolhidos por regra fixa): com a v2, o valor real caiu dentro do intervalo de 80% em 12 de 12, e a ordem prevista dentro de cada canal teve Spearman de +1,0, +0,5, +0,5 e −1,0 (este na Eudora, cujos 3 vídeos têm de 20 a 64 curtidas). Com a v1, também 12 de 12, e Spearman de +0,5 a +1,0. É uma demonstração, não uma avaliação; as respostas completas ficam em `resultados/demo/`.

## 8. Limitações

- **Marcas são minoria** na base (52 de 260 canais); o desempenho em canais de marca, o perfil de cliente da Galaxies, é menos garantido e é reportado à parte.
- **Vídeo e VLM numa amostra:** a parte de vídeo do bloco C cobre 8.219 vídeos e o VLM 2.905 (cota gratuita do Gemini; 1.086 no teste), contra 32 mil vídeos-alvo. O modelo aprende esses blocos com poucos exemplos, e o ganho deles é medido com intervalos largos. Os canais novos do teste ficaram fora da amostra, então os blocos C (vídeo) e D não são avaliados neles.
- **Métricas no dia da coleta:** curtidas acumuladas até a coleta, inclusive as do histórico; inscritos do dia da coleta, por isso fora do modelo.
- **Vieses da base:** canais escolhidos por relevância atual (mais canais em ascensão); vídeos apagados não aparecem; textos e thumbnails podem ter sido editados depois da publicação.
- **Fatores não observados:** mídia paga, distribuição do algoritmo, trends e collabs.
- **Recorte:** só YouTube Shorts, moda e beleza, Brasil. Outras plataformas, setores (inclusive automotivo, que a tool aceita, mas sinaliza) ou países exigem retreino.
- **Previsão absoluta é imprecisa por natureza:** o modelo é para comparar criativos, não para prometer números. A chance de um criativo superar outro (`ordenar_criativos`) supõe erros independentes, o que a torna conservadora.
- **Validação com um único corte e sem lacuna:** hiperparâmetros e blocos foram escolhidos num só período de validação (abr–jul/2026), sem lacuna de 30 dias entre treino, validação e teste. Três análises posteriores mostram que isso não inflou os resultados: nos vídeos do teste publicados a partir de 01/08 (lacuna de 30 dias ou mais), o desempenho é igual ou melhor; o backtest com origem móvel, com lacuna, dá o mesmo ganho em 4 períodos; e o teste prospectivo repete a ordenação.
- **A v2 foi escolhida depois do teste temporal**, na validação, entre 13 variantes, e só foi julgada no teste prospectivo (852 vídeos em 9 dias). O ganho sobre a v1 no erro tem IC abaixo de zero; na ordenação, o IC ainda cruza zero. Para voltar à v1: `GALAXIES_MODELO=models/lgbm_final.joblib`.
- **As features de comparáveis da v2 dependem da base:** a busca usa os embeddings dos 40 mil Shorts da base (não versionados; sem eles, a tool usa a v1). Para um canal de fora informado só por `channel_stats` (datas e curtidas), os comparáveis do próprio canal não existem e só os do setor entram.
- **Viés de maturidade:** as curtidas do treino foram medidas com idades de 30 dias a 18 meses. Em vídeos recém-maduros (30 a 39 dias, teste prospectivo), o modelo superestima o nível em ~15% (mediana); a ordenação não é afetada.
- **Confidencialidade dos criativos:** no free tier, o Gemini pode usar o conteúdo enviado para treino. Para vídeos já publicados isso é aceitável; um criativo **ainda não publicado** de um cliente não deve ir para o free tier: use `analisar_video=False` (só texto e imagem, com a confiança reduzida), a camada paga do Gemini ou um VLM aberto local.
- **Criativo só em texto ou só imagem:** as colunas de vídeo e do VLM ficam ausentes, e o modelo, que viu a maioria dos vídeos de treino sem elas, ainda assim reage à ausência (por exemplo, "duração ausente" pesa na previsão). A resposta vem com `confidence_flag` ligado (`extracao_incompleta`) e deve ser lida como ordem de grandeza.
- **O explicador usa um LLM próprio** (Gemini), além do LLM da persona que chama a tool. A explicação é validada e tem um texto padrão de reserva, mas o número nunca passa pelo LLM. No free tier, o `gemini-3.5-flash` permite só 20 pedidos por dia e os modelos ficam sobrecarregados com frequência: nesses casos a resposta usa o texto padrão (`explicacao_fonte = padrao`), que é sempre fiel. Em produção, use a camada paga ou deixe a narração para o LLM da persona.

## 9. Evolução

- **Treino dinâmico com a base pública como ponto de partida:** a tool `treinar_modelo_cliente` já treina um especialista com a base de um cliente. No exemplo com os canais de marca, o especialista ordena (+6,3 p.p. na mesma época), mas sofre com a deriva no nível; o próximo passo é treinar com a base pública + a do cliente (com uma coluna de origem), em vez de só a do cliente.
- **Cobrir a base inteira com vídeo e VLM** (camada paga do Gemini ou um VLM aberto), para medir esses blocos com mais precisão e em canais novos.
- **Mais marcas e o setor automotivo**, para chegar mais perto do perfil de cliente da Galaxies.
- **Extensões do TAPI ainda não feitas:** what-if de gancho (exigiria gerar variações do vídeo), embeddings de vídeo completo (X-CLIP/VideoMAE) e de áudio (CLAP), clusterização de descritores do VLM (a rubrica não coleta descritores livres) e um VLM local de reserva (Qwen2.5-VL). Mapa completo do que o TAPI pede e onde está: [docs/TAPI_CONFORMIDADE.md](docs/TAPI_CONFORMIDADE.md).
- **Coletas periódicas e recalibração do nível:** medir as curtidas num horizonte fixo (ex.: 30 dias após a publicação), acompanhar a deriva da plataforma e recalibrar o centro da previsão com os vídeos mais recentes; o teste prospectivo mostrou um viés de maturidade que isso reduziria sem mexer na ordenação.
