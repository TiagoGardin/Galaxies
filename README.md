# Galaxies — Agente Preditivo de Performance em Mídias Sociais

Um agente que prevê, **antes da publicação**, quantas curtidas um YouTube Short de moda ou beleza deve receber em relação ao padrão do canal, com intervalo de confiança, ordenação entre criativos e explicação dos fatores. Entregue como tool (função Python e servidor MCP) para a persona sintética da Galaxies.

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

Detalhes, filtros, esquema de colunas, termos de uso e vieses: [docs/DATA_CARD.md](docs/DATA_CARD.md)

## 3. Arquitetura

O número sempre vem do modelo treinado. O LLM orquestra, interpreta e explica; nunca chuta a previsão.

```
criativo (URL/arquivo) + legenda + data planejada + canal + setor
        │
        ▼
 Agente orquestrador (LangGraph)
   ├─ Tool 1  extrair_atributos   → texto, thumbnail/frames, áudio (offline) + rubrica VLM
   ├─ Tool 2  historico_canal     → features do canal + criativos comparáveis do histórico
   ├─ Tool 3  prever              → LightGBM: previsão + intervalo de 80% + faixa
   ├─ Tool 4  explicar            → SHAP por bloco + subagente que redige a explicação
   └─ Tool 5  ordenar             → ranking entre vários criativos / what-if
        │
        ▼
 Saída validada com Pydantic → função Python e servidor MCP (FastMCP)
```

**Contrato da tool** (esquema agnóstico à base, conforme o TAPI):

| Entrada | Saída |
|---|---|
| `creative`: URL ou caminho do vídeo/imagem | `predicted_metric`: curtidas previstas |
| `caption`: legenda planejada | `interval_low` / `interval_high`: intervalo de 80% |
| `planned_datetime`: data e hora previstas | `confidence_flag`: baixa confiança |
| `channel_id` ou `channel_stats`: canal e histórico | `relative_to_channel`: desempenho relativo ao canal e faixa |
| `sector`: `moda` ou `beleza` | `top_drivers`: atributos que mais pesaram e sentido do efeito |
| | `explanation`: justificativa em linguagem natural |
| | `comparables`: criativos do histórico usados como evidência |

## 4. Instalação e execução

**Pré-requisitos:** [uv](https://docs.astral.sh/uv/) (instala o Python 3.12 do projeto), chave da YouTube Data API v3, chave do Gemini (Google AI Studio) e, para a camada multimodal, `ffmpeg` e [Deno](https://deno.com/) (exigido pelo yt-dlp atual).

```bash
uv sync                      # cria o ambiente e instala as dependências
cp .env.example .env         # preencha YOUTUBE_API_KEY e GEMINI_API_KEY
```

**Etapas já implementadas** (rodar da raiz do repositório):

```bash
# 1. Seleção de canais (já executada; resultado revisado em data/canais.csv)
uv run python -m galaxies_tool.descobrir_canais   # descoberta via busca (~2.400 unidades de cota)
uv run python -m galaxies_tool.verificar_canais   # verificação contra os critérios do data card

# 2. Base de dados
uv run python -m galaxies_tool.coletar            # coleta dos canais incluídos → data/raw/coleta=AAAA-MM-DD/
uv run python -m galaxies_tool.processar          # filtros + validação → data/processed/base_v1.parquet
uv run python -m galaxies_tool.splits             # cortes do experimento → data/processed/splits_v1.json

# Testes
uv run pytest
```

A coleta guarda um cache por canal em `data/cache/` e pode ser retomada se for interrompida (por exemplo, se a cota diária acabar).

**Etapas em construção** (os comandos serão documentados aqui conforme forem implementados): extração de features, treino com registro no MLflow, agente e servidor MCP.

## 5. Modelos e configurações

| Componente | Escolha | Por quê |
|---|---|---|
| Modelo preditivo | LightGBM (L1) + dois modelos de quantil (10% e 90%) | Bom em dados tabulares heterogêneos, trata ausentes, tem quantil nativo para o intervalo |
| Calibração do intervalo | Conformal (MAPIE), se necessário | Garante cobertura próxima de 80% |
| Hiperparâmetros | Optuna na validação | Busca eficiente sem tocar no teste |
| Embedding de texto | `paraphrase-multilingual-mpnet-base-v2` + PCA 16 | Funciona bem em português |
| Embedding visual | SigLIP 2 (thumbnail e gancho) + PCA | Encoder aberto, recomendado no TAPI |
| Transcrição | faster-whisper (local) | Gratuito e offline |
| Detectores | MediaPipe (rosto), EasyOCR (texto), PySceneDetect (cortes), OpenCV (cor e movimento) | Gratuitos, rodam localmente |
| VLM | Gemini Flash-Lite (free tier), fallback Qwen-VL local | Aceita URL do YouTube diretamente, dispensando download para essa camada |
| Explicabilidade | SHAP (`TreeExplainer`) | Atribuição por feature, agregada por bloco |
| Agente | LangGraph | Orquestração com tools e subagente |
| Integração | Função Python + FastMCP | Consumo como tool pela Galaxies |
| Experimentos | MLflow + pytest | Registro reprodutível e testes de consistência número do modelo = número do agente |

Detalhes: [docs/MODEL_CARD.md](docs/MODEL_CARD.md) · [docs/FEATURES.md](docs/FEATURES.md)

## 6. Metodologia

- **Alvo:** `log1p(curtidas) − log1p(mediana dos últimos (até) 20 Shorts do canal publicados ≥ 30 dias antes)`; o baseline (mediana do canal) equivale a prever zero.
- **Features em 4 blocos**, medidos por ablação: A (canal, tempo, metadados), B (texto), C (visual e áudio offline), D (VLM).
- **Validação:** split temporal treino/validação/teste (~70/15/15) como principal, mais um split por canal (20% dos canais nunca vistos). Sem validação cruzada aleatória. O teste é avaliado uma única vez.
- **Métricas:** MAE-log, MAPE, Spearman global e intra-canal, acurácia par-a-par intra-canal (com todos os pares e só com pares publicados a até 7 dias, que é a decisão real entre criativos), cobertura do intervalo de 80% e F1 das faixas, todas também por setor e por tipo de canal.
- **Anti-vazamento:** só informação existente antes da publicação; histórico do canal só com Shorts maduros (≥ 30 dias antes); inscritos do dia da coleta nunca são feature; transformadores ajustados só no treino; `channel_id` nunca é feature; o VLM não recebe título, canal nem métricas. A lista completa e as features descartadas (com o motivo) estão em [docs/FEATURES.md](docs/FEATURES.md).

Hipóteses registradas antes do treino e protocolo completo: [docs/EXPERIMENTO.md](docs/EXPERIMENTO.md)

## 7. Resultados

_Pendente: preencher após o treino final com a tabela baseline × modelo, a ablação por bloco e o resultado em canais não vistos._

| Modelo | MAE-log | Spearman intra-canal | Acurácia par-a-par | Cobertura 80% |
|---|---|---|---|---|
| Baseline (mediana do canal) | _pendente_ | — | 50% | — |
| LightGBM A+B+C+D | _pendente_ | _pendente_ | _pendente_ | _pendente_ |

## 8. Limitações

- **Marcas são minoria** na base (52 de 260 canais); o desempenho em canais de marca, o perfil de cliente da Galaxies, é menos garantido e é reportado à parte.
- **Métricas no dia da coleta:** curtidas acumuladas até a coleta, inclusive as do histórico; inscritos do dia da coleta, por isso fora do modelo.
- **Vieses da base:** canais escolhidos por relevância atual (mais canais em ascensão); vídeos apagados não aparecem; textos e thumbnails podem ter sido editados depois da publicação.
- **Fatores não observados:** mídia paga, distribuição do algoritmo, trends e collabs.
- **Recorte:** só YouTube Shorts, moda e beleza, Brasil. Outras plataformas, setores ou países exigem retreino.
- **Previsão absoluta é imprecisa por natureza:** o modelo é para comparar criativos, não para prometer números.

## 9. Evolução

- **Retreino com a base de um cliente:** o pipeline segue o esquema mínimo do TAPI (`post_id`, `channel_id`, `published_at`, `caption`, `media_ref`, `metric_target`, `channel_stats_at_collection`, `sector`), o que permite treinar um modelo especialista quando um cliente liberar sua base de ads.
- **Coletas periódicas** para medir curtidas num horizonte fixo (ex.: 30 dias após a publicação) e acompanhar a deriva da plataforma.
- **Mais marcas e o setor automotivo**, para chegar mais perto do perfil de cliente da Galaxies.
- **Extensões do TAPI:** RAG sobre criativos comparáveis de alto desempenho, loop de otimização what-if (legenda, horário, gancho), embeddings de vídeo completo (X-CLIP/VideoMAE) e comparação LLM zero-shot × modelo treinado.
