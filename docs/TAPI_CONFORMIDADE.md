# Conformidade com o TAPI

Cada exigência do documento "TAPI — Inteli Academy x Galaxies: Agente Preditivo de Performance em Mídias Sociais", na ordem do documento, com o que foi feito e onde está. Legenda: **✅** feito; **◐** feito com desvio (justificado); **—** não feito (todos opcionais, com o motivo).

Resumo: todo o núcleo obrigatório (seção 4.1) e todos os entregáveis (seção 10) estão feitos. Das 6 extensões opcionais (4.2), 5 estão feitas no todo ou em parte; a que falta (embeddings de vídeo completo e de áudio) está justificada abaixo.

## 2. Objetivos específicos

| Objetivo | | Onde |
|---|---|---|
| Montar, documentar e criticar uma base coletada pela YouTube Data API v3 | ✅ | [DATA_CARD.md](DATA_CARD.md): 260 canais, 40.865 vídeos-alvo, funil de filtros, vieses e "o que a base não representa" |
| Modelo tabular sobre criativo, canal e contexto, com baseline declarado e validação temporal | ✅ | [MODEL_CARD.md](MODEL_CARD.md), [EXPERIMENTO.md](EXPERIMENTO.md) |
| Atributos multimodais (visual, áudio, texto) com ferramentas gratuitas, e o ganho deles sobre metadados e texto | ✅ | Blocos C e D em [FEATURES.md](FEATURES.md); ablação com IC no teste |
| Agente orquestrador: criativo + contexto → previsão, intervalo, ranking e justificativa | ✅ | `galaxies_tool/agente.py` (LangGraph) |
| Tool integrável (função e MCP), agnóstica à base | ✅ | Função Python e servidor MCP; `channel_stats` aceita o histórico de qualquer canal; treino dinâmico com o esquema mínimo (`treino_cliente.py`) |

## 3. Decisões de escopo

| Tema | Decisão do TAPI | | Como ficou |
|---|---|---|---|
| Caso de uso | Prever antes de publicar | ✅ | Só informação existente antes da publicação (regras anti-vazamento em FEATURES.md); teste prospectivo com vídeos posteriores a todo o treino |
| Plataforma | YouTube Shorts; SMTPD e MicroLens como referência | ◐ | YouTube Shorts pela Data API v3. SMTPD e MicroLens foram usados só como referência de literatura (expectativa de Spearman e do ganho do conteúdo, hipótese H3), não como dados: a base própria ficou pronta na primeira semana (o papel de "bootstrap" do SMTPD não foi necessário), e os dois são de outro país e de outro recorte (SMTPD é YouTube geral, MicroLens é uma plataforma chinesa), o que não ajudaria a prever Shorts brasileiros de moda e beleza. O experimento de early popularity do SMTPD está fora do escopo do caso (4.3) |
| Setores | Pelo menos dois de moda, beleza e automotivo | ✅ | Moda e beleza. Automotivo é aceito pela tool, mas sinalizado (`setor_fora_do_treino`) |
| Multimodal | Obrigatório em versão leve: thumbnail/frames + transcrição + VLM | ✅ | SigLIP 2 na thumbnail e nos quadros do gancho, faster-whisper, rubrica do Gemini |
| Acurácia | Bater o baseline e acertar a ordenação | ✅ | Teste temporal: 59,1% dos pares da mesma época (mediana do canal: 50%), MAE-log 0,760 × 0,835. Teste prospectivo: 61,7% (v2) |
| Interface | Streamlit/Gradio opcional; tool obrigatória (função + MCP) | ✅ | Função + MCP, testado com o MCP Inspector e com o cliente do SDK oficial. Interface Streamlit opcional (`app.py`), que mostra cada tool do grafo terminando e abre a demo sem rede |
| Explicabilidade | SHAP + subagente em linguagem natural | ✅ | TreeSHAP exato dos dois modelos + explicador LLM com validador de fidelidade e texto padrão de reserva |
| Agnosticismo (opcional) | Treino reexecutável sobre qualquer base com esquema mínimo | ✅ | `treino_cliente.py`, tool MCP `treinar_modelo_cliente` |

## 4.1 Núcleo obrigatório

| Item | | Onde / como |
|---|---|---|
| Base documentada: origem, período, volume, licença, setores e o que ela não representa | ✅ | DATA_CARD.md (licença: Termos de Serviço da YouTube API, seção "Termos de uso e ética") |
| Opção A: features do canal, temporais, do texto e embeddings; gradient boosting; baseline = média histórica do canal | ✅ | Blocos A e B; média de LightGBM e CatBoost; baseline = **mediana** do canal (o TAPI aceita "média ou mediana"; a mediana resiste a um viral isolado) |
| Opção B reduzida: embedding de thumbnail/frames, transcrição e 5 a 8 atributos de VLM; medir o ganho sobre A | ✅ | SigLIP 2 (thumbnail e 3 quadros do gancho), faster-whisper, 17 campos do VLM; ganho por bloco com IC 95% |
| Experimento: validação temporal, split por canal, MAE em log, MAPE, Spearman, par-a-par, análise de erros | ✅ | EXPERIMENTO.md: todas as métricas, recortes, 20 maiores erros classificados |
| Intervalo (quantis do GBM) e sinal de baixa confiança | ✅ | Quantis 10%/90% + CQR; `confidence_flag` com 7 motivos codificados, validado no teste (sinalizados erram 1,6× mais) |
| Agente em LangGraph com tools de extração, histórico/comparáveis, predição, explicabilidade e ordenação | ✅ | As 5 tools da arquitetura são nós do grafo (`agente.py`) |
| Função Python documentada, servidor MCP e demonstração executável | ✅ | README seção 4; `servidor_mcp.py`; `demo.py` (com plano B sem rede) |
| README e registro de experimentos (MLflow) | ✅ | README; MLflow com versão e hash da base, cortes, parâmetros, métricas e SHAP |

## 4.2 Extensões opcionais

| Extensão | | Como ficou |
|---|---|---|
| Treinamento dinâmico | ✅ | `treino_cliente.py` + tool MCP `treinar_modelo_cliente`: valida o esquema mínimo (pandera), monta os blocos A e B (e C da capa, opcional) com as mesmas funções do treino, split temporal, CQR, avaliação única contra a mediana do canal com IC e avisos de deriva. O modelo sai pronto para o agente (`"modelo": "<nome>"`). Exemplo: `data/exemplos/base_cliente_marcas.parquet` |
| Intervalos calibrados por predição conformal | ✅ | CQR (Romano et al., 2019) na validação; cobertura medida no teste (82,7%) e no prospectivo (81,9%) |
| RAG sobre criativos comparáveis | ✅ | Comparáveis com desempenho real na saída da tool (evidência) e, na v2, o desempenho dos comparáveis como features (`rag.py`), escolhidas na validação e confirmadas no teste prospectivo |
| What-if (legenda, horário, gancho) | ◐ | Horário (`what_if_horarios`) e legenda (`what_if_legendas`). Gancho não: exigiria gerar variações do vídeo |
| LLM zero-shot × modelo treinado | ✅ | A nota `potencial_viral_0a10` do VLM acerta 52,4% dos pares; o modelo, 60,2% nos mesmos vídeos |
| Embeddings de vídeo completo (X-CLIP/VideoMAE) e de áudio (CLAP) | — | Não feito. A parte de vídeo que já existe (cortes, movimento, voz, energia e gancho) não somou no teste; os vídeos são apagados depois de medidos (o disco não comporta a amostra), e refazer exigiria ~20 h de download e GPU. Fica como evolução |

## 5. Dados

| Item | | Como ficou |
|---|---|---|
| Lista curada de canais de marcas e criadores | ✅ | Sementes + descoberta + verificação automática + revisão manual (com thumbnails) |
| Vídeos com idade ≥ 30 dias | ✅ | Filtro de maturidade; o histórico do canal também usa só Shorts com 30 dias |
| Campos: título, descrição, tags, duração, data/hora, categoria, inscritos, views, likes, comentários, thumbnail, URL | ✅ | Esquema em DATA_CARD.md (views, comentários e inscritos guardados, mas nunca usados como feature) |
| Meta de 5.000 a 15.000 vídeos e 60+ canais, com a data de coleta | ✅ | 40.865 vídeos-alvo (79.402 Shorts com o histórico) de 260 canais; `collected_at` em cada linha |
| Cota: playlists em vez de search | ✅ | `channels` → playlist de uploads → `videos`; `search` só na descoberta de canais |
| SMTPD e MicroLens | ◐ | Só como referência (ver seção 3) |
| Criativos de teste de canais fora do treino, com métrica real conhecida (de-para) | ✅ | 4 canais de demonstração reservados desde a coleta; `demo.py` |

## 6.1 Alvo, baseline, splits e métricas

| Elemento | | Como ficou |
|---|---|---|
| Alvo primário: log da razão curtidas / mediana do canal | ✅ | `log1p(curtidas) − log1p(mediana dos últimos 20 Shorts maduros)` |
| Alvo secundário: faixas abaixo / na média / acima / viral | ✅ | `relative_to_channel.faixa`; F1 macro reportado |
| Baseline: média ou mediana do canal; regressão linear sobre inscritos | ◐ | Mediana do canal e regressão linear sobre a **mediana de curtidas do canal**, não sobre inscritos: a API só dá os inscritos do dia da coleta, que para vídeos antigos são informação do futuro (o vídeo viral gerou os inscritos). A mediana de curtidas mede o tamanho da audiência no momento do post (FEATURES.md, "features descartadas") |
| Splits: temporal como principal, por canal para canais novos, sem validação cruzada aleatória | ✅ | 70/15/15 no tempo + 51 canais nunca vistos; e, depois do teste, um teste prospectivo e um backtest com origem móvel |
| Métricas: MAE e MAPE em log, Spearman global e intra-canal, par-a-par, cobertura, F1 | ✅ | Todas, mais o par-a-par na mesma época (decisão real) e o interval score (proper scoring rule) |
| Vazamento: sem views para prever curtidas; histórico só com posts anteriores | ✅ | 9 regras em FEATURES.md, incluindo a maturidade de 30 dias do histórico |
| Expectativa: Spearman 0,55–0,85 em curtidas absolutas | ✅ | 0,882 em curtidas absolutas no teste; no alvo relativo, 0,405 |

## 6.2 Camada multimodal

| Item do pipeline de referência | | Como ficou |
|---|---|---|
| yt-dlp em 360p, ID como chave de cache, cada etapa em disco | ✅ | `video.py`, cache por vídeo em JSON |
| 5 a 8 frames, 2 ou 3 do gancho, mais a thumbnail; PySceneDetect | ◐ | 3 quadros do gancho (0,5 s, 1,5 s e 2,5 s) + a thumbnail no embedding; o movimento usa o vídeo inteiro a 2 quadros/s; cortes com PySceneDetect |
| SigLIP 2 ou CLIP nos frames e na thumbnail, média + PCA 16–32 | ✅ | SigLIP 2; PCA 16 (v1) e 32 (v2) no texto e na thumbnail, 8 no gancho |
| DINOv2 (opcional) | — | Não feito: o SigLIP já cobre a thumbnail e a estética vem do VLM |
| faster-whisper: idioma, nº de palavras, densidade de fala | ◐ | Palavras e densidade (`tem_voz`, `voz_por_segundo`); o idioma não é usado porque o Whisper transcreve letra de música (regra 9 de FEATURES.md) |
| CLAP ou métricas simples de energia e música | ✅ | Energia RMS; música de fundo pelo VLM |
| MediaPipe (rosto), EasyOCR (texto), OpenCV (cor e movimento) | ✅ | Na thumbnail; optical flow no vídeo |
| YOLO (produto/pessoa) | — | Não usado: "produto em destaque" e "pessoa" vêm do VLM (`close_up_produto`, `tem_rosto`, `try_on_ou_demonstracao`), que entende o contexto de moda e beleza melhor que as classes genéricas do COCO |
| VLM com questionário estruturado em JSON | ✅ | Gemini 3.1 Flash-Lite, saída estruturada validada com Pydantic, só o vídeo no prompt |
| Fusão em colunas + ganho de cada bloco | ✅ | Escada A → A+B → +C → +D e "todos menos um", com controle negativo de ruído |
| Rubrica sugerida | ◐ | Cobre gancho, pessoas e produto, texto e edição, áudio, estética e mensagem, mais a adaptação por setor (swatch, aplicação na pele, try-on, corpo inteiro × detalhe). Não coleta os campos de texto livre (`hook_primeiros_3s`, `elementos_*_top5`) nem cor e brilho (medidos por OpenCV) |
| Clusterização de descritores (K-Means) | — | Depende dos descritores de texto livre, que a rubrica não coleta (trocar o prompt exigiria refazer a amostra inteira dentro da cota gratuita) |

## 6.3 Camadas de ambição

| Camada | | Como ficou |
|---|---|---|
| 1. Mínima (100% offline) | ✅ | Bloco C inteiro roda localmente |
| 2. Recomendada (VLM + clusterização) | ◐ | VLM na amostra (2.905 vídeos na base, mais os do teste prospectivo); sem a clusterização |
| 3. Ambiciosa (X-CLIP, fusão thumbnail + gancho, retrieval, ensemble) | ◐ | Fusão thumbnail + gancho (os dois embeddings), retrieval de comparáveis (`rag.py`, busca exata em numpy, no lugar do FAISS, que não é necessário para 40 mil vetores) e ensemble LightGBM + CatBoost. Sem X-CLIP |

## 6.4 Pegadinhas conhecidas

| Pegadinha | Como foi tratada |
|---|---|
| yt-dlp bloqueado por "not a bot" | IP residencial, pausa de 8 s entre downloads, parada automática ao detectar bloqueio, retomada por cache |
| Free tier do Gemini usa os dados e muda cotas | Só vídeos públicos; backoff para 429/5xx; sem cota, as colunas do VLM ficam ausentes e a tool sinaliza `extracao_incompleta`. O caminho local (Qwen2.5-VL) está documentado, não implementado |
| PCA antes do boosting, contagens em log, validação temporal | ✅ em todas |

## 7. Arquitetura e contrato

| Item | | Como ficou |
|---|---|---|
| Inspeção: quais atributos entraram, qual previsão saiu, em que evidência ela se apoia | ✅ | `atributos_extraidos` (todas as features, inclusive as RAG), `top_drivers` (SHAP), `contribuicao_por_bloco`, `comparables`, `confidence_reasons`, `modelo_versao` |
| O modelo é treinado offline e chamado como ferramenta; o LLM nunca chuta o número | ✅ | O LLM só redige; um validador determinístico confere todo número e fator citado |
| Entrada: creative, caption, planned_datetime, channel_id ou channel_stats, sector | ✅ | `contrato.EntradaCriativo` (Pydantic) |
| Saída: predicted_metric, interval_low/high, confidence_flag, relative_to_channel, top_drivers, explanation, comparables | ✅ | `contrato.PrevisaoCriativo` (Pydantic) |
| Esquema mínimo da base para retreino | ✅ | `treino_cliente.py` valida exatamente as 8 colunas |
| Entradas e saídas validadas com Pydantic | ✅ | Inclusive o relatório do treino dinâmico (`RelatorioTreino`) |

## 8. Tecnologias

Todas as recomendações principais foram seguidas (google-api-python-client, yt-dlp, pandas/Parquet, LightGBM e CatBoost, scikit-learn, sentence-transformers com `paraphrase-multilingual-mpnet-base-v2`, SigLIP 2, faster-whisper, Gemini Flash, SHAP exato nativo, LangGraph, SDK oficial do MCP, MLflow, pytest). As escolhas e versões estão justificadas na seção 5 do README. Streamlit e Plotly (opcionais) na interface `app.py`, num grupo de dependências à parte (`uv sync --group app`). Não usados: Colab/Kaggle (tudo rodou numa máquina local com Apple M4).

## 10. Entregáveis

| Entregável | | Onde |
|---|---|---|
| Agente em funcionamento, exposto como função e servidor MCP | ✅ | README seção 4 |
| Modelo treinado: base documentada, features declaradas, comparação com o baseline, artefatos versionados | ✅ | `models/lgbm_final.*` (v1) e `models/lgbm_v2.*` (v2, servida), DATA_CARD, FEATURES, MODEL_CARD |
| Experimento: hipótese, validação temporal, erro absoluto e relativo, ordenação, cobertura, ganho multimodal, análise de erros | ✅ | EXPERIMENTO.md (hipóteses registradas antes do treino) |
| Demonstração sobre um conjunto de criativos | ✅ | `python -m galaxies_tool.demo` (12 Shorts de 4 canais fora do treino; `--salvas` sem rede) e a interface `app.py` |
| README: problema e recorte, base, arquitetura, instalação, modelos, metodologia, resultados, limitações, evolução | ✅ | README seções 1 a 9 |
