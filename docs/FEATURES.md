# Features

## Regras gerais anti-vazamento

1. **Só entra o que existe antes da publicação.** Views, comentários e curtidas do próprio vídeo nunca são features.
2. **O histórico do canal usa só Shorts do mesmo canal publicados pelo menos 30 dias antes do vídeo-alvo.** A regra é idêntica no treino e na previsão, para que as curtidas do histórico estejam sempre "maduras". O denominador do alvo relativo usa a mesma função.
3. **Transformadores são ajustados só no treino:** PCA, scalers, K-Means e encoders são ajustados no split de treino e depois aplicados à validação e ao teste.
4. **`channel_id` nunca é feature.** O modelo decoraria o canal, e o split por canal perderia o sentido.
5. **Listas de palavras-chave (formato, CTA) são definidas pelo conhecimento do domínio,** nunca escolhendo palavras pela correlação com o alvo.
6. **O VLM recebe só o vídeo:** sem título, sem nome do canal e sem métricas no prompt.
7. **Horários são convertidos de UTC para `America/Sao_Paulo`.**
8. **Split temporal treino / validação / teste.** Hiperparâmetros são ajustados na validação, e o teste é avaliado uma única vez, no final.
9. **Features de áudio do Whisper medem voz, não fala:** o Whisper transcreve também letra de música. Por isso as features se chamam `tem_voz` e `voz_por_segundo`, e o idioma detectado não é usado. A distinção entre narração e música fica com o VLM.

Os blocos seguem a ablação do experimento (A → A+B → A+B+C → A+B+C+D): cada bloco é uma fonte de informação diferente, com custo e risco diferentes, e o ganho de cada um é medido separadamente.

## Bloco A — Canal, tempo e metadados

Pergunta do bloco: **sem olhar o conteúdo, o que já dá para saber?** Todas as features de histórico seguem a regra 2 (só Shorts publicados pelo menos 30 dias antes do vídeo-alvo).

| Feature | Descrição | Cálculo | Por que ajuda |
|---|---|---|---|
| `log_mediana_likes_ult20` | Patamar normal de curtidas do canal | `log1p` da mediana de curtidas dos últimos (até) 20 Shorts do histórico | Mede o tamanho da audiência no momento do post; mediana resiste a um viral isolado. É também o denominador do alvo |
| `std_log_likes_ult20` | Volatilidade do canal | Desvio padrão de `log1p(curtidas)` dos mesmos Shorts | Canal estável fica perto da mediana; instável oscila. Alimenta o intervalo e o `confidence_flag` |
| `tendencia` | Canal em alta ou em queda | `log1p(mediana ult5) − log1p(mediana ult20)` | Canal crescendo tende a superar a própria média |
| `n_shorts_anteriores` | Tamanho do histórico (truncado em 25) | `min(n_historico, 25)`: Shorts do canal publicados ≥ 30 dias antes do alvo | Histórico curto = menos confiança. O teto de 25 é necessário: a coleta só garante a contagem exata até 25 (ver DATA_CARD); acima disso o número dependeria de onde a coleta parou e viraria um indicador disfarçado de data |
| `shorts_ult_30d` | Frequência recente de postagem | Shorts publicados nos 30 dias antes do alvo | Frequência afeta distribuição pelo algoritmo e divisão da audiência |
| `dias_desde_ultimo_post` | Intervalo desde o último Short | Dias entre o alvo e o Short anterior | Volta após pausa longa tem comportamento diferente |
| `tipo_canal` | Marca ou criador | `data/canais.csv` | Engajamento orgânico difere entre marcas e criadores |
| `setor` | Beleza ou moda | `data/canais.csv` | Públicos curtem de forma diferente |
| `hora` | Hora local da publicação | `published_at` convertido para `America/Sao_Paulo` | Audiência online varia ao longo do dia; variável controlável pela marca |
| `dia_semana`, `fim_de_semana` | Dia da publicação | Idem | Idem |
| `data_sazonal` | Data comercial relevante | Flag para janelas de Black Friday, Dia das Mães, Dia dos Namorados e Natal | Picos de interesse em moda e beleza |
| `mes` | Mês da publicação | Idem | Sazonalidade. Só entra se o treino cobrir 12 meses; na `base_v1` cobre 12,7, então **entra** |
| `duracao_s` | Duração do Short | `contentDetails.duration` | Shorts curtos tendem a ter retenção maior |
| `categoria_id` | Categoria declarada | `snippet.categoryId` | Sinal fraco do tipo de conteúdo |
| `n_tags` | Quantidade de tags | `len(snippet.tags)` | Cuidado com descoberta/SEO; sinal fraco |

## Bloco B — Texto

Pergunta do bloco: **como o vídeo é anunciado?** Fonte: título + descrição + tags, normalizados (minúsculas, sem acento).

| Feature | Descrição | Cálculo | Por que ajuda |
|---|---|---|---|
| `titulo_n_chars`, `titulo_n_palavras` | Tamanho do título | Contagem | Títulos curtos e diretos tendem a funcionar melhor em Shorts |
| `n_hashtags`, `tem_hashtag_shorts` | Estratégia de descoberta | Contagem de `#`; presença de `#shorts` | Hashtags ajudam a distribuição |
| `n_emojis` | Tom emocional | Contagem de emojis no título | Emojis sinalizam tom e chamam atenção |
| `titulo_pergunta`, `titulo_exclamacao` | Curiosidade/ênfase | Presença de `?` e `!` no título | Gancho de curiosidade |
| `prop_maiusculas` | Grau de "clickbait" | Fração de letras maiúsculas no título | Títulos gritados chamam atenção, mas podem saturar |
| `titulo_tem_numero` | Formato de lista | Dígitos no título ("3 looks", "5 dicas") | Listas são formato clássico de engajamento |
| `n_mencoes`, `tem_link` | Collab e foco em venda | `@` e URLs na descrição | Collab amplia alcance; link indica intenção de venda |
| `tem_cta` | Pedido explícito de interação | Regex: "comenta", "salva", "compartilha", "link na bio", "cupom", "eu quero" | Pedir interação aumenta interação |
| `fmt_grwm`, `fmt_tutorial`, `fmt_haul`, `fmt_review`, `fmt_provador`, `fmt_antes_depois`, `fmt_trend` | Formato **declarado** no texto (0/1, multi-rótulo) | Listas de palavras-chave definidas pelo domínio (ver regra 5) | Cada formato tem desempenho típico diferente |
| `emb_texto_pca_1..16` | Significado do texto | `paraphrase-multilingual-mpnet-base-v2` em título + descrição → PCA para 16 dimensões (ajustado só no treino) | Captura tema e nuances que as regras não pegam |

Listas de palavras-chave dos formatos:

| Flag | Termos |
|---|---|
| `fmt_grwm` | grwm, get ready with me, arrume-se comigo, se arrume comigo |
| `fmt_tutorial` | tutorial, passo a passo, como fazer, aprenda, dica |
| `fmt_haul` | haul, comprinhas, recebidos, unboxing, abrindo |
| `fmt_review` | resenha, review, testei, testando, vale a pena, primeiras impressões |
| `fmt_provador` | provador, provando, look do dia, lookbook, try on, ootd |
| `fmt_antes_depois` | antes e depois, transformação, resultado |
| `fmt_trend` | trend, desafio, challenge |

Validação contra 100 vídeos rotulados à mão (EXPERIMENTO.md, "Validação das flags de formato"): precisão de 0,50 a 0,75 em GRWM, haul, provador e trend, mas recall de 0,15 a 0,43 (a maioria dos vídeos não declara o formato no texto); tutorial e antes e depois disparam por palavras genéricas (precisão 0,18 e 0,00).

## Bloco C — Visual e áudio (camada multimodal mínima, 100% offline)

Pergunta do bloco: **como o vídeo é fisicamente (pixels, som)?**

| Feature | Descrição | Cálculo | Por que ajuda |
|---|---|---|---|
| `emb_thumb_pca_1..16` | Conteúdo visual da thumbnail | SigLIP 2 na thumbnail → PCA para 16 dimensões (ajustado só no treino) | Resume "do que é a imagem" |
| `thumb_brilho`, `thumb_contraste`, `thumb_saturacao`, `thumb_colorfulness` | Estética da thumbnail | OpenCV (média de V em HSV, desvio de luminância, média de S, métrica de Hasler-Süsstrunk) | Vídeos brilhantes e saturados formam cluster de alto engajamento |
| `thumb_tem_rosto`, `thumb_area_rosto` | Presença humana | MediaPipe Face Detection (Tasks API) | Rosto atrai o olhar; close-up é central em beleza |
| `thumb_tem_texto`, `thumb_n_palavras_ocr` | Texto sobreposto | EasyOCR | Text overlay aparece entre as features importantes na literatura |
| `cortes_por_segundo` | Ritmo de edição | PySceneDetect (`ContentDetector`) ÷ duração | Edição rápida segura atenção |
| `movimento_medio` | Quanto a imagem mexe | Magnitude média do optical flow (Farneback) a 2 quadros/s, em cinza 180x320 | Vídeo estático perde o espectador |
| `emb_gancho_pca_1..8` | Primeiros 3 segundos | SigLIP 2 nos quadros de 0,5 s, 1,5 s e 2,5 s → média renormalizada → PCA para 8 dimensões (ajustado só no treino), separado do PCA da thumbnail | O gancho decide se a pessoa continua assistindo |
| `tem_voz`, `voz_por_segundo` | Presença de voz (fala **ou canto**) | faster-whisper `small` com VAD: ≥ 3 palavras transcritas; palavras ÷ duração (ver regra 9) | Narração explicativa forma cluster próprio |
| `energia_rms` | Energia do áudio | RMS médio do áudio (16 kHz, mono) | Energia/música é a feature nº 1 no estudo de Shorts de referência |

O vídeo é baixado em 360p com yt-dlp e apagado logo depois de medido; o ID do vídeo é a chave de cache de cada etapa (`data/features/video/<id>.json`, `galaxies_tool/video.py`; o embedding do gancho sai de `python -m galaxies_tool.thumbs siglip_gancho`). Frames e áudio rodam na mesma amostra do VLM (ver "Amostra dos blocos C (vídeo) e D" em [EXPERIMENTO.md](EXPERIMENTO.md)). Fora da amostra, essas colunas ficam ausentes (NaN), e o LightGBM aprende para que lado mandar os ausentes; a marca de "tem vídeo" (`tem_video`) é só diagnóstico e nunca entra no modelo (em produção ela seria sempre verdadeira).

No agente, um vídeo novo passa pelo mesmo caminho (`extracao.py`): uma URL do YouTube é baixada em 360p como a base, e um arquivo local do cliente é convertido para 360p no lado menor com ffmpeg antes de ser medido. O agente reproduz exatamente as colunas do treino (conferido em vídeos da validação: diferença zero no vídeo, no gancho e na rubrica; ~1e-5 no embedding da thumbnail, pelo arredondamento em float16 do cache).

Vídeos "made for kids" ficam sem esta parte: são públicos, mas sem login o YouTube não entrega o arquivo (só o storyboard), com nenhum cliente do yt-dlp. Passar cookies de uma conta Google resolveria, mas arrisca bloquear a conta, e por isso não foi feito. Na amostra, são 136 de 8.361 vídeos (1,6%), concentrados em canais de marca: 11% dos vídeos de moda/marca e 5% de beleza/marca, quase todos de FILA Brasil, Dafiti, Lupo e Simple Organic. Outros 6 vídeos do teste (0,1%) pedem login em qualquer tentativa ("Please sign in"), embora a API os declare públicos e sem restrição; ficaram sem esta parte. No total, 8.219 dos 8.361 vídeos da amostra (98,3%) têm a parte de vídeo. O LightGBM recebe essas colunas como ausentes. O VLM analisa os vídeos "made for kids" normalmente, então o bloco D cobre o que a parte de vídeo do bloco C perde.

## Bloco D — VLM

Pergunta do bloco: **que julgamento de conteúdo um especialista faria?** O Gemini Flash-Lite (`gemini-3.1-flash-lite`, temperatura 0) recebe **só o vídeo** (a URL pública do YouTube; no agente, um vídeo local vai pela Files API do Gemini), sem título, canal ou métricas (regra 6), e responde um JSON fixo validado com Pydantic (`galaxies_tool/vlm.py`). Não há fallback local implementado: se o Gemini falhar, as colunas do bloco D ficam ausentes e o agente sinaliza baixa confiança (`extracao_incompleta`). Um VLM aberto (Qwen2.5-VL) é o caminho para produção e para criativos confidenciais (ver README, Limitações).

Escolha do modelo (13/10/2026): o `gemini-3.5-flash-lite` estava constantemente sobrecarregado (erro 503, ~30 s por vídeo quando respondia), o que inviabilizaria a amostra; o `gemini-2.5-flash-lite` não está mais disponível para contas novas. O `gemini-3.1-flash-lite` processa o vídeo de verdade (~8,7 mil tokens de entrada para 95 s) em 2–3 s. Toda a amostra usa o mesmo modelo; as 134 rubricas feitas antes com o 3.5 ficam em `data/features/vlm_gemini35/` só para medir a concordância entre os dois modelos. Resultado nesses 134 vídeos: campos sim/não concordam em 87–99% (o mais fraco é `tem_musica_de_fundo`); notas 0–10 têm Spearman 0,66–0,84 (o mais fraco é `forca_hook_0a10`); `audio_original_ou_musica` e `tom_narracao` concordam em 77–80%, e `formato` em 72% (contra 34% de acertar sempre a classe mais comum). A troca de modelo não muda o que a rubrica mede; `forca_hook_0a10` e `formato` são os campos mais subjetivos; contra 100 vídeos rotulados à mão, o `formato` acerta 45,0% (54,2% sem provador e antes e depois, que o esquema não oferece; a classe mais comum sozinha acerta 36,1%), e superusa "tutorial" (EXPERIMENTO.md).

| Feature | Tipo | Por que ajuda |
|---|---|---|
| `forca_hook_0a10` | nota 0–10 | Qualidade do gancho, não só a existência |
| `tem_rosto` | bool | Presença humana confirmada semanticamente |
| `close_up_produto` | bool | Produto em destaque |
| `before_after_ou_transformacao` | bool | Formato mais forte de beleza |
| `try_on_ou_demonstracao` | bool | Formato mais forte de moda |
| `swatch` | bool | Específico de beleza |
| `aplicacao_na_pele` | bool | Específico de beleza |
| `corpo_inteiro_vs_detalhe` | `corpo_inteiro` / `detalhe` / `nao_se_aplica` | Específico de moda |
| `texto_na_tela`, `densidade_texto_0a10` | bool, nota | Overlay e quanto texto aparece |
| `ritmo_cortes_0a10` | nota | Percepção de ritmo (complementa o PySceneDetect) |
| `tem_musica_de_fundo` | bool | Substitui `usa_audio_em_alta` (ver descartadas) |
| `audio_original_ou_musica` | `fala` / `musica` / `ambos` / `ambiente` | Separa narração de música (o Whisper não separa) |
| `tom_narracao` | `ausente` / `calmo` / `animado` / `instrucional` | Tom da voz |
| `estetica_geral_0a10` | nota | Qualidade de produção |
| `clareza_beneficio_produto_0a10` | nota | Se fica claro por que comprar |
| `formato` | `tutorial` / `grwm` / `haul` / `review` / `trend` / `humor` / `outro` | Formato **real** do vídeo (o bloco B pega o declarado) |
| `potencial_viral_0a10` | nota | **Coletado, nunca usado no modelo**; só para a comparação LLM zero-shot vs. modelo |

Os campos binários são os mais confiáveis; as notas 0–10 são subjetivas e o VLM pode reconhecer criadores famosos, por isso o ganho delas é medido à parte (hipótese H5). No teste, binárias e notas empataram: H5 não se confirmou.

No modelo, as colunas ganham o prefixo `vlm_` (para não colidir com as de outros blocos, como `tem_rosto` × `thumb_tem_rosto`): binárias viram 0/1, notas ficam numéricas e os campos de escolha viram categorias com a lista fixa do esquema (não aprendida dos dados). A conversão é a mesma no treino e no agente (`galaxies_tool/multimodal.py`).

Prompt (texto exato em `galaxies_tool/vlm.py`): _"Você é um analista de criativos de moda e beleza. Assista ao vídeo e preencha a rubrica. Avalie apenas o que aparece e se ouve no vídeo. Não use conhecimento sobre quem é o criador, sobre a marca ou sobre a popularidade do vídeo. Notas de 0 a 10: 0 = ausente/péssimo, 10 = excelente."_ O esquema vai como `response_schema` (JSON estruturado), com temperatura 0. A resposta é validada com Pydantic; respostas inválidas são reenviadas uma vez e depois marcadas como ausentes.

Diferenças em relação à rubrica sugerida no TAPI: a rubrica cobre gancho, pessoas e produto, texto e edição, áudio, estética e mensagem, mas não coleta os campos de texto livre (`hook_primeiros_3s`, `elementos_video_top5`, `elementos_audio_top5`) nem `cor_dominante`, `brilho_saturacao_0a10` e `energia_musica_0a10`: brilho, saturação e cor já são medidos na thumbnail (OpenCV) e a energia do áudio no bloco C (RMS). Sem os descritores livres, a clusterização de descritores da "camada 2" do TAPI não foi feita. Limitação conhecida: o prompt ancora só os extremos da escala (0 e 10), não o meio; trocar o prompt agora exigiria refazer toda a amostra (aula 4.7), então fica registrado como limitação.

## Bloco R — Comparáveis (só na v2)

Pergunta do bloco: **como Shorts parecidos se saíram?** É a extensão "RAG sobre criativos comparáveis" do TAPI (`galaxies_tool/rag.py`). Para cada vídeo, busca os Shorts mais parecidos (cosseno na média dos embeddings brutos de texto e de thumbnail, renormalizada) entre os vídeos-alvo da base publicados **pelo menos 30 dias antes** dele (regra 2), e usa o desempenho real deles (o alvo, relativo ao canal de cada um):

| Feature | Cálculo | Por que ajuda |
|---|---|---|
| `rag_canal_alvo` | Média do alvo dos 5 Shorts mais parecidos do mesmo canal, ponderada pela similaridade | "Quando este canal fez algo parecido, foi bem?" |
| `rag_canal_sim` | Similaridade com o Short mais parecido do canal | Criativo repetido × novidade para a audiência |
| `rag_canal_recente` | Média do alvo dos 10 Shorts maduros mais recentes do canal | Controle sem conteúdo: o canal vem superando a própria mediana? |
| `rag_setor_alvo` | Média do alvo dos 20 mais parecidos de outros canais do mesmo setor, ponderada pela similaridade | O que esse tipo de conteúdo costuma render em relação ao próprio canal |

O próprio vídeo nunca é vizinho de si mesmo, e a regra dos 30 dias vale no treino e no agente (`canal.features_rag`, mesma função; um teste confere). Sem comparáveis maduros do canal, as três primeiras ficam ausentes. A busca é exata (numpy): com 40 mil vetores, um índice aproximado (FAISS) não é necessário. Na v2, os PCAs de texto e de thumbnail passam de 16 para 32 componentes (o TAPI sugere de 16 a 32); a escolha das duas mudanças está em EXPERIMENTO.md ("Depois do teste").

## Detectores não usados

- **YOLO (produto/pessoa), sugerido no TAPI:** "produto em destaque" e "pessoa" vêm do VLM (`close_up_produto`, `try_on_ou_demonstracao`, `tem_rosto`), que entende o contexto de moda e beleza; as classes genéricas do COCO não distinguem um batom de um frasco qualquer. Rosto e texto na thumbnail vêm do MediaPipe e do EasyOCR.
- **DINOv2 (opcional no TAPI):** o SigLIP 2 já descreve a imagem, e a estética vem do VLM.

## Features descartadas

| Feature | Motivo |
|---|---|
| `log_inscritos` | A API só dá os inscritos do dia da coleta. Para vídeos antigos é informação do futuro, com causalidade invertida (o vídeo viral gerou os inscritos) e métricas de validação infladas. O tamanho da audiência já é medido por `log_mediana_likes_ult20`, calculado no momento do post. |
| `idade_na_coleta_dias` | Com uma única coleta, é a data de publicação disfarçada: no split temporal o teste fica fora do intervalo do treino, e o modelo aprende tendências da plataforma que não se repetem. A maturidade das curtidas é garantida pelo filtro de ≥ 30 dias. |
| `idioma_audio` | `defaultAudioLanguage` é declarado pelo criador, não detectado no áudio. Vem vazio com frequência e fica quase constante com o recorte pt-BR. |
| `usa_audio_em_alta` (VLM) | O VLM não sabe o que estava em alta no dia da publicação, só o que ficou famoso depois. Substituída por `tem_musica_de_fundo` e `audio_original_ou_musica`. |
| `potencial_viral_0a10` (VLM) | É o LLM chutando o resultado ("o LLM nunca chuta o número"). Guardada só para a comparação opcional LLM zero-shot vs. modelo. |
| `channel_id` | Faria o modelo decorar o canal. |
| `largura`, `altura`, `vertical`, `duracao_video_s` (vídeo) | Medidos no download, mas quase todo Short é 360x640 e a duração já vem da API (`duracao_s`, bloco A). |
| `tem_video`, `tem_vlm` | Marcam quem está na amostra (diagnóstico). Em produção seriam sempre verdadeiras, e a falha de download pode depender do que aconteceu depois da publicação (aula 4.8). |
| Views, comentários | Não existem antes da publicação. |
