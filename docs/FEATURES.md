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

A precisão das flags é validada contra cerca de 100 vídeos rotulados à mão (ver EXPERIMENTO.md).

## Bloco C — Visual e áudio (camada multimodal mínima, 100% offline)

Pergunta do bloco: **como o vídeo é fisicamente (pixels, som)?**

| Feature | Descrição | Cálculo | Por que ajuda |
|---|---|---|---|
| `emb_thumb_pca_1..16` | Conteúdo visual da thumbnail | SigLIP 2 na thumbnail → PCA para 16 dimensões (ajustado só no treino) | Resume "do que é a imagem" |
| `thumb_brilho`, `thumb_contraste`, `thumb_saturacao`, `thumb_colorfulness` | Estética da thumbnail | OpenCV (média de V em HSV, desvio de luminância, média de S, métrica de Hasler-Süsstrunk) | Vídeos brilhantes e saturados formam cluster de alto engajamento |
| `thumb_tem_rosto`, `thumb_area_rosto` | Presença humana | MediaPipe Face Detection (Tasks API) | Rosto atrai o olhar; close-up é central em beleza |
| `thumb_tem_texto`, `thumb_n_palavras_ocr` | Texto sobreposto | EasyOCR | Text overlay aparece entre as features importantes na literatura |
| `cortes_por_segundo` | Ritmo de edição | PySceneDetect (`ContentDetector`) ÷ duração | Edição rápida segura atenção |
| `movimento_medio` | Quanto a imagem mexe | Magnitude média do optical flow (Farneback) entre frames amostrados | Vídeo estático perde o espectador |
| `emb_gancho_pca_1..8` | Primeiros 3 segundos | SigLIP 2 em 2–3 frames de 0–3 s → média → PCA | O gancho decide se a pessoa continua assistindo |
| `tem_voz`, `voz_por_segundo` | Presença de voz (fala **ou canto**) | faster-whisper: palavras transcritas ÷ duração (ver regra 9) | Narração explicativa forma cluster próprio |
| `energia_rms` | Energia do áudio | RMS médio do áudio (librosa) | Energia/música é a feature nº 1 no estudo de Shorts de referência |

O vídeo é baixado em 360p com yt-dlp; o ID do vídeo é a chave de cache de cada etapa. Se não houver tempo para a base inteira, frames e áudio rodam numa amostra estratificada, documentada no experimento.

## Bloco D — VLM

Pergunta do bloco: **que julgamento de conteúdo um especialista faria?** O Gemini Flash-Lite recebe **só o vídeo** (URL pública do YouTube), sem título, canal ou métricas (regra 6), e responde um JSON fixo. Fallback local: Qwen-VL quantizado.

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

Os campos binários são os mais confiáveis; as notas 0–10 são subjetivas e o VLM pode reconhecer criadores famosos, por isso o ganho delas é medido à parte.

Prompt (resumo): _"Assista ao vídeo e responda somente o JSON abaixo, sem texto adicional. Avalie apenas o que aparece no vídeo; não use conhecimento sobre o criador ou sobre a popularidade do vídeo."_ seguido do esquema com os campos acima. A resposta é validada com Pydantic; respostas inválidas são reenviadas uma vez e depois marcadas como ausentes.

## Features descartadas

| Feature | Motivo |
|---|---|
| `log_inscritos` | A API só dá os inscritos do dia da coleta. Para vídeos antigos é informação do futuro, com causalidade invertida (o vídeo viral gerou os inscritos) e métricas de validação infladas. O tamanho da audiência já é medido por `log_mediana_likes_ult20`, calculado no momento do post. |
| `idade_na_coleta_dias` | Com uma única coleta, é a data de publicação disfarçada: no split temporal o teste fica fora do intervalo do treino, e o modelo aprende tendências da plataforma que não se repetem. A maturidade das curtidas é garantida pelo filtro de ≥ 30 dias. |
| `idioma_audio` | `defaultAudioLanguage` é declarado pelo criador, não detectado no áudio. Vem vazio com frequência e fica quase constante com o recorte pt-BR. |
| `usa_audio_em_alta` (VLM) | O VLM não sabe o que estava em alta no dia da publicação, só o que ficou famoso depois. Substituída por `tem_musica_de_fundo` e `audio_original_ou_musica`. |
| `potencial_viral_0a10` (VLM) | É o LLM chutando o resultado ("o LLM nunca chuta o número"). Guardada só para a comparação opcional LLM zero-shot vs. modelo. |
| `channel_id` | Faria o modelo decorar o canal. |
| Views, comentários | Não existem antes da publicação. |
