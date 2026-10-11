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

## Bloco A — Canal, tempo e metadados

<!-- Pergunta do bloco: sem olhar o conteúdo, o que já dá para saber? -->

| Feature | Descrição | Cálculo | Por que ajuda |
|---|---|---|---|
| | | | |

## Bloco B — Texto

<!-- Pergunta do bloco: como o vídeo é anunciado? -->

| Feature | Descrição | Cálculo | Por que ajuda |
|---|---|---|---|
| | | | |

## Bloco C — Visual e áudio

<!-- Pergunta do bloco: como o vídeo é fisicamente (pixels, som)? -->

| Feature | Descrição | Cálculo | Por que ajuda |
|---|---|---|---|
| | | | |

## Bloco D — VLM

<!-- Pergunta do bloco: que julgamento de conteúdo um especialista faria? Inclua o prompt/rubrica. -->

| Feature | Descrição | Cálculo | Por que ajuda |
|---|---|---|---|
| | | | |

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
