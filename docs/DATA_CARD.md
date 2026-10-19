# Data Card — Base de YouTube Shorts (moda e beleza, Brasil)

## Responsável e versão

- **Responsável:** Tiago Gardin (Inteli Academy), case Galaxies.
- **Versão:** `base_v1` (primeira coleta completa). Cada recoleta gera uma nova partição datada em `data/raw/`; a base processada só muda de versão quando os filtros ou a lista de canais mudam.
- **Lista de canais:** [`data/canais.csv`](../data/canais.csv), revisada em 11/10/2026.

## Fonte

**YouTube Data API v3**, chave própria (cota gratuita de 10.000 unidades/dia).

Fluxo de coleta, escolhido pelo custo de cota:

1. `channels.list` (`snippet,statistics,contentDetails`): dados do canal e ID da playlist de uploads (`UU…`). 1 unidade.
2. `playlistItems.list`: lista todos os vídeos da playlist de uploads, 50 por página. 1 unidade por página.
3. `videos.list` (`snippet,contentDetails,statistics`): detalhes e métricas, 50 IDs por chamada. 1 unidade.

`search.list` **não é usado para listar vídeos**: custa 100 unidades, está limitado a 100 chamadas/dia desde 06/2026 e devolve no máximo ~500 resultados. Ele foi usado **só na descoberta de canais** (abaixo).

**Canais**, em três etapas:

1. **Candidatos.** Vêm de duas fontes:
   - **Sementes:** canais conhecidos de moda e beleza, sobretudo marcas, listados por handle em [`data/canais_semente.csv`](../data/canais_semente.csv). Os handles das marcas foram confirmados testando variações via `channels.list` (1 unidade cada). Handles que apontavam para canais sem relação (ex.: `@farm` é um canal tailandês, `@avon` é o canal global dos EUA) foram descartados.
   - **Descoberta:** 24 consultas `search.list` por Shorts publicados desde 01/04/2025 (`regionCode=BR`, `relevanceLanguage=pt`, `videoDuration=short`), 12 por setor, guardando o canal de cada resultado ([`data/canais_descobertos.csv`](../data/canais_descobertos.csv), ~2.400 unidades). Reduz o viés de incluir só canais famosos.
2. **Verificação automática** ([`data/canais_candidatos.csv`](../data/canais_candidatos.csv)): país BR ou não declarado e pelo menos 30 Shorts com curtidas visíveis na janela-alvo (10 para marcas), contados nos 200 uploads mais recentes. Nos canais descobertos, filtro prévio de pelo menos 5 mil inscritos e 50 vídeos. Também é calculada a **aderência ao tema**: a fração dos Shorts do canal na janela cujo título, descrição ou tags citam termos de beleza ou de moda. Canais **descobertos** com aderência abaixo de 30% são marcados como provavelmente fora do tema. O limite foi escolhido olhando os dados, não vem da literatura. As sementes não passam por essa triagem, porque o tema delas já é conhecido e marcas raramente usam esses termos nos títulos. A busca trouxe muitos canais infantis, de família, humor e games, por isso essa triagem é necessária. A lista de termos serve **só para triagem de canais**, nunca como feature.
3. **Revisão manual** ([`data/canais.csv`](../data/canais.csv)): setor, tipo (marca/criador), país e idioma confirmados manualmente, porque `snippet.country` vem vazio com frequência e o setor da descoberta é só uma sugestão. A revisão leu título, descrição e tags de uma amostra de Shorts de cada canal e, para os canais descobertos, inspecionou as thumbnails dos 6 Shorts mais recentes. Motivos de exclusão (coluna `observacao`):
   - **conteúdo não autoral:** repostagem de vídeos de outros criadores, compilação de imagens (estilo Pinterest), imagens geradas por IA, fotos de catálogo de fornecedor, página de fã;
   - **fora do tema:** humor, infantil, lifestyle genérico, culinária, achadinhos de casa, loteria;
   - **não brasileiro:** conteúdo em inglês, francês, espanhol ou híndi;
   - **engajamento incentivado:** canais dominados por sorteios;
   - **duplicação de criador:** canal de cortes de outro canal já incluído (evita o mesmo criador no treino e no teste do split por canal);
   - **handle errado:** `@cea` é o CEA francês (energia atômica), não a C&A.

## Período e coletas

- **Vídeos-alvo:** Shorts publicados entre **01/04/2025** e **(data da coleta − 30 dias)**. O início foi escolhido depois de duas mudanças da plataforma: o limite de 3 min dos Shorts (15/10/2024) e a mudança na contagem de views (31/03/2025).
- **Vídeos de histórico:** Shorts anteriores a 01/04/2025 são coletados **só para calcular as features de histórico do canal** e não entram como alvo. A coleta de cada canal desce pela playlist até ter 25 Shorts com curtidas publicados antes de 02/03/2025 (20 da janela de histórico + 5 de margem) ou até o fim da playlist. Assim, a contagem de Shorts anteriores é exata até 25 em todos os canais; acima disso, depende de onde a coleta parou.
- **Datas de coleta:** verificação de canais em 11/10/2026; coleta completa da `base_v1` em 12/10/2026, entre 15h34 e 16h31 (horário de Brasília). Cota usada: cerca de 4.500 unidades (em três execuções, por duas quedas de rede; o cache evitou recoletar canais). Os canais da primeira execução foram coletados quando a regra de Short ainda não excluía vídeos anteriores a 07/2021; isso só afetou o ponto de parada (8 canais desceram até vídeos antigos), sem perda de dados: todos os Shorts reais desses canais foram coletados.

## Volume

**Canais aprovados** (revisão de 11/10/2026):

| Setor | Criadores | Marcas | Total |
|---|---|---|---|
| Beleza | 152 | 17 | 169 |
| Moda | 56 | 35 | 91 |
| **Total** | **208** | **52** | **260** |

**Vídeos na `base_v1`** (coleta de 12/10/2026, resumo completo em `data/processed/base_v1_resumo.json`):

| Setor | Canais | Vídeos-alvo elegíveis (criadores / marcas) | Vídeos só de histórico |
|---|---|---|---|
| Beleza | 169 (167 com alvos) | 26.682 (24.539 / 2.143) | 23.064 |
| Moda | 91 | 14.183 (9.207 / 4.976) | 14.773 |
| **Total** | **260 (258 com alvos)** | **40.865 (33.746 / 7.119)** | **37.837** |

Além desses, 700 vídeos-alvo ficam fora do treino por terem menos de 5 Shorts de histórico.

- **Período dos vídeos-alvo:** 01/04/2025 a 12/09/2026. O histórico começa em 08/07/2021.
- **Vídeos-alvo elegíveis por canal:** mediana de 165, de 9 a 250 (teto).
- **Curtidas dos vídeos-alvo elegíveis:** p10 = 19, mediana = 374, p90 = 4.826, p99 = 66.196. A mediana é de 501 nos criadores e de 43 nas marcas.
- **Por porte do canal:** médio 13.990, grande 11.089, pequeno 8.507, muito grande 7.279.
- **Canais de demonstração:** 688 vídeos elegíveis, fora de treino e avaliação.

**Funil de filtros** (do bruto à base):

| Etapa | Vídeos |
|---|---|
| Uploads coletados (todos os formatos) | 112.627 |
| Shorts (data e duração) | 84.166 |
| Com curtidas visíveis | 83.498 |
| Com pelo menos 30 dias na coleta (`base_v1`) | 79.402 |
| Na janela-alvo (a partir de 01/04/2025) | 68.083 |
| − texto que comenta o desempenho do próprio vídeo | 1 |
| − além do teto de 250 por canal (ficam como histórico) | 26.517 |
| **Vídeos-alvo (`is_target`)** | **41.565** |
| − sem 5 Shorts de histórico | 700 |
| **Vídeos-alvo elegíveis ao treino (`elegivel_treino`)** | **40.865** |

Dois canais incluídos (Carolina Bruna e Val Petrin) não têm nenhum vídeo-alvo elegível: publicaram quase todos os Shorts em poucas semanas, sem histórico de 30 dias.

## Camada multimodal (derivada dos vídeos)

| Parte | Cobertura | Como | Datas |
|---|---|---|---|
| Thumbnail (bloco C) | todos os 40.865 vídeos-alvo | thumbnail da API, quadro central 9:16 → cor, rosto (MediaPipe), OCR (EasyOCR), SigLIP 2 | 13–14/10/2026 |
| Vídeo (bloco C) | 8.219 de 8.361 vídeos da amostra (98,3%) | yt-dlp em 360p → PySceneDetect, optical flow, faster-whisper, energia RMS, 3 quadros do gancho (SigLIP 2); vídeo e áudio apagados depois de medidos | 13–17/10/2026 |
| Rubrica do VLM (bloco D) | 2.905 vídeos da amostra (1.419 do treino, 400 da validação, 1.086 do teste), até a cota gratuita acabar | `gemini-3.1-flash-lite` pela URL pública, só o vídeo no prompt | 13–19/10/2026 |

A amostra (`data/features/amostra_vlm.txt`, semente 42) tem 2.500 vídeos do treino, 1.000 da validação e todo o teste temporal; o VLM segue a mesma amostra em outra ordem (`fila_vlm.txt`). Falhas da parte de vídeo: 136 vídeos "made for kids" (o YouTube não entrega o arquivo sem login) e 6 vídeos que pedem login sem motivo aparente; ver [FEATURES.md](FEATURES.md). Na coleta, nenhum vídeo da amostra estava removido ou privado. As medidas por vídeo ficam em `data/features/video/` e `data/features/vlm/` (um JSON por vídeo, com o modelo usado); quadros, áudios e thumbnails ficam em `data/media/`, fora do git.

## Filtros aplicados

### Vídeos

| Filtro | Regra | Motivo |
|---|---|---|
| Definição de Short | Publicado a partir de 01/07/2021 e duração ≤ 180 s (≤ 60 s antes de 15/10/2024); lives e estreias sem duração ficam fora | A API não informa se o vídeo é Short nem se é vertical. O limite segue a regra da plataforma em cada época, e vídeos curtos anteriores ao lançamento global dos Shorts (07/2021) não eram Shorts. |
| Idade mínima | ≥ 30 dias na data da coleta | Curtidas já estabilizadas. |
| Curtidas ocultas | `likeCount` ausente → **excluído** (alvo e histórico), nunca tratado como 0 | Ausente ≠ zero. |
| Duplicatas | Chave `video_id`; recoletas não duplicam linhas | Cada coleta é guardada na própria partição com `collected_at`. |
| Texto editado após publicação | Título ou descrição que comenta o desempenho do **próprio vídeo** ("esse vídeo viralizou", "obrigada pelos 100 mil views", "1 milhão de visualizações") → **não é alvo**, fica só como histórico. Regex em `regras.TEXTO_POS_PUBLICACAO` | O texto atual reflete o resultado, não o criativo original. "Viralizou" sozinho não conta: "testei a base que viralizou" fala do produto. |
| Teto por canal | Máximo de **250 vídeos-alvo** por canal, **espaçados uniformemente no tempo** ao longo da janela | Evita que poucos canais muito ativos dominem a base. Pegar os 250 mais recentes concentraria esses canais na validação e no teste e os tiraria do treino. O histórico usa todos os Shorts. |
| Histórico mínimo | Vídeo-alvo só entra no treino (`elegivel_treino`) se o canal tiver pelo menos 5 Shorts publicados ≥ 30 dias antes dele | Sem histórico não há denominador confiável para o alvo relativo. |

### Canais

| Critério | Regra |
|---|---|
| Geografia | Canais brasileiros, conteúdo para o público brasileiro (pt-BR). O filtro é **por canal**: vídeos com música em outro idioma entram normalmente. |
| Setores | Moda e beleza. A meta era ~50/50, mas a revisão manual aprovou mais canais de beleza: 169 contra 91 (65%/35%). |
| Tipo | Marcas e criadores, com pelo menos 40% de criadores. |
| Porte | Pequenos, médios e grandes. |
| Volume | Pelo menos 30 Shorts na janela de alvo, com curtidas visíveis. **Marcas: pelo menos 10**, porque postam menos Shorts e são o perfil de cliente da Galaxies. |
| Canais de demonstração | 4 canais (`demo = sim`): Eudora, Alice Salazar, Luciane Cachinski e Liza Block. **Nunca usados no treino nem na avaliação.** |

## Esquema de colunas

Esquema de `data/processed/base_v1.parquet` (uma linha por Short com curtidas visíveis e pelo menos 30 dias de vida), validado com pandera em `galaxies_tool/processar.py`. Gerado por `python -m galaxies_tool.processar`; o resumo numérico fica em `data/processed/base_v1_resumo.json`. Compatível com o esquema mínimo de retreino do TAPI (`post_id`, `channel_id`, `published_at`, `caption`, `media_ref`, `metric_target`, `channel_stats_at_collection`, `sector`), indicado na última coluna.

| Coluna | Tipo | Descrição | Origem | Esquema TAPI |
|---|---|---|---|---|
| `video_id` | str | ID do vídeo (chave única) | `videos.list` | `post_id` |
| `channel_id` | str | ID do canal | `videos.list` | `channel_id` |
| `published_at` | datetime (UTC) | Data e hora de publicação | `snippet.publishedAt` | `published_at` |
| `title` | str | Título atual | `snippet.title` | `caption` (com `description`) |
| `description` | str | Descrição atual | `snippet.description` | `caption` |
| `tags` | list[str] | Tags declaradas | `snippet.tags` | — |
| `category_id` | str | Categoria do YouTube | `snippet.categoryId` | — |
| `duration_s` | int | Duração em segundos | `contentDetails.duration` | — |
| `thumbnail_url` | str | URL da thumbnail de maior resolução | `snippet.thumbnails` | `media_ref` |
| `url` | str | `https://www.youtube.com/shorts/<video_id>` | derivado | `media_ref` |
| `likes` | int | Curtidas na data da coleta | `statistics.likeCount` | `metric_target` |
| `views` | int (pode faltar) | Visualizações na data da coleta (**só análise, nunca feature**) | `statistics.viewCount` | — |
| `comments` | int (pode faltar) | Comentários na data da coleta (**só análise, nunca feature**) | `statistics.commentCount` | — |
| `collected_at` | datetime (UTC) | Data da coleta desta linha | coleta | — |
| `channel_subscribers_at_collection` | int (pode faltar) | Inscritos do canal no dia da coleta (arredondado pela API; **nunca feature**) | `channels.list` | `channel_stats_at_collection` |
| `sector` | str | `beleza` ou `moda` | `data/canais.csv` | `sector` |
| `channel_type` | str | `marca` ou `criador` | `data/canais.csv` | — |
| `is_demo_channel` | bool | Canal reservado para a demonstração (fora de treino e avaliação) | `data/canais.csv` | — |
| `texto_pos_publicacao` | bool | Título/descrição comenta o desempenho do próprio vídeo | derivado (regex) | — |
| `n_historico` | int | Shorts do canal (na base) publicados ≥ 30 dias antes deste | derivado | — |
| `is_target` | bool | Na janela-alvo, sem texto pós-publicação e dentro do teto de 250 por canal; `False` = só histórico | derivado | — |
| `elegivel_treino` | bool | `is_target` e `n_historico` ≥ 5 | derivado | — |

## Termos de uso e ética

- **Fonte oficial:** coleta só pela YouTube Data API v3, dentro dos [Termos de Serviço da API](https://developers.google.com/youtube/terms/api-services-terms-of-service) e da cota gratuita. Sem scraping de páginas.
- **Dados públicos:** só metadados e métricas públicas de canais e vídeos públicos. Nenhum dado de usuário, comentário individual ou informação privada é coletado.
- **Mídia não é redistribuída:** vídeos, áudios, frames e thumbnails baixados para a camada multimodal ficam em `data/media/`, que é ignorado pelo git.
- **Metadados publicados:** o repositório é público e inclui a coleta bruta e a base processada (`data/raw/`, `data/processed/`): IDs, títulos, descrições, tags e contagens públicas de curtidas, views e comentários, para que os resultados possam ser reproduzidos e auditados. Isso é uma redistribuição de dados da API. Os Termos da API restringem armazenar e exibir esses dados sem atualização periódica, e a trilha do curso (aula 1.8) recomenda não publicar a base; a decisão de mantê-la pública foi do responsável pelo projeto, para fins acadêmicos. Quem quiser só o código pode apagar `data/raw/` e `data/processed/` e recoletar com a própria chave.
- **VLM no free tier do Gemini:** os vídeos enviados podem ser usados pelo Google para treino. É aceitável porque são vídeos públicos, sem dados pessoais além do que o próprio criador publicou.
- **Atualidade:** os Termos da API pedem que dados armazenados sejam atualizados ou apagados periodicamente. As coletas são datadas (`collected_at`) e não são atualizadas: a base publicada é um retrato de 12/10/2026, de uso acadêmico.
- **Uso pretendido:** pesquisa e prova de conceito para a Galaxies. A base não deve ser usada para avaliar ou ranquear criadores individualmente fora deste contexto.

## Vieses e limitações — o que a base NÃO representa

- **Viés de seleção dos canais:** os canais foram escolhidos por serem relevantes hoje, e muitos cresceram justamente durante a janela coletada. A base tem mais canais "em ascensão" do que a realidade.
- **Viés de sobrevivência:** vídeos apagados ou privados (frequentemente os de baixo desempenho) não aparecem na API.
- **Métricas medidas no dia da coleta, não na publicação:** as curtidas são acumuladas até a coleta, inclusive as dos vídeos usados como histórico. Os inscritos são os do dia da coleta, arredondados pela API para 3 dígitos significativos, e por isso **não são usados como feature**.
- **Edição após a publicação:** título, descrição, tags e thumbnail retornados são a versão atual. Os casos óbvios são filtrados, os sutis não.
- **Heurística de Short:** baseada na data e na duração. Vídeos horizontais curtos publicados depois de 07/2021 podem entrar.
- **Recorte:** só moda e beleza, só Brasil, só canais com curtidas visíveis. Não representa outros setores, países ou plataformas (TikTok, Instagram).
- **Poucos vídeos de marca:** muitas marcas brasileiras de moda e beleza postam poucos ou nenhum Short no YouTube (na verificação, Riachuelo, Sallve, FARM Rio, Nivea, Garnier, Malwee, Sephora Brasil, L'Oréal Paris Brasil e AMARO ficaram abaixo de 10 Shorts com curtidas na janela). Mesmo com o mínimo reduzido para marcas, a base é dominada em volume por criadores, e o desempenho do modelo em canais de marca, que são os clientes da Galaxies, é menos garantido. Reportar as métricas separadas por tipo de canal.
- **Desequilíbrio de porte em moda:** na coleta, só 2 canais de moda tinham mais de 500 mil inscritos, contra 41 em beleza. O modelo vê poucos exemplos de canais de moda muito grandes.
- **Mais beleza que moda:** 169 contra 91 canais. Reportar as métricas separadas por setor.
- **Revisão manual com amostra:** a decisão de incluir cada canal usou uma amostra de títulos, descrições e thumbnails, e não o conteúdo completo. Pode haver canais incluídos com parte dos Shorts não autoral.
- **Concentração:** 82 dos 258 canais com alvos somam 50% dos vídeos-alvo elegíveis; o teto de 250 por canal evita concentração maior.
- **Engajamento de marcas é muito menor:** mediana de 43 curtidas por Short nas marcas, contra 501 nos criadores. Com poucas curtidas, a variação ao acaso pesa mais e a previsão relativa fica mais ruidosa.
- **Inconsistência da API:** 4 vídeos têm mais curtidas que visualizações (as contagens são atualizadas em momentos diferentes). Não afeta o modelo, que não usa visualizações.
- **Falso positivo conhecido do filtro de texto:** o único vídeo marcado fala de um Reels ("O primeiro REELS já bateu 1 MILHÃO de views"), não do próprio Short. Impacto desprezível.
