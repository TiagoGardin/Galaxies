# Data Card — Base de YouTube Shorts (moda e beleza, Brasil)

## Responsável e versão

<!-- Quem montou, versão da base (ex.: base_v1), data. -->

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
- **Vídeos de histórico:** Shorts anteriores a 01/04/2025 são coletados **só para calcular as features de histórico do canal** e não entram como alvo. Para cada canal, coleta-se o suficiente para que o primeiro vídeo-alvo tenha 20 Shorts publicados pelo menos 30 dias antes dele.
- **Datas de coleta:** _a preencher_.

## Volume

<!-- Nº de vídeos e canais, total e por setor e tipo de canal (marca/criador). Uma tabela. -->

## Filtros aplicados

### Vídeos

| Filtro | Regra | Motivo |
|---|---|---|
| Definição de Short | Duração ≤ 180 s (≤ 60 s para vídeos publicados antes de 15/10/2024) | A API não informa se o vídeo é Short nem se é vertical. O limite segue a regra da plataforma em cada época. |
| Idade mínima | ≥ 30 dias na data da coleta | Curtidas já estabilizadas. |
| Curtidas ocultas | `likeCount` ausente → **excluído** (alvo e histórico), nunca tratado como 0 | Ausente ≠ zero. |
| Duplicatas | Chave `video_id`; recoletas não duplicam linhas | Cada coleta é guardada na própria partição com `collected_at`. |
| Texto editado após publicação | Título/descrição com termos como "viralizou", "1 milhão", "obrigad\* pelos" → **excluído do treino** | O texto atual reflete o resultado, não o criativo original. |
| Teto por canal | Máximo de **250 vídeos-alvo** por canal (os mais recentes da janela) | Evita que poucos canais grandes dominem o treino. O histórico usa todos os Shorts. |

### Canais

| Critério | Regra |
|---|---|
| Geografia | Canais brasileiros, conteúdo para o público brasileiro (pt-BR). O filtro é **por canal**: vídeos com música em outro idioma entram normalmente. |
| Setores | Moda e beleza, ~50/50. |
| Tipo | Marcas e criadores, com pelo menos 40% de criadores. |
| Porte | Pequenos, médios e grandes. |
| Volume | Pelo menos 30 Shorts na janela de alvo, com curtidas visíveis. **Marcas: pelo menos 10**, porque postam menos Shorts e são o perfil de cliente da Galaxies. |
| Canais de demonstração | 3–4 canais reservados, **nunca usados no treino**, para o "de-para" da apresentação. |

## Esquema de colunas

<!-- Tabela: coluna | tipo | descrição | origem. -->

| Coluna | Tipo | Descrição | Origem |
|---|---|---|---|
| | | | |

## Termos de uso e ética

<!-- Termos da YouTube API, dados públicos, o que não é redistribuído (mídia). -->

## Vieses e limitações — o que a base NÃO representa

- **Viés de seleção dos canais:** os canais foram escolhidos por serem relevantes hoje, e muitos cresceram justamente durante a janela coletada. A base tem mais canais "em ascensão" do que a realidade.
- **Viés de sobrevivência:** vídeos apagados ou privados (frequentemente os de baixo desempenho) não aparecem na API.
- **Métricas medidas no dia da coleta, não na publicação:** as curtidas são acumuladas até a coleta, inclusive as dos vídeos usados como histórico. Os inscritos são os do dia da coleta, arredondados pela API para 3 dígitos significativos, e por isso **não são usados como feature**.
- **Edição após a publicação:** título, descrição, tags e thumbnail retornados são a versão atual. Os casos óbvios são filtrados, os sutis não.
- **Heurística de Short:** baseada só na duração. Vídeos horizontais curtos podem entrar.
- **Recorte:** só moda e beleza, só Brasil, só canais com curtidas visíveis. Não representa outros setores, países ou plataformas (TikTok, Instagram).
- **Poucos vídeos de marca:** muitas marcas brasileiras de moda e beleza postam poucos ou nenhum Short no YouTube (na verificação, Riachuelo, Sallve, FARM Rio, Nivea, Garnier, Malwee, Sephora Brasil, L'Oréal Paris Brasil e AMARO ficaram abaixo de 10 Shorts com curtidas na janela). Mesmo com o mínimo reduzido para marcas, a base é dominada em volume por criadores, e o desempenho do modelo em canais de marca, que são os clientes da Galaxies, é menos garantido. Reportar as métricas separadas por tipo de canal.
- _Completar após a coleta (ex.: distribuição por porte, concentração por canal)._
