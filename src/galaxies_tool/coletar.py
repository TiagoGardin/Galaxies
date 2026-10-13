"""Coleta completa dos canais incluídos em data/canais.csv.

Uso: uv run python -m galaxies_tool.coletar
Saída: data/raw/coleta=AAAA-MM-DD/{canais,videos}.parquet

Para cada canal, pagina a playlist de uploads (mais recentes primeiro) e busca os detalhes
de cada lote de 50 vídeos, até cobrir a janela-alvo e ter JANELA_HISTORICO Shorts com
curtidas publicados antes de CORTE_HISTORICO (histórico do primeiro vídeo-alvo), ou até
MAX_UPLOADS. Cada canal é salvo em cache assim que termina, então a coleta pode ser
retomada se cair ou se a cota acabar.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from galaxies_tool.regras import CORTE_HISTORICO, JANELA_HISTORICO, eh_short
from galaxies_tool.youtube_api import YouTube, duracao_segundos

MAX_UPLOADS = 3000  # limite de segurança de cota por canal (60 páginas)
MARGEM_HISTORICO = 5  # Shorts antigos a mais, porque a playlist nem sempre é estritamente cronológica


def _shorts_antigos_com_likes(videos: list[dict]) -> int:
    n = 0
    for v in videos:
        pub = v["snippet"]["publishedAt"]
        if (pub < CORTE_HISTORICO and "likeCount" in v.get("statistics", {})
                and eh_short(duracao_segundos(v["contentDetails"].get("duration", "")), pub)):
            n += 1
    return n


def coletar_canal(yt: YouTube, canal: dict) -> tuple[list[dict], str]:
    playlist = canal["contentDetails"]["relatedPlaylists"]["uploads"]
    videos, lote = [], []
    for item in yt.uploads(playlist):
        lote.append(item["contentDetails"]["videoId"])
        if len(lote) == 50:
            videos += yt.videos(lote)
            lote = []
            if (_shorts_antigos_com_likes(videos) >= JANELA_HISTORICO + MARGEM_HISTORICO
                    or len(videos) >= MAX_UPLOADS):
                break
    if lote:
        videos += yt.videos(lote)
    motivo = ("historico_completo" if _shorts_antigos_com_likes(videos) >= JANELA_HISTORICO + MARGEM_HISTORICO
              else "limite_uploads" if len(videos) >= MAX_UPLOADS else "fim_da_playlist")
    return videos, motivo


def _linha_video(v: dict, coletado_em: str) -> dict:
    s, cd, st = v["snippet"], v["contentDetails"], v.get("statistics", {})
    thumbs = s.get("thumbnails", {})
    ordem = ("maxres", "standard", "high", "medium", "default")
    thumb = next((thumbs[k]["url"] for k in ordem if k in thumbs), "")

    def inteiro(chave):
        return int(st[chave]) if chave in st else None

    return {
        "video_id": v["id"],
        "channel_id": s["channelId"],
        "published_at": s["publishedAt"],
        "title": s.get("title", ""),
        "description": s.get("description", ""),
        "tags": s.get("tags", []),
        "category_id": s.get("categoryId", ""),
        "duration_s": duracao_segundos(cd.get("duration", "")),
        "live_broadcast_content": s.get("liveBroadcastContent", ""),
        "thumbnail_url": thumb,
        "likes": inteiro("likeCount"),
        "views": inteiro("viewCount"),
        "comments": inteiro("commentCount"),
        "collected_at": coletado_em,
    }


def main():
    hoje = datetime.now(UTC)
    pasta_cache = Path(f"data/cache/coleta/{hoje:%Y-%m-%d}")
    pasta_saida = Path(f"data/raw/coleta={hoje:%Y-%m-%d}")
    pasta_cache.mkdir(parents=True, exist_ok=True)
    pasta_saida.mkdir(parents=True, exist_ok=True)

    lista = pd.read_csv("data/canais.csv", dtype=str).fillna("")
    lista = lista[lista.incluir == "sim"]
    yt = YouTube()
    canais = yt.canais_por_ids(lista.channel_id.tolist())
    faltando = set(lista.channel_id) - {c["id"] for c in canais}
    if faltando:
        print(f"ATENÇÃO: {len(faltando)} canais não retornados pela API: {sorted(faltando)}")

    linhas_canal, resumo, falhas = [], [], []
    for i, c in enumerate(canais, 1):
        arq = pasta_cache / f"{c['id']}.json"
        if arq.exists():
            dados = json.loads(arq.read_text())
        else:
            try:
                videos, motivo = coletar_canal(yt, c)
            except RuntimeError as e:
                if "Cota" in str(e):
                    raise
                print(f"Falha de rede no canal {c['id']}; segue para o próximo ({e})")
                falhas.append(c["id"])
                continue
            dados = {"collected_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "motivo_parada": motivo, "videos": videos}
            arq.write_text(json.dumps(dados, ensure_ascii=False))
        st = c["statistics"]
        linhas_canal.append({
            "channel_id": c["id"],
            "handle": c["snippet"].get("customUrl", ""),
            "title": c["snippet"]["title"],
            "country": c["snippet"].get("country", ""),
            "subscribers": int(st["subscriberCount"]) if "subscriberCount" in st else None,
            "video_count": int(st.get("videoCount", 0)),
            "view_count": int(st.get("viewCount", 0)),
            "collected_at": dados["collected_at"],
        })
        resumo.append({"channel_id": c["id"], "n_videos": len(dados["videos"]),
                       "motivo": dados["motivo_parada"]})
        if i % 20 == 0:
            print(f"{i}/{len(canais)} canais (cota usada nesta execução: {yt.cota_usada})")

    if falhas:
        raise SystemExit(f"{len(falhas)} canais falharam: {falhas}. Rode de novo para completar "
                         "(os canais já coletados vêm do cache).")

    videos = []
    for r in resumo:
        dados = json.loads((pasta_cache / f"{r['channel_id']}.json").read_text())
        videos += [_linha_video(v, dados["collected_at"]) for v in dados["videos"]]
    df_v = pd.DataFrame(videos).drop_duplicates("video_id")
    df_c = pd.DataFrame(linhas_canal).merge(pd.DataFrame(resumo), on="channel_id")
    df_v.to_parquet(pasta_saida / "videos.parquet", index=False)
    df_c.to_parquet(pasta_saida / "canais.parquet", index=False)
    print(f"{len(df_v)} vídeos de {len(df_c)} canais salvos em {pasta_saida}.")
    print(f"Cota usada nesta execução: {yt.cota_usada}")
    print(df_c.motivo.value_counts().to_string())


if __name__ == "__main__":
    main()
