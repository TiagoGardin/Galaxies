"""Teste prospectivo: Shorts publicados DEPOIS do último vídeo da base_v1, avaliados pelo caminho do agente.

Uso:
  uv run python -m galaxies_tool.prospectivo coletar           # curtidas de hoje dos Shorts novos (~25 un.)
  uv run python -m galaxies_tool.prospectivo extrair [--video]  # atributos com a Tool 1 do agente (cache)
  uv run python -m galaxies_tool.prospectivo exportar          # cache JSON → Parquet versionado
  uv run python -m galaxies_tool.prospectivo avaliar --parte principal|video   # avaliação ÚNICA de cada parte

Por que existe: o teste temporal (02/07 a 12/09/2026) foi avaliado uma única vez em 19/10, e qualquer
melhoria escolhida depois dele não pode ser medida nele de novo. A coleta de 12/10/2026 já listava os Shorts
publicados até 12/10, mas os publicados depois de 12/09 ainda não tinham 30 dias e ficaram fora da base. Em
21/10, os publicados até 21/09 completaram 30 dias: são vídeos que nenhum modelo, nenhuma escolha de
hiperparâmetro e nenhuma análise viu. Este módulo mede neles, uma vez, o artefato congelado e as variantes
pré-registradas antes de abrir os números.

Como cada vídeo é montado (o mesmo caminho de uma chamada real da tool, sem o LLM explicador):
- texto, título, descrição e tags: os da coleta de 12/10 (mais perto da publicação que os de hoje);
- curtidas (o alvo): as de hoje, pela YouTube Data API (`videos.list`, 1 unidade a cada 50 vídeos);
- bloco A: `canal.features_do_canal` com a base_v1 (histórico de Shorts publicados >= 30 dias antes);
- blocos B, C e D: `extracao.extrair_isolado`, a Tool 1 do agente. Sem `--video`, só texto e thumbnail;
  com `--video`, também o vídeo em 360p e a rubrica do VLM (lento e limitado pela cota do Gemini).
Saídas em data/prospectivo/ (fora do git a mídia; as tabelas e a coleta são versionadas).
"""

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from galaxies_tool.regras import IDADE_MINIMA_DIAS, eh_short, texto_pos_publicacao

PASTA = Path("data/prospectivo")
RAW_12OUT = Path("data/raw/coleta=2026-10-12/videos.parquet")
BASE = Path("data/processed/base_v1.parquet")
SPLITS = Path("data/processed/splits_v1.json")
CANAIS = Path("data/canais.csv")
ATRIBUTOS = PASTA / "atributos"  # um JSON por vídeo (sem vídeo) e outro com vídeo e VLM
MIN_HISTORICO = 5  # mesmo mínimo da base (processar.MIN_HISTORICO_TREINO)


def _coleta_mais_recente() -> Path:
    return max(PASTA.glob("coleta=*"))


# ---------------------------------------------------------------- coleta


def candidatos() -> pd.DataFrame:
    """Shorts da coleta de 12/10 publicados depois do último vídeo da base_v1 e com 30 dias ou mais hoje."""
    raw = pd.read_parquet(RAW_12OUT)
    raw = raw[[eh_short(d, p) for d, p in zip(raw.duration_s, raw.published_at, strict=True)]]
    raw = raw[raw.live_broadcast_content.isin(["none", ""])].copy()
    raw["published_at"] = pd.to_datetime(raw.published_at, utc=True).dt.as_unit("ns")
    fim_base = pd.read_parquet(BASE, columns=["published_at"]).published_at.max()
    maduro = pd.Timestamp(datetime.now(UTC)) - timedelta(days=IDADE_MINIMA_DIAS)
    return raw[(raw.published_at > fim_base) & (raw.published_at <= maduro)]


def coletar():
    from galaxies_tool.youtube_api import YouTube

    cand = candidatos()
    yt = YouTube()
    agora = datetime.now(UTC)
    itens = {v["id"]: v for v in yt.videos(cand.video_id.tolist())}
    st = pd.DataFrame(
        [
            {
                "video_id": vid,
                "likes_hoje": int(v["statistics"]["likeCount"]) if "likeCount" in v["statistics"] else None,
                "views_hoje": int(v["statistics"]["viewCount"]) if "viewCount" in v["statistics"] else None,
                "comments_hoje": (
                    int(v["statistics"]["commentCount"]) if "commentCount" in v["statistics"] else None
                ),
                "title_hoje": v["snippet"].get("title", ""),
            }
            for vid, v in itens.items()
        ]
    )
    df = cand.merge(st, on="video_id", how="left")
    df["disponivel_hoje"] = df.video_id.isin(itens)
    df["collected_at_hoje"] = agora.strftime("%Y-%m-%dT%H:%M:%SZ")
    pasta = PASTA / f"coleta={agora:%Y-%m-%d}"
    pasta.mkdir(parents=True, exist_ok=True)
    df.to_parquet(pasta / "videos.parquet", index=False)
    print(
        f"{len(df)} Shorts publicados de {df.published_at.min():%d/%m %H:%M} a "
        f"{df.published_at.max():%d/%m %H:%M} (UTC); {int(df.disponivel_hoje.sum())} ainda disponíveis, "
        f"{int(df.likes_hoje.notna().sum())} com curtidas visíveis. Cota usada: {yt.cota_usada} → {pasta}"
    )


# ---------------------------------------------------------------- montagem


def alvos() -> pd.DataFrame:
    """Vídeos-alvo do teste prospectivo, com as mesmas regras da base (DATA_CARD), menos o teto de 250 por
    canal (os vídeos novos não concorrem com os da janela antiga) e com o histórico da base_v1."""
    from galaxies_tool import canal

    df = pd.read_parquet(_coleta_mais_recente() / "videos.parquet")
    lista = pd.read_csv(CANAIS, dtype=str).fillna("")
    lista = lista[lista.incluir == "sim"].set_index("channel_id")
    splits = json.loads(SPLITS.read_text())
    funil = {"candidatos": len(df)}
    df = df[df.disponivel_hoje]
    funil["disponiveis_hoje"] = len(df)
    df = df[df.likes_hoje.notna()]
    funil["com_curtidas"] = len(df)
    df = df[df.channel_id.isin(lista.index)]
    df = df[lista.loc[df.channel_id, "demo"].to_numpy() != "sim"]
    funil["fora_dos_canais_demo"] = len(df)
    df = df[[not texto_pos_publicacao(t, d) for t, d in zip(df.title, df.description, strict=True)]]
    funil["sem_texto_pos_publicacao"] = len(df)
    linhas = []
    for r in df.itertuples():
        planned = pd.Timestamp(r.published_at)
        h = canal.features_do_canal(planned, channel_id=r.channel_id)
        if h["n_historico_total"] < MIN_HISTORICO:
            continue
        linhas.append(
            {
                "video_id": r.video_id,
                "channel_id": r.channel_id,
                "published_at": planned,
                "likes": int(r.likes_hoje),
                "views": r.views_hoje,
                "title": r.title,
                "description": r.description,
                "tags": list(r.tags) if r.tags is not None else [],
                "duracao_s": float(r.duration_s),
                "categoria_id": r.category_id,
                "setor": lista.loc[r.channel_id, "setor"],
                "tipo_canal": lista.loc[r.channel_id, "tipo"],
                "split": (
                    "prospectivo_canais_novos" if r.channel_id in splits["canais_novos"] else "prospectivo"
                ),
                **{k: v for k, v in h.items() if k != "mediana_likes_ult20"},
                **canal.features_de_tempo(planned),
            }
        )
    out = pd.DataFrame(linhas)
    funil["com_historico_minimo"] = len(out)
    out["alvo"] = np.log1p(out.likes) - out.log_mediana_likes_ult20
    out.attrs["funil"] = funil
    return out


# ---------------------------------------------------------------- extração (Tool 1 do agente)


def _arquivo(video_id: str, com_video: bool) -> Path:
    return ATRIBUTOS / ("com_video" if com_video else "sem_video") / f"{video_id}.json"


def extrair(com_video: bool, limite: int | None = None):
    """Roda a Tool 1 do agente em cada vídeo (texto, thumbnail e, com `com_video`, vídeo e VLM). Cache por
    vídeo; a ordem é aleatória (semente 42), para que uma parada no meio deixe uma amostra aleatória."""
    import time

    from galaxies_tool import extracao

    a = alvos()
    ordem = a.sample(frac=1, random_state=42)
    fila = [r for r in ordem.itertuples() if not _arquivo(r.video_id, com_video).exists()]
    if limite:
        fila = fila[:limite]
    _arquivo("x", com_video).parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(fila)} vídeos na fila ({'com' if com_video else 'sem'} vídeo)", flush=True)
    seguidas = 0
    for i, r in enumerate(fila, 1):
        t0 = time.time()
        try:
            atr = extracao.extrair_isolado(
                f"https://www.youtube.com/shorts/{r.video_id}", r.title, r.description, r.tags, com_video
            )
        except Exception as e:  # noqa: BLE001 — o processo de extração caiu: registra e segue
            print(f"{r.video_id}: {e}", flush=True)
            continue
        avisos = "; ".join(atr["_avisos"])
        if com_video and "rubrica do VLM" in atr["_faltando"] and "429" in avisos:
            # cota do Gemini (a mensagem vem truncada; por minuto já foi tentada de novo na extração): o vídeo
            # já ficou no cache do agente, e a rubrica é refeita na próxima rodada
            print(f"{r.video_id}: cota diária do VLM; fica para a próxima rodada", flush=True)
            cota_vlm = True
        else:
            _arquivo(r.video_id, com_video).write_text(json.dumps(atr, ensure_ascii=False))
            cota_vlm = False
        falhou_video = com_video and "vídeo" in atr["_faltando"]
        seguidas = seguidas + 1 if falhou_video else 0
        if com_video and atr["_faltando"] and not cota_vlm:
            print(f"{r.video_id}: faltou {atr['_faltando']} ({avisos[:300]})", flush=True)
        if seguidas >= 10:
            print("10 falhas seguidas na parte de vídeo: provável bloqueio do YouTube. Parando.", flush=True)
            return
        if i % 25 == 0:
            print(f"{i}/{len(fila)} ({time.time() - t0:.0f} s no último)", flush=True)
        if com_video:
            time.sleep(8.0)  # mesma pausa do video.py, para não ser bloqueado pelo YouTube
    print("Fim", flush=True)


def _parquet(com_video: bool) -> Path:
    return PASTA / f"atributos_{'com' if com_video else 'sem'}_video.parquet"


def exportar():
    """Junta os JSONs da extração (60 MB, fora do git) num Parquet por parte (versionado)."""
    for com_video in (False, True):
        arquivos = sorted(_arquivo("x", com_video).parent.glob("*.json"))
        if not arquivos:
            continue
        linhas = []
        for a in arquivos:
            atr = json.loads(a.read_text())
            atr["_avisos"], atr["_faltando"] = json.dumps(atr["_avisos"]), json.dumps(atr["_faltando"])
            atr["_extraido_em"] = datetime.fromtimestamp(a.stat().st_mtime, UTC).isoformat(timespec="seconds")
            linhas.append({"video_id": a.stem, **atr})
        df = pd.DataFrame(linhas)
        for c in df.columns:
            if c.startswith(("embtxt_", "embimg_", "embgancho_")):
                df[c] = df[c].astype("float32")
        df.to_parquet(_parquet(com_video), index=False)
        print(f"{len(df)} vídeos → {_parquet(com_video)}")


def _atributos(com_video: bool, ate: datetime | None = None) -> dict[str, dict]:
    """Atributos por vídeo: dos JSONs da extração ou, sem eles (clone do repositório), do Parquet. Com `ate`,
    só as extrações gravadas até esse instante (o corte pré-registrado da parte com vídeo)."""
    pasta = _arquivo("x", com_video).parent
    if pasta.exists() and any(pasta.glob("*.json")):
        return {a.stem: json.loads(a.read_text()) for a in pasta.glob("*.json")
                if ate is None or a.stat().st_mtime <= ate.timestamp()}
    if not _parquet(com_video).exists():
        return {}
    saida = {}
    for r in pd.read_parquet(_parquet(com_video)).to_dict("records"):
        if ate is not None and datetime.fromisoformat(r["_extraido_em"]) > ate:
            continue
        r["_avisos"], r["_faltando"] = json.loads(r["_avisos"]), json.loads(r["_faltando"])
        r.pop("_extraido_em", None)
        saida[r.pop("video_id")] = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}
    return saida


def tabela(com_video: bool = False, ate: datetime | None = None) -> pd.DataFrame:
    """Uma linha por vídeo-alvo com o bloco A e os atributos extraídos (os que existirem no cache).
    Com `com_video`, usa a extração completa onde ela existe e a sem vídeo no resto."""
    a = alvos()
    sem, com = _atributos(False), (_atributos(True, ate) if com_video else {})
    linhas = []
    for r in a.to_dict("records"):
        atr = dict(com.get(r["video_id"]) or sem.get(r["video_id"]) or {})
        if not atr:
            continue
        meta = {k: atr.pop(k) for k in list(atr) if k.startswith("_")}
        # duração e categoria: as da coleta (a extração também as busca na API, com o mesmo valor)
        atr.pop("duracao_s", None), atr.pop("categoria_id", None)
        linhas.append({**r, **atr, "faltando": meta["_faltando"], "tipo_criativo": meta["_tipo"]})
    df = pd.DataFrame(linhas)
    df["tem_video"] = df.get("cortes_por_segundo", pd.Series(np.nan, index=df.index)).notna()
    df["tem_vlm"] = df.get("vlm_formato", pd.Series(np.nan, index=df.index)).notna()
    df.attrs["funil"] = a.attrs["funil"]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("acao", choices=["coletar", "extrair", "exportar", "avaliar"])
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--limite", type=int)
    args, _ = ap.parse_known_args()
    if args.acao == "coletar":
        coletar()
    elif args.acao == "extrair":
        extrair(args.video, args.limite)
    elif args.acao == "exportar":
        exportar()
    else:
        from galaxies_tool import avaliacao_prospectiva

        avaliacao_prospectiva.main(sys.argv[2:])


if __name__ == "__main__":
    main()
