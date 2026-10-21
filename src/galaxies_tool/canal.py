"""Tool 2 do agente: histórico do canal (features do bloco A) e criativos comparáveis do histórico.

O histórico vem da base coletada (`channel_id`) ou do cliente (`channel_stats`), e passa pelas MESMAS
funções do treino (features.historico_do_alvo e atividade_de_postagem), com a data planejada no lugar da
data de publicação. Só entra o que existiria antes da publicação: curtidas de Shorts publicados >= 30
dias antes (regra 2) e datas de Shorts anteriores.

Comparáveis: Shorts do mesmo canal publicados pelo menos 30 dias antes da data planejada (curtidas
maduras, como no histórico), mais parecidos com o criativo
(cosseno na média dos embeddings de texto e de thumbnail), com o desempenho deles em relação ao canal.
Sem histórico do canal na base, busca no mesmo setor.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from galaxies_tool.features import (
    FUSO,
    TETO_N_HISTORICO,
    atividade_de_postagem,
    historico_do_alvo,
    shorts_da_coleta,
)
from galaxies_tool.modelo import EMB_TEXTO, EMB_THUMB, FEATURES
from galaxies_tool.regras import IDADE_MINIMA_DIAS

CANAIS = Path("data/canais.csv")
N_COMPARAVEIS = 5
CANAL_EXTERNO = "__canal_do_cliente__"


@lru_cache(maxsize=1)
def _base() -> pd.DataFrame:
    b = pd.read_parquet(
        "data/processed/base_v1.parquet",
        columns=[
            "video_id",
            "channel_id",
            "published_at",
            "likes",
            "title",
            "url",
            "category_id",
            "sector",
            "channel_type",
        ],
    )
    return b.assign(published_at=pd.to_datetime(b.published_at, utc=True).dt.as_unit("ns"))


@lru_cache(maxsize=1)
def _shorts_raw() -> pd.DataFrame:
    return shorts_da_coleta()


@lru_cache(maxsize=1)
def _alvos() -> pd.DataFrame:
    """Vídeos-alvo com o desempenho relativo real (para mostrar os comparáveis)."""
    return pd.read_parquet(
        FEATURES, columns=["video_id", "channel_id", "published_at", "setor", "alvo", "likes"]
    ).set_index("video_id")


@lru_cache(maxsize=1)
def _embeddings() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Embeddings de texto e de thumbnail alinhados pelos mesmos ids (os vídeos que têm os dois). Sem os
    arquivos (não versionados; ver README), não há comparáveis, e o resto do agente funciona normalmente."""
    if not (EMB_TEXTO.exists() and EMB_THUMB.exists()):
        return [], np.zeros((0, 768), np.float32), np.zeros((0, 768), np.float32)

    def ler(arq):
        ids = arq.with_name(arq.stem + "_ids.txt").read_text().split("\n")
        return pd.DataFrame(np.load(arq).astype(np.float32), index=ids)

    txt, img = ler(EMB_TEXTO), ler(EMB_THUMB)
    ids = txt.index.intersection(img.index)
    return list(ids), txt.loc[ids].to_numpy(), img.loc[ids].to_numpy()


@lru_cache(maxsize=1)
def pool_rag() -> tuple[pd.DataFrame, np.ndarray]:
    """Conjunto de busca das features RAG (variante v2): os vídeos-alvo da base com o alvo relativo e os
    vetores de texto + thumbnail, os mesmos usados no treino (rag.features_da_base)."""
    from galaxies_tool import rag

    ids, txt, img = _embeddings()
    pool = _alvos().reindex(ids).reset_index().rename(columns={"index": "video_id"})
    return pool, rag.vetores(txt, img) if ids else np.zeros((0, 768), np.float32)


def features_rag(atr: dict, planned: pd.Timestamp, setor: str, channel_id: str | None) -> dict:
    """Features RAG de um criativo novo (mesma função do treino). Sem os embeddings da base, ficam NaN."""
    from galaxies_tool import rag

    pool, vp = pool_rag()
    if pool.empty:
        return dict.fromkeys(rag.COLUNAS, np.nan)
    consulta = pd.DataFrame(
        {"video_id": ["__novo__"], "channel_id": [channel_id or CANAL_EXTERNO], "setor": [setor],
         "published_at": [planned]}
    )
    vq = rag.vetores_do_df(pd.DataFrame([atr]))
    return rag.features_rag(consulta, vq, pool, vp).iloc[0].to_dict()


def canal_conhecido(channel_id: str) -> bool:
    return bool((_base().channel_id == channel_id).any())


def tipo_do_canal(channel_id: str) -> str | None:
    tipos = _base().loc[_base().channel_id == channel_id, "channel_type"]
    return tipos.iloc[0] if len(tipos) else None


def features_do_canal(
    planned: pd.Timestamp, channel_id: str | None = None, posts: list[tuple[pd.Timestamp, int]] | None = None
) -> dict:
    """Bloco A do canal para um criativo publicado em `planned` (UTC)."""
    if channel_id is not None:
        base, raw, cid = _base(), _shorts_raw(), channel_id
    else:
        cid = CANAL_EXTERNO
        datas = pd.to_datetime([p for p, _ in posts], utc=True).as_unit("ns")
        base = pd.DataFrame({"channel_id": cid, "published_at": datas, "likes": [lk for _, lk in posts]})
        raw = base[["channel_id", "published_at"]]
    alvo = pd.DataFrame({"channel_id": [cid], "published_at": [planned.as_unit("ns")]}, index=[0])
    h = historico_do_alvo(base, alvo).loc[0]
    a = atividade_de_postagem(raw, alvo).loc[0]
    return {
        "log_mediana_likes_ult20": h.log_mediana_likes_ult20,
        "mediana_likes_ult20": h.mediana_likes_ult20,
        "std_log_likes_ult20": h.std_log_likes_ult20,
        "tendencia": h.tendencia,
        "n_shorts_anteriores": min(int(h.n_hist), TETO_N_HISTORICO),
        "n_historico_total": int(h.n_hist),
        "shorts_ult_30d": int(a.shorts_ult_30d),
        "dias_desde_ultimo_post": a.dias_desde_ultimo_post,
    }


def categoria_mais_comum(channel_id: str) -> str | None:
    cats = _base().loc[_base().channel_id == channel_id, "category_id"]
    return cats.mode().iloc[0] if len(cats) else None


def features_de_tempo(planned: pd.Timestamp) -> dict:
    from galaxies_tool.features import features_tempo

    return features_tempo(pd.Series([planned])).iloc[0].to_dict()


def comparaveis(
    atr: dict, planned: pd.Timestamp, setor: str, channel_id: str | None, n: int = N_COMPARAVEIS
) -> list[dict]:
    """Shorts mais parecidos com o criativo entre os publicados pelo menos 30 dias antes de `planned`
    (mesmo canal ou setor). A janela de maturidade é a mesma do histórico (regra 2): um Short de poucos
    dias ainda não tem as curtidas finais, que são as que aparecem aqui (aula 6.4)."""
    ids, txt, img = _embeddings()
    if not ids:
        return []
    alvos = _alvos().reindex(ids)
    maduro = planned - pd.Timedelta(days=IDADE_MINIMA_DIAS)
    elegivel = (alvos.published_at <= maduro).to_numpy().copy()
    mesmo_canal = (alvos.channel_id == channel_id).to_numpy() if channel_id else np.zeros(len(ids), bool)
    if (elegivel & mesmo_canal).sum() >= n:
        elegivel = elegivel & mesmo_canal
    else:
        elegivel = elegivel & (alvos.setor == setor).to_numpy()
    if not elegivel.any():
        return []
    sim = np.zeros(len(ids))
    partes = 0
    if "embtxt_0" in atr:
        q = np.array([atr[f"embtxt_{i}"] for i in range(txt.shape[1])], dtype=np.float32)
        sim += txt @ q
        partes += 1
    if "embimg_0" in atr:
        q = np.array([atr[f"embimg_{i}"] for i in range(img.shape[1])], dtype=np.float32)
        sim += img @ q
        partes += 1
    sim = np.where(elegivel, sim / max(partes, 1), -np.inf)
    base = _base().set_index("video_id")
    saida = []
    for i in np.argsort(-sim)[:n]:
        vid = ids[i]
        b, a = base.loc[vid], alvos.loc[vid]
        saida.append(
            {
                "video_id": vid,
                "url": b.url,
                "titulo": b.title,
                "publicado_em": a.published_at.tz_convert(FUSO).strftime("%d/%m/%Y"),
                "curtidas": int(a.likes),
                "desempenho_vs_canal": round(float(np.exp(a.alvo)), 2),
                "similaridade": round(float(sim[i]), 3),
            }
        )
    return saida
