"""Previsão aumentada por recuperação (extensão "RAG sobre criativos comparáveis" do TAPI, seção 4.2).

Para cada vídeo-alvo, busca os Shorts mais parecidos (cosseno na média dos embeddings de texto e de
thumbnail, os mesmos dos comparáveis da Tool 2) entre os publicados pelo menos 30 dias antes dele, e
transforma o desempenho real deles (o alvo relativo ao canal) em features:

- `rag_canal_alvo`: média do alvo dos 5 mais parecidos do MESMO canal, ponderada pela similaridade;
- `rag_canal_sim`: similaridade do mais parecido do mesmo canal (quão "repetido" é o criativo);
- `rag_canal_recente`: média do alvo dos 10 Shorts maduros mais recentes do canal, sem olhar o conteúdo
  (controle: separa "este canal vem superando a própria mediana" de "conteúdo parecido foi bem");
- `rag_setor_alvo`: média do alvo dos 20 mais parecidos de OUTROS canais do mesmo setor, ponderada pela
  similaridade (o que esse tipo de conteúdo costuma render em relação ao próprio canal).

Anti-vazamento (regra 2 de docs/FEATURES.md): só entram vizinhos publicados >= 30 dias antes do alvo, cujas
curtidas já estariam maduras no momento da previsão; o próprio vídeo nunca é vizinho de si mesmo. O conjunto
de busca é o dos vídeos-alvo da base (que têm alvo relativo), de todos os períodos: para um vídeo de julho,
um vídeo de maio com 60 dias já é histórico conhecido, mesmo que esteja na validação.
"""

import numpy as np
import pandas as pd

K_CANAL = 5
K_SETOR = 20
N_RECENTES = 10
DEFASAGEM = np.timedelta64(30, "D")
COLUNAS = ["rag_canal_alvo", "rag_canal_sim", "rag_canal_recente", "rag_setor_alvo"]
DESCRICOES = {
    "rag_canal_alvo": "Desempenho de Shorts parecidos do próprio canal",
    "rag_canal_sim": "Semelhança com Shorts anteriores do canal",
    "rag_canal_recente": "Desempenho recente do canal em relação à própria mediana",
    "rag_setor_alvo": "Desempenho de Shorts parecidos de outros canais do setor",
}


def _media_ponderada(sim: np.ndarray, alvo: np.ndarray) -> float:
    w = np.clip(sim, 1e-3, None)
    return float((w * alvo).sum() / w.sum())


def vetores(emb_texto: np.ndarray, emb_thumb: np.ndarray) -> np.ndarray:
    """Média dos dois embeddings normalizados, renormalizada (cosseno = produto escalar). Sem thumbnail,
    usa só o texto."""
    t = np.nan_to_num(emb_texto.astype(np.float32))
    i = np.nan_to_num(emb_thumb.astype(np.float32))
    v = t + i
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(n == 0, 1, n)


def features_rag(
    consulta: pd.DataFrame,
    vq: np.ndarray,
    pool: pd.DataFrame,
    vp: np.ndarray,
    bloco: int = 2048,
) -> pd.DataFrame:
    """Features RAG para cada linha de `consulta` (video_id, channel_id, setor, published_at), com vetores
    `vq`, buscando em `pool` (video_id, channel_id, setor, published_at, alvo) com vetores `vp`."""
    q_t = pd.to_datetime(consulta.published_at, utc=True).to_numpy("datetime64[ns]")
    p_t = pd.to_datetime(pool.published_at, utc=True).to_numpy("datetime64[ns]")
    p_alvo = pool.alvo.to_numpy(float)
    p_canal, q_canal = pool.channel_id.to_numpy(), consulta.channel_id.to_numpy()
    p_setor, q_setor = pool.setor.astype(str).to_numpy(), consulta.setor.astype(str).to_numpy()
    p_id, q_id = pool.video_id.to_numpy(), consulta.video_id.to_numpy()
    saida = np.full((len(consulta), len(COLUNAS)), np.nan)

    # mesmo canal: poucos vídeos por canal, matriz completa
    idx_pool = pd.Series(np.arange(len(pool))).groupby(p_canal).apply(np.asarray).to_dict()
    for c, idx_q in pd.Series(np.arange(len(consulta))).groupby(q_canal):
        ip = idx_pool.get(c)
        if ip is None:
            continue
        sim = vq[idx_q.to_numpy()] @ vp[ip].T
        for linha, iq in enumerate(idx_q.to_numpy()):
            ok = (p_t[ip] <= q_t[iq] - DEFASAGEM) & (p_id[ip] != q_id[iq])
            if not ok.any():
                continue
            s, a, t = sim[linha][ok], p_alvo[ip][ok], p_t[ip][ok]
            top = np.argsort(-s)[:K_CANAL]
            recentes = np.argsort(t)[-N_RECENTES:]
            saida[iq, 0] = _media_ponderada(s[top], a[top])
            saida[iq, 1] = float(s.max())
            saida[iq, 2] = float(a[recentes].mean())

    # outros canais do mesmo setor: em blocos, para caber na memória
    for setor in np.unique(q_setor):
        iq_setor = np.flatnonzero(q_setor == setor)
        ip = np.flatnonzero(p_setor == setor)
        for ini in range(0, len(iq_setor), bloco):
            iq = iq_setor[ini : ini + bloco]
            sim = vq[iq] @ vp[ip].T
            invalido = (p_t[ip][None, :] > (q_t[iq] - DEFASAGEM)[:, None]) | (
                p_canal[ip][None, :] == q_canal[iq][:, None]
            )
            sim[invalido] = -np.inf
            k = min(K_SETOR, sim.shape[1])
            top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            s = np.take_along_axis(sim, top, axis=1)
            a = p_alvo[ip][top]
            valido = np.isfinite(s)
            w = np.where(valido, np.clip(s, 1e-3, None), 0)
            soma = w.sum(axis=1)
            media = (w * np.where(valido, a, 0)).sum(axis=1) / np.where(soma > 0, soma, 1)
            saida[iq, 3] = np.where(soma > 0, media, np.nan)
    return pd.DataFrame(saida, columns=COLUNAS, index=consulta.index)


def vetores_do_df(df: pd.DataFrame) -> np.ndarray:
    """Vetores de busca a partir das colunas brutas embtxt_* e embimg_* (como no treino e no agente)."""
    tx = df.reindex(columns=[f"embtxt_{i}" for i in range(768)]).to_numpy(np.float32)
    im = df.reindex(columns=[f"embimg_{i}" for i in range(768)]).to_numpy(np.float32)
    return vetores(tx, im)


def features_da_base(df: pd.DataFrame) -> pd.DataFrame:
    """Features RAG de todos os vídeos-alvo da base, buscando na própria base (modelo.carregar)."""
    v = vetores_do_df(df)
    return features_rag(df, v, df, v)
