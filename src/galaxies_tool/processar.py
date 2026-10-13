"""Gera a base processada a partir de uma coleta bruta, aplicando os filtros do data card.

Uso: uv run python -m galaxies_tool.processar [AAAA-MM-DD]   (padrão: coleta mais recente)
Entradas: data/raw/coleta=AAAA-MM-DD/{videos,canais}.parquet, data/canais.csv
Saídas: data/processed/base_v1.parquet e data/processed/base_v1_resumo.json

Uma linha por Short com curtidas visíveis e pelo menos 30 dias de vida. `is_target` marca os
vídeos que podem ser alvo; os demais ficam na base só como histórico do canal.
"""

import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa

from galaxies_tool.regras import (
    IDADE_MINIMA_DIAS,
    INICIO_JANELA_ALVO,
    TETO_ALVOS_POR_CANAL,
    eh_short,
    texto_pos_publicacao,
)

MIN_HISTORICO_TREINO = 5  # model card: pelo menos 5 Shorts de histórico para entrar no treino
SAIDA = Path("data/processed/base_v1.parquet")

ESQUEMA = pa.DataFrameSchema(
    {
        "video_id": pa.Column(str, unique=True),
        "channel_id": pa.Column(str),
        "published_at": pa.Column("datetime64[ns, UTC]"),
        "title": pa.Column(str),
        "description": pa.Column(str),
        "tags": pa.Column(object),
        "category_id": pa.Column(str),
        "duration_s": pa.Column(int, pa.Check.in_range(1, 180)),
        "thumbnail_url": pa.Column(str),
        "url": pa.Column(str, pa.Check.str_startswith("https://www.youtube.com/shorts/")),
        "likes": pa.Column(int, pa.Check.ge(0)),
        "views": pa.Column("Int64", nullable=True),
        "comments": pa.Column("Int64", nullable=True),
        "collected_at": pa.Column("datetime64[ns, UTC]"),
        "channel_subscribers_at_collection": pa.Column("Int64", nullable=True),
        "sector": pa.Column(str, pa.Check.isin(["beleza", "moda"])),
        "channel_type": pa.Column(str, pa.Check.isin(["marca", "criador"])),
        "is_demo_channel": pa.Column(bool),
        "texto_pos_publicacao": pa.Column(bool),
        "n_historico": pa.Column(int, pa.Check.ge(0)),
        "is_target": pa.Column(bool),
        "elegivel_treino": pa.Column(bool),
    },
    strict=True,
)


def pasta_coleta(data: str | None) -> Path:
    if data:
        return Path(f"data/raw/coleta={data}")
    return max(Path("data/raw").glob("coleta=*"))


def contar_historico(df: pd.DataFrame) -> pd.Series:
    """Para cada vídeo, quantos Shorts do mesmo canal foram publicados >= 30 dias antes."""
    n = pd.Series(0, index=df.index)
    defasagem = np.timedelta64(IDADE_MINIMA_DIAS, "D")
    for _, g in df.groupby("channel_id"):
        datas = np.sort(g.published_at.values)
        n.loc[g.index] = np.searchsorted(datas, g.published_at.values - defasagem, side="right")
    return n


def selecionar_alvos(candidatos: pd.DataFrame, teto: int = TETO_ALVOS_POR_CANAL) -> pd.Index:
    """Até `teto` vídeos por canal, espaçados uniformemente no tempo.

    Pegar os mais recentes concentraria os canais muito ativos no fim da janela (validação e
    teste) e os tiraria do treino; o espaçamento uniforme preserva a distribuição no tempo.
    """
    escolhidos = []
    for _, g in candidatos.sort_values("published_at").groupby("channel_id"):
        if len(g) <= teto:
            escolhidos.append(g.index)
        else:
            posicoes = np.unique(np.round(np.linspace(0, len(g) - 1, teto)).astype(int))
            escolhidos.append(g.index[posicoes])
    return pd.Index(np.concatenate(escolhidos)) if escolhidos else pd.Index([])


def processar(pasta: Path) -> tuple[pd.DataFrame, dict]:
    videos = pd.read_parquet(pasta / "videos.parquet")
    canais = pd.read_parquet(pasta / "canais.parquet")
    lista = pd.read_csv("data/canais.csv", dtype=str).fillna("")
    lista = lista[lista.incluir == "sim"]
    contagem = {"brutos": len(videos)}

    v = videos.copy()
    v["short"] = [eh_short(d, p) for d, p in zip(v.duration_s, v.published_at, strict=True)]
    v = v[v.short & (v.live_broadcast_content.isin(["none", ""]))]
    contagem["shorts"] = len(v)
    v = v[v.likes.notna()]
    contagem["shorts_com_curtidas"] = len(v)

    v["published_at"] = pd.to_datetime(v.published_at, utc=True).dt.as_unit("ns")
    v["collected_at"] = pd.to_datetime(v.collected_at, utc=True).dt.as_unit("ns")
    v = v[v.published_at <= v.collected_at - timedelta(days=IDADE_MINIMA_DIAS)]
    contagem["shorts_maduros"] = len(v)

    v = v.merge(lista[["channel_id", "setor", "tipo", "demo"]], on="channel_id", how="inner")
    v = v.merge(canais[["channel_id", "subscribers"]], on="channel_id", how="left")
    v = v.rename(columns={"setor": "sector", "tipo": "channel_type",
                          "subscribers": "channel_subscribers_at_collection"})
    v["is_demo_channel"] = v.demo == "sim"
    v["url"] = "https://www.youtube.com/shorts/" + v.video_id
    v["likes"] = v.likes.astype(int)
    for col in ("views", "comments", "channel_subscribers_at_collection"):
        v[col] = v[col].astype("Int64")
    v["texto_pos_publicacao"] = [texto_pos_publicacao(t, d) for t, d in zip(v.title, v.description, strict=True)]
    v = v.reset_index(drop=True)
    v["n_historico"] = contar_historico(v)

    na_janela = v.published_at >= pd.Timestamp(INICIO_JANELA_ALVO)
    candidato = na_janela & ~v.texto_pos_publicacao
    v["is_target"] = False
    v.loc[selecionar_alvos(v[candidato]), "is_target"] = True
    v["elegivel_treino"] = v.is_target & (v.n_historico >= MIN_HISTORICO_TREINO)

    contagem.update({
        "na_janela_alvo": int(na_janela.sum()),
        "texto_pos_publicacao_na_janela": int((na_janela & v.texto_pos_publicacao).sum()),
        "fora_do_teto_250": int(candidato.sum() - v.is_target.sum()),
        "alvos": int(v.is_target.sum()),
        "alvos_com_historico_minimo": int(v.elegivel_treino.sum()),
        "so_historico": int((~v.is_target).sum()),
    })
    v = v[list(ESQUEMA.columns)].sort_values(["channel_id", "published_at"]).reset_index(drop=True)
    return ESQUEMA.validate(v), contagem


def resumo(base: pd.DataFrame, contagem: dict, pasta: Path) -> dict:
    alvos = base[base.is_target]
    elegiveis = base[base.elegivel_treino]
    por_canal = elegiveis.groupby("channel_id").size().sort_values(ascending=False)
    acumulado = por_canal.cumsum() / por_canal.sum()
    inscritos = base.groupby("channel_id").channel_subscribers_at_collection.first()
    porte = pd.cut(inscritos, [0, 20_000, 100_000, 500_000, np.inf],
                   labels=["pequeno (<20k)", "médio (20–100k)", "grande (100–500k)", "muito grande (>500k)"])
    elegiveis_porte = elegiveis.channel_id.map(porte).astype(str)

    def tabela(df, chave):
        return {"|".join(map(str, k)) if isinstance(k, tuple) else str(k): int(n)
                for k, n in df.groupby(chave, observed=True).size().items()}

    return {
        "coleta": pasta.name.split("=")[1],
        "coletado_entre": [str(base.collected_at.min()), str(base.collected_at.max())],
        "funil": contagem,
        "canais_com_videos": int(base.channel_id.nunique()),
        "canais_com_alvos_elegiveis": int(elegiveis.channel_id.nunique()),
        "periodo_alvos": [str(alvos.published_at.min().date()), str(alvos.published_at.max().date())],
        "periodo_historico_inicio": str(base.published_at.min().date()),
        "elegiveis_por_setor_tipo": tabela(elegiveis, ["sector", "channel_type"]),
        "elegiveis_por_porte": {k: int(n) for k, n in elegiveis_porte.value_counts().items()},
        "elegiveis_demo": int(elegiveis.is_demo_channel.sum()),
        "canais_para_50pct_dos_elegiveis": int((acumulado < 0.5).sum() + 1),
        "curtidas_elegiveis": {q: float(elegiveis.likes.quantile(p))
                               for q, p in [("p10", .1), ("mediana", .5), ("p90", .9), ("p99", .99)]},
    }


def main():
    pasta = pasta_coleta(sys.argv[1] if len(sys.argv) > 1 else None)
    base, contagem = processar(pasta)
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    base.to_parquet(SAIDA, index=False)
    r = resumo(base, contagem, pasta)
    SAIDA.with_name("base_v1_resumo.json").write_text(json.dumps(r, ensure_ascii=False, indent=2))
    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
