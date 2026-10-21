"""Treino dinâmico, agnóstico à base (TAPI: "Agnosticismo" na seção 3 e "Treinamento dinâmico" na 4.2).

Recebe uma base no esquema mínimo do TAPI e treina, calibra e avalia um modelo especialista com o MESMO
protocolo e as MESMAS funções do modelo principal (docs/EXPERIMENTO.md), em escala menor. É o caminho para
quando um cliente liberar a própria base de posts: o modelo sai pronto para o agente.

Uso:
  uv run python -m galaxies_tool.treino_cliente treinar <base.csv|.parquet> --nome <nome> [--midia]
  uv run python -m galaxies_tool.treino_cliente exemplo   # base de exemplo no esquema mínimo (marcas da base_v1)
Também como função (`treinar_modelo_cliente`) e como tool do servidor MCP (mesmo nome). Depois, o agente usa
o modelo com `"modelo": "<nome>"` na entrada de `prever_criativo`.

Esquema mínimo (as 8 colunas do TAPI, validadas com pandera; `media_ref` e `channel_stats_at_collection`
podem ficar vazias): post_id, channel_id, published_at, caption, media_ref, metric_target,
channel_stats_at_collection (informativo, nunca é feature: é do dia da coleta) e sector (moda, beleza ou
automotivo). Opcionais: description, tags (lista ou texto separado por "|"), channel_type (marca/criador),
duration_s, category_id e collected_at (data da coleta; sem ela, vale a data passada em `coletado_em` ou hoje).

Protocolo (o mesmo do modelo principal):
1. só posts com 30 dias ou mais na coleta viram alvo (métrica estabilizada), e o histórico do canal usa só
   posts publicados 30 dias ou mais antes de cada alvo (regra 2 de docs/FEATURES.md);
2. alvo relativo ao canal: log1p(métrica) − log1p(mediana dos últimos 20 posts maduros); exige 5 de histórico;
3. split temporal 70/15/15 pelos quantis da data (sem validação cruzada aleatória);
4. LightGBM L1 com os hiperparâmetros padrão do projeto e early stopping temporal nos últimos 10% do treino;
   quantis de 10% e 90% calibrados por CQR na validação; limiares do confidence_flag na validação;
5. avaliação única no teste do cliente contra a mediana do canal e a regressão linear, com IC 95% por
   bootstrap de canais; depois o modelo é reajustado em treino + validação (mesmo número de árvores);
6. artefato em models/clientes/<nome>.joblib (+ .json com o relatório), no formato que o agente já lê.
Features: bloco A (histórico do canal, tempo e metadados), bloco B (texto: regras + embedding com PCA) e, com
`--midia`, o bloco C da imagem de capa a partir de `media_ref`, com a Tool 1 do agente (as mesmas funções).
Vídeo e VLM ficam fora do treino dinâmico: são lentos e limitados por cota (ver README).
"""

import argparse
import json
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa

from galaxies_tool import artefato, modelo
from galaxies_tool.contrato import FUSO, RelatorioTreino
from galaxies_tool.features import (
    TETO_N_HISTORICO,
    atividade_de_postagem,
    features_tempo,
    features_texto,
    historico_do_alvo,
)
from galaxies_tool.metricas import avaliar, bootstrap_diferenca
from galaxies_tool.regras import IDADE_MINIMA_DIAS

PASTA = Path("models/clientes")
EXEMPLO = Path("data/exemplos/base_cliente_marcas.parquet")
MIN_HISTORICO = 5
MIN_ALVOS = 300  # abaixo disso, o teste do cliente teria poucas dezenas de posts
LIMITE_VIES = 0.15  # |viés mediano| em log acima disso (~16% no número) vira aviso de deriva
QUANTIS_CORTE = (0.70, 0.85)
NOME_VALIDO = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
SETORES = ["moda", "beleza", "automotivo"]
COLUNAS_TAPI = [
    "post_id",
    "channel_id",
    "published_at",
    "caption",
    "media_ref",
    "metric_target",
    "channel_stats_at_collection",
    "sector",
]

ESQUEMA = pa.DataFrameSchema(
    {
        "post_id": pa.Column(str, unique=True),
        "channel_id": pa.Column(str),
        "published_at": pa.Column("datetime64[ns, UTC]"),
        "caption": pa.Column(str),
        "media_ref": pa.Column(str, nullable=True),
        "metric_target": pa.Column(int, pa.Check.ge(0)),
        "channel_stats_at_collection": pa.Column(float, nullable=True),
        "sector": pa.Column(str, pa.Check.isin(SETORES)),
        "description": pa.Column(str, required=False),
        "channel_type": pa.Column(str, pa.Check.isin(["marca", "criador"]), required=False),
        "duration_s": pa.Column(float, nullable=True, required=False),
        "category_id": pa.Column(str, nullable=True, required=False),
    },
    strict=False,
)


def caminho(nome: str) -> Path:
    if not NOME_VALIDO.fullmatch(nome):
        raise ValueError("nome do modelo: só letras minúsculas, números, '-' e '_' (até 64)")
    return PASTA / f"{nome}.joblib"


# ---------------------------------------------------------------- leitura e validação


def _tags(v) -> list[str]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    if isinstance(v, str):
        return [t.strip() for t in v.split("|") if t.strip()]
    return [str(t) for t in v]


def ler_base(origem: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Lê CSV ou Parquet no esquema mínimo e valida. Data sem fuso = horário de Brasília (como no contrato)."""
    df = origem.copy() if isinstance(origem, pd.DataFrame) else (
        pd.read_parquet(origem) if str(origem).endswith(".parquet") else pd.read_csv(origem, dtype=str)
    )
    faltam = [c for c in COLUNAS_TAPI if c not in df.columns]
    if faltam:
        raise ValueError(f"colunas do esquema mínimo ausentes: {faltam} (podem existir vazias)")
    pub = pd.to_datetime(df.published_at)
    if pub.dt.tz is None:
        pub = pub.dt.tz_localize(FUSO)
    df["published_at"] = pub.dt.tz_convert("UTC").dt.as_unit("ns")
    for c in ("post_id", "channel_id", "caption", "sector", "description", "channel_type", "category_id"):
        if c in df:
            df[c] = df[c].astype(object).where(df[c].notna(), "" if c in ("caption", "description") else None)
            df[c] = df[c].map(lambda v: v if v is None else str(v))
    df["media_ref"] = df.media_ref.astype(object).where(df.media_ref.notna() & (df.media_ref != ""), None)
    df["metric_target"] = pd.to_numeric(df.metric_target).astype(int)
    df["channel_stats_at_collection"] = pd.to_numeric(df.channel_stats_at_collection, errors="coerce")
    if "duration_s" in df:
        df["duration_s"] = pd.to_numeric(df.duration_s, errors="coerce")
    df["tags"] = df.tags.map(_tags) if "tags" in df else [[] for _ in range(len(df))]
    if "description" not in df:
        df["description"] = ""
    return ESQUEMA.validate(df)


# ---------------------------------------------------------------- features


def _embeddings_texto(textos: list[str]) -> np.ndarray:
    """Embeddings do bloco B num processo à parte (PyTorch e LightGBM não convivem no macOS; ver extracao)."""
    with tempfile.TemporaryDirectory() as tmp:
        entrada, saida = Path(tmp) / "textos.json", Path(tmp) / "emb.npy"
        entrada.write_text(json.dumps(textos, ensure_ascii=False))
        subprocess.run(
            [sys.executable, "-m", "galaxies_tool.embeddings_texto", str(entrada), str(saida)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        return np.load(saida).astype(np.float32)


def _midia(alvos: pd.DataFrame) -> pd.DataFrame:
    """Bloco C da imagem de capa com a Tool 1 do agente, a partir de media_ref (URL do YouTube ou imagem)."""
    from galaxies_tool import extracao

    linhas = []
    for r in alvos.itertuples():
        atr = {}
        if r.media_ref:
            try:
                atr = extracao.extrair_isolado(r.media_ref, r.caption, r.description, r.tags, False)
            except Exception as e:  # noqa: BLE001 — mídia inacessível: o post fica sem o bloco C
                print(f"{r.video_id}: mídia não analisada ({e})", flush=True)
        linhas.append({k: v for k, v in atr.items() if k.startswith(("thumb_", "embimg_"))})
    return pd.DataFrame(linhas, index=alvos.index)


def montar(df: pd.DataFrame, coletado_em: datetime, usar_midia: bool = False) -> tuple[pd.DataFrame, dict]:
    """Uma linha por post-alvo (maduro e com histórico mínimo), com as colunas do treino principal."""
    d = df.rename(columns={"post_id": "video_id", "metric_target": "likes"}).reset_index(drop=True)
    coleta = pd.Timestamp(coletado_em)
    coleta = coleta.tz_localize(FUSO) if coleta.tz is None else coleta  # sem fuso = Brasília, como no contrato
    maduro = d.published_at <= coleta - timedelta(days=IDADE_MINIMA_DIAS)
    funil = {"posts": len(d), "maduros": int(maduro.sum())}
    alvos = d[maduro].copy()
    hist = historico_do_alvo(d[["channel_id", "published_at", "likes"]], alvos)
    ativ = atividade_de_postagem(d[["channel_id", "published_at"]], alvos)
    texto = pd.DataFrame(
        [features_texto(c, s, t) for c, s, t in zip(alvos.caption, alvos.description, alvos.tags, strict=True)],
        index=alvos.index,
    )
    out = pd.concat([alvos, hist, ativ, features_tempo(alvos.published_at), texto], axis=1)
    out = out[out.n_hist >= MIN_HISTORICO].copy()
    funil["com_historico_minimo"] = len(out)
    out["n_shorts_anteriores"] = out.n_hist.clip(upper=TETO_N_HISTORICO)
    out["n_tags"] = out.tags.map(len)
    out["alvo"] = np.log1p(out.likes) - out.log_mediana_likes_ult20
    out["setor"] = out.sector
    out["duracao_s"] = out.duration_s if "duration_s" in out else np.nan
    out["tipo_canal"] = out.channel_type if "channel_type" in out else None
    out["categoria_id"] = out.category_id if "category_id" in out else None
    if usar_midia:
        midia = _midia(out)
        out = pd.concat([out, midia], axis=1)
    emb = _embeddings_texto((out.caption + "\n" + out.description).tolist())
    out = pd.concat([out, pd.DataFrame(emb, index=out.index, columns=[f"embtxt_{i}" for i in range(768)])], axis=1)
    for c in ("setor", "tipo_canal", "categoria_id"):
        out[c] = out[c].astype("category")
    return out.reset_index(drop=True), funil


def colunas(treino: pd.DataFrame, usar_midia: bool) -> list[str]:
    """Colunas do modelo do cliente: as do modelo principal que a base permite calcular."""
    a = [c for c in modelo.BLOCO_A if c in treino and treino[c].notna().any()]
    meses = (treino.published_at.max() - treino.published_at.min()).days / 30.44
    if meses < 12:  # mesma regra do modelo principal: sem 12 meses, "mês" confunde sazonalidade com deriva
        a.remove("mes")
    cols = a + modelo.BLOCO_B_MANUAL + modelo.nomes_pca("emb_texto", 16)
    if usar_midia and treino.filter(like="embimg_").notna().any(axis=None):
        cols += [c for c in modelo.BLOCO_C_THUMB if c in treino] + modelo.nomes_pca("emb_thumb", 16)
    return cols


# ---------------------------------------------------------------- treino e avaliação


def _cortes(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    return tuple(df.published_at.quantile(q).normalize() for q in QUANTIS_CORTE)


def _fmt(bd: dict, k: str) -> str:
    return modelo._fmt_delta(bd, k, 4 if k == "mae_log" else 3)


def treinar_modelo_cliente(
    base: str | Path | pd.DataFrame,
    nome: str,
    usar_midia: bool = False,
    coletado_em: datetime | None = None,
) -> RelatorioTreino:
    """Treina, calibra, avalia e salva um modelo especialista para a base de um cliente (ver docstring)."""
    destino = caminho(nome)
    df = ler_base(base)
    if coletado_em is None:
        coletado_em = (
            pd.to_datetime(df.collected_at, utc=True).min().to_pydatetime()
            if "collected_at" in df
            else datetime.now(UTC)
        )
    avisos = []
    if df.channel_stats_at_collection.notna().any():
        avisos.append("channel_stats_at_collection é do dia da coleta: guardado, mas nunca usado como feature")
    dados, funil = montar(df, coletado_em, usar_midia)
    if len(dados) < MIN_ALVOS:
        raise ValueError(
            f"só {len(dados)} posts com 30 dias na coleta e 5 de histórico no canal (mínimo: {MIN_ALVOS})"
        )
    c1, c2 = _cortes(dados)
    treino = dados[dados.published_at < c1].copy()
    valid = dados[(dados.published_at >= c1) & (dados.published_at < c2)].copy()
    teste = dados[dados.published_at >= c2].copy()
    cols = colunas(treino, usar_midia)
    if dados.channel_id.nunique() < 5:
        avisos.append("menos de 5 canais: a ordenação dentro do canal é medida em poucos grupos")

    # desenvolvimento: treino → calibração na validação → avaliação única no teste
    params = dict(modelo.PARAMS_LGBM)
    dev = artefato.calibrar(artefato.treinar(treino, params, colunas=cols, com_catboost=False), valid)
    x = artefato.preparar(teste, dev)
    p = artefato.prever(dev, x)
    base_lin = modelo.baselines(treino, teste)
    def aval(prev, baixo=None, alto=None):
        return avaliar(teste.alvo, prev, teste.channel_id, teste.log_mediana_likes_ult20, baixo, alto,
                       datas=teste.published_at)

    met = {"modelo": aval(p.alvo_previsto, p.alvo_baixo, p.alvo_alto)}
    # viés = mediana de (previsto − real): perto de 0 é o esperado com perda L1; longe de 0 indica deriva
    xv = artefato.preparar(valid, dev)
    vies = {
        "validacao": float(np.median(artefato.prever_ponto(dev, xv) - valid.alvo)),
        "teste": float(np.median(p.alvo_previsto - teste.alvo)),
    }
    met["modelo"]["vies_mediano"] = vies["teste"]
    if abs(vies["teste"]) > LIMITE_VIES:
        avisos.append(
            f"deriva: no teste, o modelo {'super' if vies['teste'] > 0 else 'sub'}estima o nível em "
            f"~{abs(np.expm1(vies['teste'])) * 100:.0f}% (mediana; na validação, {np.expm1(vies['validacao']) * 100:+.0f}%). "
            "O desempenho relativo dos posts mudou com o tempo: use o modelo para ORDENAR criativos e "
            "retreine com dados recentes"
        )
    met.update({nome_b: aval(pb) for nome_b, pb in base_lin.items()})
    bd = bootstrap_diferenca(teste.alvo, p.alvo_previsto.to_numpy(), base_lin["baseline_mediana_canal"],
                             teste.channel_id, teste.published_at)
    shap = pd.Series(np.abs(artefato.contribuicoes(dev, x)[:, :-1]).mean(axis=0), index=dev["colunas"])
    from galaxies_tool.previsao import DESCRICOES, grupo

    fatores = shap.groupby(grupo).sum().sort_values(ascending=False)

    # final: reajuste em treino + validação com o mesmo número de árvores, calibração e limiares do passo acima
    final = pd.concat([treino, valid])
    art = artefato.treinar(final, params, dev["arvores"], colunas=cols, com_catboost=False)
    art["cqr"], art["calibracao"] = dev["cqr"], dev["calibracao"]
    art["limiares"].update({k: v for k, v in dev["limiares"].items() if k not in art["limiares"]})
    agora = datetime.now(UTC).isoformat(timespec="seconds")
    art.update({"versao": f"cliente_{nome}", "tipo": "cliente", "dados_treino": "treino + validacao do cliente",
                "n_treino": len(final), "criado_em": agora})
    relatorio = RelatorioTreino(
        nome=nome,
        modelo=str(destino),
        criado_em=agora,
        n_posts=len(df),
        n_canais=int(df.channel_id.nunique()),
        funil=funil,
        periodo=[str(dados.published_at.min().date()), str(dados.published_at.max().date())],
        cortes={"treino_validacao": str(c1.date()), "validacao_teste": str(c2.date())},
        n_por_split={"treino": len(treino), "validacao": len(valid), "teste": len(teste)},
        features=cols,
        metricas_teste={k: {m: (None if pd.isna(v) else float(v)) for m, v in r.items()}
                        for k, r in met.items()},
        ganho_vs_mediana_do_canal={
            k: _fmt(bd, k) if not np.isnan(bd[k]["diferenca"]) else "indefinido (o baseline empata tudo)"
            for k in bd
        },
        principais_fatores=[DESCRICOES.get(g, g) for g in fatores.index[:8]],
        cobertura_crua_na_validacao=dev["calibracao"]["cobertura_crua"],
        avisos=avisos,
    )
    art["relatorio"] = relatorio.model_dump()
    PASTA.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(art, destino, compress=3)
    destino.with_suffix(".json").write_text(relatorio.model_dump_json(indent=2))
    _registrar(relatorio)
    return relatorio


def _registrar(r: RelatorioTreino):
    import mlflow

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("galaxies_clientes")
    with mlflow.start_run(run_name=r.nome):
        mlflow.set_tags({"avaliado_em": "teste do cliente", **r.cortes})
        mlflow.log_params({"n_features": len(r.features), "n_posts": r.n_posts, "n_canais": r.n_canais})
        mlflow.log_metrics({f"{k}_{m}": v for k, d in r.metricas_teste.items() for m, v in d.items()
                            if v is not None and not np.isnan(v)})
        mlflow.log_text(r.model_dump_json(indent=2), "relatorio.json")


# ---------------------------------------------------------------- base de exemplo


def exportar_exemplo(destino: Path = EXEMPLO) -> Path:
    """Os canais de MARCA da base_v1 (o perfil de cliente da Galaxies) no esquema mínimo do TAPI."""
    b = pd.read_parquet("data/processed/base_v1.parquet")
    b = b[(b.channel_type == "marca") & ~b.is_demo_channel]
    ex = pd.DataFrame({
        "post_id": b.video_id,
        "channel_id": b.channel_id,
        "published_at": b.published_at,
        "caption": b.title,
        "media_ref": b.url,
        "metric_target": b.likes,
        "channel_stats_at_collection": b.channel_subscribers_at_collection.astype(float),
        "sector": b.sector,
        "description": b.description,
        "tags": b.tags.map(lambda t: "|".join(t) if t is not None else ""),
        "channel_type": b.channel_type,
        "duration_s": b.duration_s.astype(float),
        "category_id": b.category_id,
        "collected_at": b.collected_at,
    })
    destino.parent.mkdir(parents=True, exist_ok=True)
    ex.to_parquet(destino, index=False)
    print(f"{len(ex)} posts de {ex.channel_id.nunique()} canais de marca → {destino}")
    return destino


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("acao", choices=["treinar", "exemplo"])
    ap.add_argument("base", nargs="?")
    ap.add_argument("--nome")
    ap.add_argument("--midia", action="store_true", help="bloco C da imagem de capa a partir de media_ref")
    args = ap.parse_args()
    if args.acao == "exemplo":
        exportar_exemplo()
        return
    r = treinar_modelo_cliente(args.base, args.nome, args.midia)
    print(r.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
