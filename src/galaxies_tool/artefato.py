"""Treina e salva o artefato do modelo que o agente usa: LightGBM pontual, quantis de 10% e 90% e os PCAs.

Uso: uv run python -m galaxies_tool.artefato [--dados treino|treino+validacao] [--versao NOME]
Saída: models/<versao>.joblib (artefato) e models/<versao>.json (metadados legíveis: features, limiares e
métricas na validação).

Por padrão treina só no treino e reporta a validação (artefato de desenvolvimento). Com
`--dados treino+validacao`, as métricas da validação deixam de ser honestas e não são reportadas: é o
modelo que vai para o agente depois que as escolhas estão congeladas.

Mesmos blocos, features e hiperparâmetros de `modelo.py` (o conjunto mais completo disponível).
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from galaxies_tool import modelo
from galaxies_tool.metricas import avaliar, cobertura

PASTA = Path("models")
QUANTIS = (0.10, 0.90)
CATEGORICAS = ("setor", "tipo_canal", "categoria_id")
MIN_HISTORICO_CONFIAVEL = 10  # menos Shorts que isso no histórico: a mediana do canal é ruidosa
PERCENTIL_INSTAVEL = 90  # canal mais instável que 90% do treino (std das curtidas recentes)


def colunas_do_modelo(df: pd.DataFrame) -> list[str]:
    """O conjunto mais completo já avaliado em modelo.py (A + B + C da thumbnail)."""
    thumb = [c for c in modelo.BLOCO_C_THUMB if c in df.columns and df[c].notna().mean() > 0.99]
    return (
        modelo.BLOCO_A
        + modelo.BLOCO_B_MANUAL
        + modelo.nomes_pca("emb_texto")
        + thumb
        + modelo.nomes_pca("emb_thumb")
    )


def treinar(treino: pd.DataFrame) -> dict:
    """Ajusta PCAs e os três modelos no `treino`. Retorna o artefato (sem métricas)."""
    pcas = {}
    for prefixo, nome in (("embtxt_", "emb_texto"), ("embimg_", "emb_thumb")):
        cols_emb = [c for c in treino.columns if c.startswith(prefixo)]
        pca = modelo.PCA(n_components=modelo.N_PCA, random_state=modelo.SEMENTE).fit(
            treino[cols_emb].dropna()
        )
        nomes = modelo.nomes_pca(nome)
        treino = modelo.aplicar_pca(pca, treino, cols_emb, nomes)
        pcas[nome] = {"pca": pca, "colunas": cols_emb, "nomes": nomes}
    cols = colunas_do_modelo(treino)
    ponto, n_ponto = modelo.treinar_lgbm(treino, cols)
    quantis = {}
    for q in QUANTIS:
        params = {**modelo.PARAMS_LGBM, "objective": "quantile", "alpha": q}
        quantis[q], _ = modelo.treinar_lgbm(treino, cols, params, metrica="quantile")
    return {
        "colunas": cols,
        "categorias": {c: list(treino[c].cat.categories) for c in CATEGORICAS},
        "pcas": pcas,
        "ponto": ponto,
        "n_arvores": n_ponto,
        "quantis": quantis,
        "limiares": {
            "min_historico": MIN_HISTORICO_CONFIAVEL,
            "std_instavel": float(np.nanpercentile(treino.std_log_likes_ult20, PERCENTIL_INSTAVEL)),
        },
    }


def preparar(df: pd.DataFrame, art: dict) -> pd.DataFrame:
    """Aplica os PCAs do artefato e devolve a matriz com as colunas e categorias do treino."""
    for p in art["pcas"].values():
        df = modelo.aplicar_pca(p["pca"], df, p["colunas"], p["nomes"])
    x = modelo.matriz(df, art["colunas"])
    for c, cats in art["categorias"].items():
        if c in x.columns:
            x[c] = pd.Categorical(x[c].astype(object), categories=cats)
    return x


def prever(art: dict, x: pd.DataFrame) -> pd.DataFrame:
    """Alvo relativo previsto e intervalo de 80%. Os quantis são ordenados para nunca cruzar o ponto."""
    ponto = art["ponto"].predict(x)
    baixo, alto = (art["quantis"][q].predict(x) for q in QUANTIS)
    baixo, alto = np.minimum(baixo, ponto), np.maximum(alto, ponto)
    return pd.DataFrame({"alvo_previsto": ponto, "alvo_baixo": baixo, "alvo_alto": alto}, index=x.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dados", choices=["treino", "treino+validacao"], default="treino")
    ap.add_argument("--versao", default=None)
    args = ap.parse_args()
    df = modelo.carregar()
    splits = ["treino"] if args.dados == "treino" else ["treino", "validacao"]
    treino = df[df.split.isin(splits)].copy()
    art = treinar(treino)

    metricas = {}
    if args.dados == "treino":
        valid = df[df.split == "validacao"].copy()
        p = prever(art, preparar(valid, art))
        metricas = avaliar(
            valid.alvo,
            p.alvo_previsto,
            valid.channel_id,
            valid.log_mediana_likes_ult20,
            datas=valid.published_at,
        )
        metricas["cobertura_80"] = cobertura(valid.alvo, p.alvo_baixo, p.alvo_alto)
        metricas["largura_media_intervalo"] = float((p.alvo_alto - p.alvo_baixo).mean())
        art["limiares"]["largura_larga"] = float(np.percentile(p.alvo_alto - p.alvo_baixo, 90))
    else:
        p = prever(art, preparar(treino, art))
        art["limiares"]["largura_larga"] = float(np.percentile(p.alvo_alto - p.alvo_baixo, 90))

    versao = args.versao or f"lgbm_{'dev' if args.dados == 'treino' else 'final'}"
    art.update(
        {
            "versao": versao,
            "dados": args.dados,
            "n_treino": len(treino),
            "criado_em": datetime.now(UTC).isoformat(timespec="seconds"),
            "metricas_validacao": metricas,
        }
    )
    PASTA.mkdir(exist_ok=True)
    joblib.dump(art, PASTA / f"{versao}.joblib", compress=3)
    meta = {k: v for k, v in art.items() if k not in ("pcas", "ponto", "quantis")}
    (PASTA / f"{versao}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str))

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("galaxies_validacao")
    with mlflow.start_run(run_name=f"artefato_{versao}"):
        mlflow.set_tag("avaliado_em", "validacao" if metricas else "nenhum (treino+validacao)")
        mlflow.log_params(
            {"dados": args.dados, "n_features": len(art["colunas"]), "n_arvores": art["n_arvores"]}
        )
        mlflow.log_metrics({k: v for k, v in metricas.items() if isinstance(v, float) and not np.isnan(v)})
        mlflow.log_artifact(str(PASTA / f"{versao}.json"))
    print(
        f"{versao}: {len(art['colunas'])} features, {art['n_arvores']} árvores, "
        f"treino com {len(treino)} vídeos"
    )
    for k in (
        "mae_log",
        "spearman_intra_canal",
        "acuracia_par_a_par_7d",
        "cobertura_80",
        "largura_media_intervalo",
    ):
        if k in metricas:
            print(f"  {k}: {metricas[k]:.4f}")


if __name__ == "__main__":
    main()
