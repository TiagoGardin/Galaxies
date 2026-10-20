"""Treina, calibra e salva o artefato que o agente usa (e que a avaliação final mede no teste).

Uso: uv run python -m galaxies_tool.artefato [--versao lgbm_final]
Saída: models/<versao>.joblib (artefato) e models/<versao>.json (metadados legíveis: features, limiares,
calibração e métricas na validação).

Protocolo (aulas 3.4 e 5.3 da trilha; docs/EXPERIMENTO.md):
1. desenvolvimento: PCAs, previsão pontual (média de LightGBM e CatBoost, L1) e quantis de 10% e 90% do
   LightGBM ajustados só no split de treino;
2. calibração = validação, a fatia mais recente antes do teste: nela, o modelo de desenvolvimento (que não a
   viu) dá a correção conformal dos quantis (CQR, Romano et al. 2019), os limiares do confidence_flag e as
   métricas de validação. Nada disso olha o teste;
3. final: o mesmo procedimento reajustado em treino + validação (o modelo não perde os meses mais
   recentes), com a correção e os limiares do passo 2, que estimam o comportamento fora da amostra (o limiar
   de distância é medido no espaço padronizado do modelo de desenvolvimento e aplicado no do final: uma
   aproximação, conferida no teste, onde a regra disparou em 1% dos vídeos, como o p99 prevê);
4. o artefato final é avaliado uma única vez no teste (avaliacao_final.py) e é o que o agente serve.

confidence_flag: regras explícitas, cada uma com um código estável e uma frase (aula 5.4). Os limiares são
calculados aqui, uma vez, e versionados no artefato.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from galaxies_tool import modelo
from galaxies_tool.metricas import avaliar, cobertura

PASTA = Path("models")
VERSAO = "lgbm_final"
QUANTIS = (0.10, 0.90)
NIVEL = 0.80
MIN_FOLHA_QUANTIL = 100  # quantis extremos precisam de folhas maiores (aula 5.2)
K_VIZINHOS = 10
ESPACO_OOD = ("emb_texto", "emb_thumb")  # PCAs do texto e da thumbnail, padronizados
MIN_HISTORICO_CONFIAVEL = 10  # menos Shorts maduros que isso: a mediana do canal é ruidosa (aula 5.4)
PERCENTIL_INSTAVEL = 90  # canal mais instável que 90% do treino
PERCENTIL_LARGURA = 90  # intervalo mais largo que 90% da calibração
PERCENTIL_OOD = 99  # distância ao 10º vizinho acima do p99 da calibração
MAX_DIAS_FUTURO = 180  # publicação planejada muito depois do fim dos dados
MAX_DIAS_PASSADO = 365

TEXTOS_MOTIVO = {
    "canal_pouco_historico": "o canal tem só {n} Shorts maduros no histórico (o mínimo confiável é {min})",
    "canal_instavel": "as curtidas recentes do canal são muito instáveis (entre as 10% mais instáveis)",
    "intervalo_largo": "o intervalo é largo: o limite de cima é {razao} vezes o de baixo",
    "fora_da_distribuicao": "o criativo é diferente dos Shorts que o modelo viu no treino",
    "setor_fora_do_treino": "o setor '{setor}' não aparece nos dados de treino (só moda e beleza)",
    "extracao_incompleta": "parte do criativo não pôde ser analisada ({o_que}); a previsão usa só o restante",
    "data_fora_do_periodo": "a publicação planejada está a {dias} dias do fim dos dados de treino",
}


def _br(x: float) -> str:
    return f"{x:.1f}".replace(".", ",")


# ---------------------------------------------------------------- treino


def treinar(treino: pd.DataFrame, params: dict | None = None, arvores: dict | None = None) -> dict:
    """Ajusta PCAs, os modelos e o índice de vizinhos no `treino`. Retorna o artefato sem calibração.

    Ponto = média de LightGBM e CatBoost (L1), escolhida na validação (docs/EXPERIMENTO.md); intervalo =
    quantis do LightGBM. `arvores` ({"ponto": n, "catboost": n, 0.1: n, 0.9: n}): número de árvores fixo
    (refit), sem novo early stopping."""
    params = params or modelo.params_lgbm()
    arvores = arvores or {}
    pcas = modelo.ajustar_pcas(treino)
    treino = modelo.aplicar_pcas(treino, pcas)
    cols = modelo.colunas_completas(treino)
    ponto, n_ponto = modelo.treinar_lgbm(treino, cols, params, n_arvores=arvores.get("ponto"))
    catboost, n_cat = modelo.treinar_catboost(treino, cols, arvores.get("catboost"))
    quantis, n_arvores = {}, {"ponto": n_ponto, "catboost": n_cat}
    for q in QUANTIS:
        pq = {
            **params,
            "objective": "quantile",
            "alpha": q,
            "min_child_samples": max(params["min_child_samples"], MIN_FOLHA_QUANTIL),
        }
        quantis[q], n_arvores[q] = modelo.treinar_lgbm(treino, cols, pq, metrica="quantile",
                                                       n_arvores=arvores.get(q))
    ood = [c for p in ESPACO_OOD for c in pcas[p]["nomes"]]
    z = treino[ood].dropna()
    escala = StandardScaler().fit(z)
    return {
        "colunas": cols,
        "categorias": {c: list(treino[c].cat.categories) for c in modelo.CATEGORICAS if c in cols},
        "pcas": pcas,
        "params": params,
        "ponto": ponto,
        "ponto_catboost": catboost,
        "n_arvores": n_ponto,
        "arvores": n_arvores,
        "quantis": quantis,
        "ood": {
            "colunas": ood,
            "escala": escala,
            "vizinhos": NearestNeighbors(n_neighbors=K_VIZINHOS).fit(escala.transform(z)),
        },
        "setores_treino": sorted(treino.setor.astype(str).unique()),
        "periodo_treino": [str(treino.published_at.min().date()), str(treino.published_at.max().date())],
        "limiares": {
            "min_historico": MIN_HISTORICO_CONFIAVEL,
            "std_instavel": float(np.nanpercentile(treino.std_log_likes_ult20, PERCENTIL_INSTAVEL)),
        },
        "cqr": 0.0,
    }


def preparar(df: pd.DataFrame, art: dict) -> pd.DataFrame:
    """Aplica os PCAs do artefato e devolve a matriz com as colunas e categorias do treino."""
    df = modelo.aplicar_pcas(df, art["pcas"])
    x = modelo.matriz(df, art["colunas"])
    for c, cats in art["categorias"].items():
        x[c] = pd.Categorical(x[c].astype(object), categories=cats)
    return x


def _x_catboost(x: pd.DataFrame) -> pd.DataFrame:
    x = x.copy()
    for c in x.columns:
        if c in modelo.CATEGORICAS:
            x[c] = modelo.como_texto(x[c])
    return x


def prever_ponto(art: dict, x: pd.DataFrame) -> np.ndarray:
    """Alvo relativo previsto: média do LightGBM e do CatBoost."""
    return (art["ponto"].predict(x) + art["ponto_catboost"].predict(_x_catboost(x))) / 2


def contribuicoes(art: dict, x: pd.DataFrame) -> np.ndarray:
    """SHAP exato da média: média do TreeSHAP do LightGBM e do CatBoost (cada um aditivo, logo a média
    também). Uma linha por vídeo; a última coluna é o valor base. Base + soma = prever_ponto."""
    from catboost import Pool

    lgbm = art["ponto"].predict(x, pred_contrib=True)
    xc = _x_catboost(x)
    cats = [c for c in xc.columns if c in modelo.CATEGORICAS]
    cat = art["ponto_catboost"].get_feature_importance(Pool(xc, cat_features=cats), type="ShapValues")
    return (lgbm + cat) / 2


def prever(art: dict, x: pd.DataFrame) -> pd.DataFrame:
    """Alvo relativo previsto e intervalo de 80% calibrado (CQR). Os limites são ordenados para nunca
    cruzarem o ponto (rearrangement, aula 5.2)."""
    ponto = prever_ponto(art, x)
    baixo, alto = (art["quantis"][q].predict(x) for q in QUANTIS)
    baixo, alto = baixo - art["cqr"], alto + art["cqr"]
    baixo, alto = np.minimum(baixo, ponto), np.maximum(alto, ponto)
    return pd.DataFrame({"alvo_previsto": ponto, "alvo_baixo": baixo, "alvo_alto": alto}, index=x.index)


def distancia_ood(art: dict, x: pd.DataFrame) -> np.ndarray:
    """Distância ao 10º vizinho do treino no espaço padronizado dos PCAs (NaN se faltar embedding)."""
    o = art["ood"]
    d = np.full(len(x), np.nan)
    z = x.reindex(columns=o["colunas"])
    ok = z.notna().all(axis=1).to_numpy()
    if ok.any():
        d[ok] = o["vizinhos"].kneighbors(o["escala"].transform(z[ok]))[0][:, -1]
    return d


# ---------------------------------------------------------------- calibração e confidence_flag


def calibrar(art: dict, calib: pd.DataFrame) -> dict:
    """Correção conformal dos quantis e limiares do flag, medidos na calibração (validação)."""
    x = preparar(calib, art)
    baixo, alto = (art["quantis"][q].predict(x) for q in QUANTIS)
    y = calib.alvo.to_numpy()
    scores = np.sort(np.maximum(baixo - y, y - alto))  # CQR: quanto o real ficou fora (+) ou dentro (−)
    n = len(scores)
    k = int(np.ceil((n + 1) * NIVEL))
    art["cqr"] = float(scores[k - 1])
    p = prever(art, x)
    art["limiares"].update(
        {
            "largura_larga": float(np.percentile(p.alvo_alto - p.alvo_baixo, PERCENTIL_LARGURA)),
            "distancia_ood": float(np.nanpercentile(distancia_ood(art, x), PERCENTIL_OOD)),
            "ultima_data_vista": str(calib.published_at.max().date()),
            "max_dias_futuro": MAX_DIAS_FUTURO,
            "max_dias_passado": MAX_DIAS_PASSADO,
        }
    )
    art["calibracao"] = {
        "conjunto": "validacao",
        "n": n,
        "nivel": NIVEL,
        "periodo": [str(calib.published_at.min().date()), str(calib.published_at.max().date())],
        "cobertura_crua": float(cobertura(y, np.minimum(baixo, alto), np.maximum(baixo, alto))),
    }
    return art


def motivos(art: dict, linha: dict, largura: float, dist: float, extracao: list[str]) -> list[dict]:
    """Motivos de baixa confiança de UM criativo: lista de {codigo, texto}. Lista vazia = confiança normal.

    `linha`: features do criativo (n_historico_total, std_log_likes_ult20, setor, published_at);
    `extracao`: o que não pôde ser extraído (ex.: ["vídeo", "rubrica do VLM"])."""
    lim, saida = art["limiares"], []

    def add(codigo, **kw):
        saida.append({"codigo": codigo, "texto": TEXTOS_MOTIVO[codigo].format(**kw)})

    n_hist = int(linha.get("n_historico_total", linha.get("n_shorts_anteriores", 0)))
    if n_hist < lim["min_historico"]:
        add("canal_pouco_historico", n=n_hist, min=lim["min_historico"])
    std = linha.get("std_log_likes_ult20")
    if std is not None and not pd.isna(std) and std > lim["std_instavel"]:
        add("canal_instavel")
    if largura > lim["largura_larga"]:
        add("intervalo_largo", razao=_br(float(np.exp(largura))))
    if not np.isnan(dist) and dist > lim["distancia_ood"]:
        add("fora_da_distribuicao")
    if str(linha.get("setor")) not in art["setores_treino"]:
        add("setor_fora_do_treino", setor=linha.get("setor"))
    if extracao:
        add("extracao_incompleta", o_que=", ".join(extracao))
    publicado = pd.Timestamp(linha["published_at"])
    dias = (publicado.tz_convert("UTC").normalize() - pd.Timestamp(lim["ultima_data_vista"], tz="UTC")).days
    if dias > lim["max_dias_futuro"] or dias < -lim["max_dias_passado"]:
        add("data_fora_do_periodo", dias=abs(dias))
    return saida


def faltando_na_extracao(linha: dict) -> list[str]:
    """O que falta num vídeo da base (para a avaliação): thumbnail, vídeo baixado ou rubrica do VLM."""
    falta = []
    if pd.isna(linha.get("embimg_0", np.nan)):
        falta.append("imagem")
    if linha.get("esperava_video") and pd.isna(linha.get("cortes_por_segundo", np.nan)):
        falta.append("vídeo")
    return falta


def motivos_em_lote(art: dict, df: pd.DataFrame, x: pd.DataFrame, p: pd.DataFrame) -> list[list[dict]]:
    larg = (p.alvo_alto - p.alvo_baixo).to_numpy()
    dist = distancia_ood(art, x)
    linhas = df.to_dict("records")
    return [motivos(art, r, larg[i], dist[i], faltando_na_extracao(r)) for i, r in enumerate(linhas)]


# ---------------------------------------------------------------- principal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--versao", default=VERSAO)
    args = ap.parse_args()
    df = modelo.carregar()
    treino, valid = df[df.split == "treino"].copy(), df[df.split == "validacao"].copy()
    dev = calibrar(treinar(treino), valid)

    x = preparar(valid, dev)
    p = prever(dev, x)
    metricas = avaliar(
        valid.alvo,
        p.alvo_previsto,
        valid.channel_id,
        valid.log_mediana_likes_ult20,
        p.alvo_baixo,
        p.alvo_alto,
        datas=valid.published_at,
    )
    metricas["largura_media_intervalo"] = float((p.alvo_alto - p.alvo_baixo).mean())
    flags = motivos_em_lote(dev, valid.assign(esperava_video=valid.tem_video), x, p)
    metricas["taxa_baixa_confianca"] = float(np.mean([bool(m) for m in flags]))
    print(
        f"desenvolvimento: {dev['n_arvores']} árvores; CQR = {dev['cqr']:+.4f} (cobertura crua na "
        f"calibração: {dev['calibracao']['cobertura_crua']:.3f})",
        flush=True,
    )

    final = pd.concat([treino, valid])
    # refit: mesmos hiperparâmetros e mesmo número de árvores escolhidos na validação (aula 3.4)
    art = treinar(final, dev["params"], dev["arvores"])
    art["cqr"], art["calibracao"] = dev["cqr"], dev["calibracao"]
    art["limiares"].update({k: v for k, v in dev["limiares"].items() if k not in art["limiares"]})
    art.update(
        {
            "versao": args.versao,
            "dados_treino": "treino + validacao",
            "n_treino": len(final),
            "criado_em": datetime.now(UTC).isoformat(timespec="seconds"),
            "versao_base": modelo.VERSAO_BASE,
            "cortes": modelo.contexto_registro(),
            "metricas_validacao": metricas,
            "nota_metricas_validacao": "do modelo de desenvolvimento (só treino); a cobertura é ~80% por "
            "construção, porque a validação é o conjunto de calibração",
        }
    )
    PASTA.mkdir(exist_ok=True)
    joblib.dump(art, PASTA / f"{args.versao}.joblib", compress=3)
    meta = {k: v for k, v in art.items() if k not in ("pcas", "ponto", "ponto_catboost", "quantis", "ood")}
    (PASTA / f"{args.versao}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str))

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("galaxies_validacao")
    with mlflow.start_run(run_name=f"artefato_{args.versao}"):
        mlflow.set_tags({"avaliado_em": "validacao (também é a calibração do intervalo)", **art["cortes"]})
        mlflow.log_params(
            {
                "n_features": len(art["colunas"]),
                "n_arvores": art["n_arvores"],
                "cqr": round(art["cqr"], 4),
                **{f"lgbm_{k}": v for k, v in art["params"].items()},
            }
        )
        mlflow.log_metrics({k: v for k, v in metricas.items() if isinstance(v, float) and not np.isnan(v)})
        mlflow.log_artifact(str(PASTA / f"{args.versao}.json"))
        mlflow.log_artifact(str(PASTA / f"{args.versao}.joblib"))  # o próprio modelo, versionado no run
    print(
        f"{args.versao}: {len(art['colunas'])} features, {art['n_arvores']} árvores, treino + validação "
        f"com {len(final)} vídeos. Validação (modelo de desenvolvimento):"
    )
    for k in (
        "mae_log",
        "spearman_intra_canal",
        "acuracia_par_a_par_7d",
        "cobertura_80",
        "largura_media_intervalo",
        "taxa_baixa_confianca",
    ):
        print(f"  {k}: {metricas[k]:.4f}")


if __name__ == "__main__":
    main()
