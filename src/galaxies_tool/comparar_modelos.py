"""Comparação de famílias de modelo na VALIDAÇÃO, com as mesmas features e o mesmo protocolo do LightGBM.

Uso: uv run python -m galaxies_tool.comparar_modelos
Registra cada modelo no MLflow (experimento galaxies_validacao, tag comparacao=modelos).

Mesmo conjunto de features do artefato (A+B+C+D, PCAs ajustados só no treino) e mesmo early stopping
(últimos 10% do treino, em ordem de tempo). Os boostings usam perda L1, como o LightGBM; a Ridge (L2) é o
contraponto linear. O LightGBM usa os hiperparâmetros de configs/lgbm.yaml; os outros, valores próximos dos
padrões de cada biblioteca, sem busca: a pergunta é se outra família é claramente melhor, não qual ajuste
extrai mais de cada uma. Também testa a média LightGBM + CatBoost contra o LightGBM sozinho, com IC 95%
por bootstrap pareado de canais (a média só entra se o ganho superar o ruído).
"""

import time

import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from galaxies_tool import modelo
from galaxies_tool.metricas import avaliar, bootstrap_diferenca
from galaxies_tool.modelo import CATEGORICAS

TAXA = 0.03
MAX_ARVORES = 5000
PACIENCIA = 200


def dividir_parada(treino: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordenado = treino.sort_values("published_at")
    corte = int(len(ordenado) * (1 - modelo.FRACAO_PARADA))
    return ordenado.iloc[:corte], ordenado.iloc[corte:]


def lightgbm(treino, valid, cols):
    m, n = modelo.treinar_lgbm(treino, cols)
    return m.predict(modelo.matriz(valid, cols)), n


def catboost(treino, valid, cols):
    m, n = modelo.treinar_catboost(treino, cols)
    return m.predict(modelo.matriz_catboost(valid, cols)), n


def xgboost(treino, valid, cols):
    params = {"objective": "reg:absoluteerror", "learning_rate": TAXA, "max_depth": 6, "min_child_weight": 50,
              "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "tree_method": "hist",
              "enable_categorical": True, "random_state": modelo.SEMENTE, "n_jobs": -1}
    ajuste, parada = dividir_parada(treino)
    busca = XGBRegressor(n_estimators=MAX_ARVORES, early_stopping_rounds=PACIENCIA, eval_metric="mae", **params)
    busca.fit(modelo.matriz(ajuste, cols), ajuste.alvo, eval_set=[(modelo.matriz(parada, cols), parada.alvo)],
              verbose=False)
    n = busca.best_iteration + 1
    final = XGBRegressor(n_estimators=n, **params).fit(modelo.matriz(treino, cols), treino.alvo)
    return final.predict(modelo.matriz(valid, cols)), n


def ridge(treino, valid, cols):
    """Linear com todas as features: categóricas em one-hot, ausentes pela mediana (com indicador)."""
    cats = [c for c in cols if c in CATEGORICAS]
    nums = [c for c in cols if c not in cats]
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), nums),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore")), cats),
    ])
    m = make_pipeline(prep, RidgeCV(alphas=np.logspace(-2, 3, 12)))

    def x(d):
        r = modelo.matriz(d, cols)
        for c in cats:
            r[c] = modelo.como_texto(r[c])
        return r

    m.fit(x(treino), treino.alvo)
    return m.predict(x(valid)), None


def main():
    df = modelo.carregar()
    treino, valid = df[df.split == "treino"].copy(), df[df.split == "validacao"].copy()
    pcas = modelo.ajustar_pcas(treino)
    treino, valid = modelo.aplicar_pcas(treino, pcas), modelo.aplicar_pcas(valid, pcas)
    cols = modelo.colunas_completas(df)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("galaxies_validacao")
    linhas, prev = {}, {}
    for nome, fn in (("lightgbm", lightgbm), ("catboost", catboost), ("xgboost", xgboost), ("ridge", ridge),
                     ("media_lightgbm_catboost", None)):
        t = time.time()
        p, n = fn(treino, valid, cols) if fn else ((prev["lightgbm"] + prev["catboost"]) / 2, None)
        prev[nome] = p
        r = avaliar(valid.alvo, p, valid.channel_id, valid.log_mediana_likes_ult20, datas=valid.published_at)
        r["segundos"], r["n_arvores"] = round(time.time() - t), n
        linhas[nome] = r
        with mlflow.start_run(run_name=f"comparacao_{nome}"):
            mlflow.set_tags({"avaliado_em": "validacao", "comparacao": "modelos"})
            mlflow.log_param("n_features", len(cols))
            mlflow.log_metrics({k: float(v) for k, v in r.items() if v is not None and not np.isnan(v)})
        print(f"{nome}: {r['mae_log']:.4f} MAE-log ({r['segundos']} s)", flush=True)

    tabela = pd.DataFrame(linhas).T
    modelo.RESULTADOS.mkdir(exist_ok=True)
    tabela.to_csv(modelo.RESULTADOS / "validacao_familias.csv")
    print(f"\nValidação: {len(valid)} vídeos, {len(cols)} features\n")
    print(tabela[["mae_log", "mape", "spearman", "spearman_intra_canal", "acuracia_par_a_par",
                  "acuracia_par_a_par_7d", "f1_faixas", "n_arvores", "segundos"]].round(4).to_string())
    for outro in ("catboost", "media_lightgbm_catboost"):
        bd = bootstrap_diferenca(valid.alvo, prev[outro], prev["lightgbm"], valid.channel_id, valid.published_at)
        print(f"\n{outro} − lightgbm (IC 95%, bootstrap de canais):")
        for k, v in bd.items():
            print(f"  {k}: {v['diferenca']:+.4f} [{v['ic95'][0]:+.4f}; {v['ic95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
