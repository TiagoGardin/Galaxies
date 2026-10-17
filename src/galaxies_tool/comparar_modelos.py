"""Comparação de famílias de modelo na VALIDAÇÃO, com as mesmas features e o mesmo protocolo do LightGBM.

Uso: uv run python -m galaxies_tool.comparar_modelos
Registra cada modelo no MLflow (experimento galaxies_validacao, tag comparacao=modelos).

Mesmo conjunto de features do artefato (A + B + C da thumbnail, PCAs ajustados só no treino) e mesmo
early stopping (últimos 10% do treino, em ordem de tempo). Os boostings usam perda L1, como o LightGBM;
a Ridge (L2) é o contraponto linear. Hiperparâmetros próximos dos padrões de cada biblioteca, sem busca:
a pergunta é se outra família é claramente melhor, não qual ajuste extrai mais de cada uma.
"""

import time

import mlflow
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from galaxies_tool import modelo
from galaxies_tool.artefato import CATEGORICAS, colunas_do_modelo
from galaxies_tool.metricas import avaliar

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
    cats = [c for c in cols if c in CATEGORICAS]

    def x(d):
        m = modelo.matriz(d, cols)
        for c in cats:
            m[c] = m[c].astype(str)
        return m

    params = {"loss_function": "MAE", "learning_rate": TAXA, "depth": 6, "l2_leaf_reg": 3,
              "random_seed": modelo.SEMENTE, "verbose": 0, "thread_count": -1}
    ajuste, parada = dividir_parada(treino)
    busca = CatBoostRegressor(iterations=MAX_ARVORES, od_type="Iter", od_wait=PACIENCIA, **params)
    busca.fit(x(ajuste), ajuste.alvo, cat_features=cats, eval_set=(x(parada), parada.alvo))
    n = busca.get_best_iteration() + 1
    final = CatBoostRegressor(iterations=n, **params).fit(x(treino), treino.alvo, cat_features=cats)
    return final.predict(x(valid)), n


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
            r[c] = r[c].astype(str)
        return r

    m.fit(x(treino), treino.alvo)
    return m.predict(x(valid)), None


def main():
    df = modelo.carregar()
    treino, valid = df[df.split == "treino"].copy(), df[df.split == "validacao"].copy()
    for prefixo, nome in (("embtxt_", "emb_texto"), ("embimg_", "emb_thumb")):
        _, (treino, valid) = modelo.adicionar_pca(treino, [valid], prefixo, nome)
    cols = colunas_do_modelo(treino)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("galaxies_validacao")
    linhas = {}
    for nome, fn in (("lightgbm", lightgbm), ("catboost", catboost), ("xgboost", xgboost), ("ridge", ridge)):
        t = time.time()
        p, n = fn(treino, valid, cols)
        r = avaliar(valid.alvo, p, valid.channel_id, valid.log_mediana_likes_ult20, datas=valid.published_at)
        r["segundos"], r["n_arvores"] = round(time.time() - t), n
        linhas[nome] = r
        with mlflow.start_run(run_name=f"comparacao_{nome}"):
            mlflow.set_tags({"avaliado_em": "validacao", "comparacao": "modelos"})
            mlflow.log_param("n_features", len(cols))
            mlflow.log_metrics({k: float(v) for k, v in r.items() if v is not None and not np.isnan(v)})
        print(f"{nome}: {r['mae_log']:.4f} MAE-log ({r['segundos']} s)", flush=True)

    tabela = pd.DataFrame(linhas).T
    print(f"\nValidação: {len(valid)} vídeos, {len(cols)} features\n")
    print(tabela[["mae_log", "mape", "spearman", "spearman_intra_canal", "acuracia_par_a_par",
                  "acuracia_par_a_par_7d", "f1_faixas", "n_arvores", "segundos"]].round(4).to_string())


if __name__ == "__main__":
    main()
