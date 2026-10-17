"""Baselines e LightGBM nos blocos A, B e C (thumbnail), avaliados na VALIDAÇÃO (o teste fica guardado até a avaliação final).

Uso: uv run python -m galaxies_tool.modelo
Registra cada execução no MLflow (mlflow.db + mlruns/) e imprime a tabela de métricas.

Protocolo (docs/EXPERIMENTO.md): treino = split "treino" (early stopping nos últimos 10% dele, em ordem de
tempo); comparação = split "validacao", que não participa de nenhum ajuste.
PCA dos embeddings (texto e thumbnail) é ajustado só no treino (regra 3 de docs/FEATURES.md).
"""

from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from galaxies_tool.metricas import avaliar

FEATURES = Path("data/features/features_ab.parquet")
EMB_TEXTO = Path("data/features/emb_texto.npy")
EMB_THUMB = Path("data/features/emb_thumb.npy")
THUMBS_CV = Path("data/features/thumbs_cv.parquet")
THUMBS_OCR = Path("data/features/thumbs_ocr.parquet")
N_PCA = 16
FRACAO_PARADA = 0.10  # fim do treino usado só para early stopping
SEMENTE = 42

BLOCO_A = ["log_mediana_likes_ult20", "std_log_likes_ult20", "tendencia", "n_shorts_anteriores", "shorts_ult_30d",
           "dias_desde_ultimo_post", "tipo_canal", "setor", "hora", "dia_semana", "fim_de_semana", "mes",
           "data_sazonal", "duracao_s", "categoria_id", "n_tags"]
BLOCO_B_MANUAL = ["titulo_n_chars", "titulo_n_palavras", "n_hashtags", "tem_hashtag_shorts", "n_emojis",
                  "titulo_pergunta", "titulo_exclamacao", "prop_maiusculas", "titulo_tem_numero", "n_mencoes",
                  "tem_link", "tem_cta", "fmt_grwm", "fmt_tutorial", "fmt_haul", "fmt_review", "fmt_provador",
                  "fmt_antes_depois", "fmt_trend"]
BLOCO_C_THUMB = ["thumb_brilho", "thumb_contraste", "thumb_saturacao", "thumb_colorfulness", "thumb_n_rostos",
                 "thumb_tem_rosto", "thumb_area_rosto", "thumb_tem_texto", "thumb_n_palavras_ocr"]
PARAMS_LGBM = {
    "objective": "regression_l1", "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 50,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "n_estimators": 5000, "random_state": SEMENTE, "verbose": -1,
}


def _embeddings(arquivo: Path, prefixo: str, ids: pd.Series) -> pd.DataFrame | None:
    """Embeddings alinhados por video_id (não pela posição); vídeo sem embedding fica NaN."""
    if not arquivo.exists():
        return None
    ids_emb = arquivo.with_name(arquivo.stem + "_ids.txt").read_text().split("\n")
    emb = np.load(arquivo).astype(np.float32)
    assert len(ids_emb) == len(emb) == len(set(ids_emb))
    tab = pd.DataFrame(emb, index=ids_emb, columns=[f"{prefixo}{i}" for i in range(emb.shape[1])])
    return tab.reindex(ids.values).reset_index(drop=True)


def carregar() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES)
    n = len(df)
    for arquivo, prefixo in ((EMB_TEXTO, "embtxt_"), (EMB_THUMB, "embimg_")):
        emb = _embeddings(arquivo, prefixo, df.video_id)
        if emb is not None:
            df = pd.concat([df, emb], axis=1)
    for arquivo in (THUMBS_CV, THUMBS_OCR):
        if arquivo.exists():
            extra = pd.read_parquet(arquivo)
            assert extra.video_id.is_unique
            df = df.merge(extra, on="video_id", how="left", validate="one_to_one")
    assert len(df) == n
    return df


def adicionar_pca(treino: pd.DataFrame, outros: list[pd.DataFrame], prefixo: str,
                  nome: str) -> tuple[list[str], list[pd.DataFrame]]:
    """PCA ajustado só nos vídeos do treino que têm embedding; quem não tem fica NaN (o LightGBM lida)."""
    cols_emb = [c for c in treino.columns if c.startswith(prefixo)]
    pca = PCA(n_components=N_PCA, random_state=SEMENTE).fit(treino[cols_emb].dropna())
    nomes = nomes_pca(nome)
    return nomes, [aplicar_pca(pca, d, cols_emb, nomes) for d in [treino, *outros]]


def nomes_pca(nome: str) -> list[str]:
    return [f"{nome}_pca_{i + 1}" for i in range(N_PCA)]


def aplicar_pca(pca: PCA, d: pd.DataFrame, cols_emb: list[str], nomes: list[str]) -> pd.DataFrame:
    """Linhas sem embedding (alguma coluna NaN) ficam com os componentes NaN."""
    z = np.full((len(d), len(nomes)), np.nan)
    tem = d[cols_emb].notna().all(axis=1).to_numpy()
    if tem.any():
        z[tem] = pca.transform(d.loc[tem, cols_emb])
    return d.assign(**dict(zip(nomes, z.T, strict=True)))


def matriz(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = df[cols].copy()
    for c in x.columns:
        if x[c].dtype == bool:
            x[c] = x[c].astype(int)
    return x


def treinar_lgbm(treino, cols, params: dict | None = None, metrica: str = "l1"):
    """Early stopping nos últimos 10% do treino (em ordem de tempo), para a validação ficar limpa;
    depois retreina no treino inteiro com o número de árvores encontrado."""
    params = params or PARAMS_LGBM
    ordenado = treino.sort_values("published_at")
    corte = int(len(ordenado) * (1 - FRACAO_PARADA))
    ajuste, parada = ordenado.iloc[:corte], ordenado.iloc[corte:]
    busca = lgb.LGBMRegressor(**params)
    busca.fit(matriz(ajuste, cols), ajuste.alvo, eval_X=(matriz(parada, cols),), eval_y=(parada.alvo,),
              eval_metric=metrica, callbacks=[lgb.early_stopping(200, verbose=False)])
    final = lgb.LGBMRegressor(**{**params, "n_estimators": busca.best_iteration_})
    final.fit(matriz(treino, cols), treino.alvo)
    return final, busca.best_iteration_


def registrar(nome: str, metricas: dict, params: dict | None = None, cols: list[str] | None = None):
    with mlflow.start_run(run_name=nome):
        mlflow.set_tag("avaliado_em", "validacao")
        if params:
            mlflow.log_params(params)
        if cols:
            mlflow.log_param("n_features", len(cols))
            mlflow.log_text("\n".join(cols), "features.txt")
        mlflow.log_metrics({k: v for k, v in metricas.items() if not np.isnan(v)})


def main():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("galaxies_validacao")
    df = carregar()
    treino, valid = df[df.split == "treino"].copy(), df[df.split == "validacao"].copy()
    resultados = {}

    def avaliar_valid(p):
        return avaliar(valid.alvo, p, valid.channel_id, valid.log_mediana_likes_ult20,
                       datas=valid.published_at)

    resultados["baseline_mediana_canal"] = avaliar_valid(np.zeros(len(valid)))
    lin = LinearRegression().fit(treino[["log_mediana_likes_ult20"]], np.log1p(treino.likes))
    resultados["baseline_linear"] = avaliar_valid(
        lin.predict(valid[["log_mediana_likes_ult20"]]) - valid.log_mediana_likes_ult20)

    conjuntos = {"lgbm_A": BLOCO_A, "lgbm_A+B_manual": BLOCO_A + BLOCO_B_MANUAL}
    bloco_b = BLOCO_A + BLOCO_B_MANUAL
    if any(c.startswith("embtxt_") for c in df.columns):
        pca_txt, (treino, valid) = adicionar_pca(treino, [valid], "embtxt_", "emb_texto")
        bloco_b = bloco_b + pca_txt
        conjuntos["lgbm_A+B"] = bloco_b
    # coluna parcial (ex.: OCR ainda rodando) distorceria a comparação: só entra com cobertura completa
    thumb = [c for c in BLOCO_C_THUMB if c in df.columns and df[c].notna().mean() > 0.99]
    if faltando := sorted(set(BLOCO_C_THUMB) - set(thumb)):
        print(f"Bloco C sem: {faltando} (ausentes ou incompletas)")
    if thumb:
        conjuntos["lgbm_A+B+C_thumb_manual"] = bloco_b + thumb
    if any(c.startswith("embimg_") for c in df.columns):
        pca_img, (treino, valid) = adicionar_pca(treino, [valid], "embimg_", "emb_thumb")
        conjuntos["lgbm_A+B+C_thumb"] = bloco_b + thumb + pca_img

    importancias = {}
    for nome, cols in conjuntos.items():
        modelo, n_arvores = treinar_lgbm(treino, cols)
        resultados[nome] = avaliar_valid(modelo.predict(matriz(valid, cols)))
        resultados[nome]["n_arvores"] = n_arvores
        importancias[nome] = pd.Series(modelo.booster_.feature_importance("gain"), index=cols)
        registrar(nome, resultados[nome], {k: v for k, v in PARAMS_LGBM.items() if k != "verbose"}, cols)
    for nome in ("baseline_mediana_canal", "baseline_linear"):
        registrar(nome, resultados[nome])

    tabela = pd.DataFrame(resultados).T
    print(f"Treino: {len(treino)} vídeos | Validação: {len(valid)} vídeos\n")
    print(tabela[["mae_log", "mape", "spearman", "spearman_absoluto", "spearman_intra_canal",
                  "acuracia_par_a_par", "acuracia_par_a_par_7d", "f1_faixas"]].round(4).to_string())
    melhor = list(conjuntos)[-1]
    print(f"\nTop 15 features por ganho ({melhor}):")
    imp = importancias[melhor]
    print((imp / imp.sum()).sort_values(ascending=False).head(15).round(3).to_string())


if __name__ == "__main__":
    main()
