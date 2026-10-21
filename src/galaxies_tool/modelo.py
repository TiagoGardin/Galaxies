"""Baselines, escada de ablação por bloco e ablação "todos menos um", avaliados na VALIDAÇÃO.

Uso: uv run python -m galaxies_tool.modelo
Registra cada modelo no MLflow (mlflow.db + mlruns/), imprime as tabelas e salva-as em resultados/.
O teste fica guardado até a avaliação final (avaliacao_final.py).

Protocolo (docs/EXPERIMENTO.md): treino = split "treino" (early stopping nos últimos 10% dele, em ordem de
tempo); comparação = split "validacao", que não participa de nenhum ajuste. PCAs dos embeddings são ajustados
só no treino (regra 3 de docs/FEATURES.md). Todos os degraus usam os mesmos hiperparâmetros
(configs/lgbm.yaml) e a mesma semente; só mudam as colunas.

A parte de vídeo do bloco C e o bloco D existem só numa amostra (ver EXPERIMENTO.md): fora dela as colunas
ficam NaN, e os degraus que acrescentam esses blocos são comparados nos vídeos da amostra, com IC 95% por
bootstrap pareado reamostrando canais.
"""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from galaxies_tool import multimodal
from galaxies_tool.metricas import avaliar, bootstrap_diferenca

FEATURES = Path("data/features/features_ab.parquet")
BASE = Path("data/processed/base_v1.parquet")
SPLITS = Path("data/processed/splits_v1.json")
EMB_TEXTO = Path("data/features/emb_texto.npy")
EMB_THUMB = Path("data/features/emb_thumb.npy")
EMB_GANCHO = Path("data/features/emb_gancho.npy")
THUMBS_CV = Path("data/features/thumbs_cv.parquet")
THUMBS_OCR = Path("data/features/thumbs_ocr.parquet")
CONFIG = Path("configs/lgbm.yaml")
RESULTADOS = Path("resultados")
VERSAO_BASE = "base_v1"
N_PCA = 16
FRACAO_PARADA = 0.10  # fim do treino usado só para early stopping
SEMENTE = 42

# (prefixo das colunas brutas, nome do embedding, nº de componentes)
EMBEDDINGS = (("embtxt_", "emb_texto", 16), ("embimg_", "emb_thumb", 16), ("embgancho_", "emb_gancho", 8))

# Variantes do modelo. A v1 é a avaliada no teste temporal (19/10/2026) e a servida por padrão. A v2 foi escolhida
# depois, só na validação (21/10/2026): PCA de 32 componentes no texto e na thumbnail (o TAPI sugere 16 a 32) e
# as features de recuperação de comparáveis (rag.py). Ela só é medida no teste prospectivo (docs/EXPERIMENTO.md).
VARIANTES = {
    "v1": {"dims": {"emb_texto": 16, "emb_thumb": 16, "emb_gancho": 8}, "rag": False},
    "v2": {"dims": {"emb_texto": 32, "emb_thumb": 32, "emb_gancho": 8}, "rag": True},
}

BLOCO_A = ["log_mediana_likes_ult20", "std_log_likes_ult20", "tendencia", "n_shorts_anteriores", "shorts_ult_30d",
           "dias_desde_ultimo_post", "tipo_canal", "setor", "hora", "dia_semana", "fim_de_semana", "mes",
           "data_sazonal", "duracao_s", "categoria_id", "n_tags"]
BLOCO_B_MANUAL = ["titulo_n_chars", "titulo_n_palavras", "n_hashtags", "tem_hashtag_shorts", "n_emojis",
                  "titulo_pergunta", "titulo_exclamacao", "prop_maiusculas", "titulo_tem_numero", "n_mencoes",
                  "tem_link", "tem_cta", "fmt_grwm", "fmt_tutorial", "fmt_haul", "fmt_review", "fmt_provador",
                  "fmt_antes_depois", "fmt_trend"]
BLOCO_C_THUMB = ["thumb_brilho", "thumb_contraste", "thumb_saturacao", "thumb_colorfulness", "thumb_n_rostos",
                 "thumb_tem_rosto", "thumb_area_rosto", "thumb_tem_texto", "thumb_n_palavras_ocr"]
BLOCO_C_VIDEO = list(multimodal.COLUNAS_VIDEO)
BLOCO_D = list(multimodal.COLUNAS_VLM)
CATEGORICAS = ("setor", "tipo_canal", "categoria_id", *multimodal.COLUNAS_VLM_CATEGORICAS)
PARAMS_LGBM = {
    "objective": "regression_l1", "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 50,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l1": 0.0, "lambda_l2": 1.0,
    "n_estimators": 5000, "random_state": SEMENTE, "verbose": -1,
}


def params_lgbm() -> dict:
    """Hiperparâmetros padrão, sobrescritos pelos de configs/lgbm.yaml (saída de ajustar.py), se existir."""
    if not CONFIG.exists():
        return dict(PARAMS_LGBM)
    return {**PARAMS_LGBM, **yaml.safe_load(CONFIG.read_text())["params"]}


# ---------------------------------------------------------------- dados


def _embeddings(arquivo: Path, prefixo: str, ids: pd.Series) -> pd.DataFrame | None:
    """Embeddings alinhados por video_id (não pela posição); vídeo sem embedding fica NaN."""
    if not arquivo.exists():
        return None
    ids_emb = arquivo.with_name(arquivo.stem + "_ids.txt").read_text().split("\n")
    emb = np.load(arquivo).astype(np.float32)
    assert len(ids_emb) == len(emb) == len(set(ids_emb))
    tab = pd.DataFrame(emb, index=ids_emb, columns=[f"{prefixo}{i}" for i in range(emb.shape[1])])
    return tab.reindex(ids.values).reset_index(drop=True)


def carregar(rag: bool = False) -> pd.DataFrame:
    """Uma linha por vídeo-alvo com todos os blocos. Vídeo fora da amostra dos blocos C (vídeo) e D, ou
    cuja extração falhou, fica com essas colunas NaN; `tem_video` e `tem_vlm` marcam quem tem (diagnóstico,
    nunca feature). Com `rag`, acrescenta as features de recuperação de comparáveis (variante v2)."""
    df = pd.read_parquet(FEATURES)
    n = len(df)
    for arquivo, (prefixo, _, _) in zip((EMB_TEXTO, EMB_THUMB, EMB_GANCHO), EMBEDDINGS, strict=True):
        emb = _embeddings(arquivo, prefixo, df.video_id)
        if emb is not None:
            df = pd.concat([df, emb], axis=1)
    video, vlm = multimodal.tabela_video(), multimodal.tabela_vlm()
    for extra in [*(pd.read_parquet(a) for a in (THUMBS_CV, THUMBS_OCR) if a.exists()), video, vlm]:
        assert extra.video_id.is_unique
        df = df.merge(extra, on="video_id", how="left", validate="one_to_one")
    df["tem_video"] = df.video_id.isin(video.video_id)
    df["tem_vlm"] = df.video_id.isin(vlm.video_id)
    if rag:
        from galaxies_tool import rag as rag_

        df = pd.concat([df, rag_.features_da_base(df)], axis=1)
    assert len(df) == n
    return df


# ---------------------------------------------------------------- PCA (ajustado só no treino)


def nomes_pca(nome: str, n: int = N_PCA) -> list[str]:
    return [f"{nome}_pca_{i + 1}" for i in range(n)]


def aplicar_pca(pca: PCA, d: pd.DataFrame, cols_emb: list[str], nomes: list[str]) -> pd.DataFrame:
    """Linhas sem embedding (alguma coluna NaN) ficam com os componentes NaN."""
    z = np.full((len(d), len(nomes)), np.nan)
    tem = d[cols_emb].notna().all(axis=1).to_numpy()
    if tem.any():
        z[tem] = pca.transform(d.loc[tem, cols_emb])
    return pd.concat([d.drop(columns=nomes, errors="ignore"),
                      pd.DataFrame(z, columns=nomes, index=d.index)], axis=1)


def adicionar_pca(treino: pd.DataFrame, outros: list[pd.DataFrame], prefixo: str, nome: str,
                  n: int = N_PCA) -> tuple[list[str], list[pd.DataFrame]]:
    """PCA ajustado só nos vídeos do treino que têm embedding; quem não tem fica NaN (o LightGBM lida)."""
    cols_emb = [c for c in treino.columns if c.startswith(prefixo)]
    pca = PCA(n_components=n, random_state=SEMENTE).fit(treino[cols_emb].dropna())
    nomes = nomes_pca(nome, n)
    return nomes, [aplicar_pca(pca, d, cols_emb, nomes) for d in [treino, *outros]]


def ajustar_pcas(treino: pd.DataFrame, variante: str = "v1") -> dict:
    """Os três PCAs, ajustados no `treino`, no formato guardado pelo artefato."""
    pcas = {}
    for prefixo, nome, _ in EMBEDDINGS:
        n = VARIANTES[variante]["dims"][nome]
        cols = [c for c in treino.columns if c.startswith(prefixo)]
        if cols:
            pca = PCA(n_components=n, random_state=SEMENTE).fit(treino[cols].dropna())
            pcas[nome] = {"pca": pca, "colunas": cols, "nomes": nomes_pca(nome, n)}
    return pcas


def aplicar_pcas(df: pd.DataFrame, pcas: dict) -> pd.DataFrame:
    for p in pcas.values():
        df = aplicar_pca(p["pca"], df, p["colunas"], p["nomes"])
    return df


# ---------------------------------------------------------------- blocos e escada


def colunas_thumb(df: pd.DataFrame) -> list[str]:
    """Colunas da thumbnail com cobertura completa (uma coluna parcial distorceria a comparação)."""
    return [c for c in BLOCO_C_THUMB if c in df.columns and df[c].notna().mean() > 0.99]


def blocos(df: pd.DataFrame, variante: str = "v1") -> dict[str, list[str]]:
    """Colunas de cada bloco, na ordem da escada de ablação (a v2 acrescenta o bloco R, de comparáveis)."""
    d = VARIANTES[variante]["dims"]
    b = {
        "A": BLOCO_A,
        "B": BLOCO_B_MANUAL + nomes_pca("emb_texto", d["emb_texto"]),
        "C_thumb": colunas_thumb(df) + nomes_pca("emb_thumb", d["emb_thumb"]),
        "C_video": BLOCO_C_VIDEO + nomes_pca("emb_gancho", d["emb_gancho"]),
        "D": BLOCO_D,
    }
    if VARIANTES[variante]["rag"]:
        from galaxies_tool import rag

        b["R"] = list(rag.COLUNAS)
    return b


def colunas_completas(df: pd.DataFrame, variante: str = "v1") -> list[str]:
    return [c for cols in blocos(df, variante).values() for c in cols]


def matriz(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = df[cols].copy()
    for c in x.columns:
        if x[c].dtype == bool:
            x[c] = x[c].astype(int)
    return x


def treinar_lgbm(treino, cols, params: dict | None = None, metrica: str = "l1", n_arvores: int | None = None):
    """Early stopping nos últimos 10% do treino (em ordem de tempo), para a validação ficar limpa;
    depois retreina no treino inteiro com o número de árvores encontrado. Com `n_arvores` (refit em treino +
    validação), usa o número já escolhido na validação, sem novo early stopping (aula 3.4)."""
    params = params or params_lgbm()
    if n_arvores:
        final = lgb.LGBMRegressor(**{**params, "n_estimators": n_arvores})
        return final.fit(matriz(treino, cols), treino.alvo), n_arvores
    ordenado = treino.sort_values("published_at")
    corte = int(len(ordenado) * (1 - FRACAO_PARADA))
    ajuste, parada = ordenado.iloc[:corte], ordenado.iloc[corte:]
    busca = lgb.LGBMRegressor(**params)
    busca.fit(matriz(ajuste, cols), ajuste.alvo, eval_X=(matriz(parada, cols),), eval_y=(parada.alvo,),
              eval_metric=metrica, callbacks=[lgb.early_stopping(200, verbose=False)])
    final = lgb.LGBMRegressor(**{**params, "n_estimators": busca.best_iteration_})
    final.fit(matriz(treino, cols), treino.alvo)
    return final, busca.best_iteration_


PARAMS_CATBOOST = {"loss_function": "MAE", "learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 3, "has_time": True,
                   "random_seed": SEMENTE, "verbose": 0, "thread_count": -1}


def como_texto(col: pd.Series) -> pd.Series:
    """Categoria como texto, com ausente explícito (CatBoost e o one-hot não aceitam NaN em categoria)."""
    return col.astype(object).where(col.notna(), "ausente").astype(str)


def matriz_catboost(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = matriz(df, cols)
    for c in cols:
        if c in CATEGORICAS:
            x[c] = como_texto(x[c])
    return x


def treinar_catboost(treino, cols, n_iteracoes: int | None = None):
    """CatBoost (L1) com o mesmo protocolo do LightGBM: early stopping nos últimos 10% do treino em ordem de
    tempo e refit no treino inteiro; com `n_iteracoes` (refit em treino + validação), sem novo early stopping."""
    from catboost import CatBoostRegressor

    cats = [c for c in cols if c in CATEGORICAS]
    ordenado = treino.sort_values("published_at")  # has_time: as linhas precisam estar em ordem de tempo
    if not n_iteracoes:
        corte = int(len(ordenado) * (1 - FRACAO_PARADA))
        ajuste, parada = ordenado.iloc[:corte], ordenado.iloc[corte:]
        busca = CatBoostRegressor(iterations=5000, od_type="Iter", od_wait=200, **PARAMS_CATBOOST)
        busca.fit(matriz_catboost(ajuste, cols), ajuste.alvo, cat_features=cats,
                  eval_set=(matriz_catboost(parada, cols), parada.alvo))
        n_iteracoes = busca.get_best_iteration() + 1
    final = CatBoostRegressor(iterations=n_iteracoes, **PARAMS_CATBOOST)
    final.fit(matriz_catboost(ordenado, cols), ordenado.alvo, cat_features=cats)
    return final, n_iteracoes


# ---------------------------------------------------------------- avaliação e registro


def metricas_em(df: pd.DataFrame, p: np.ndarray, mascara=None) -> dict:
    m = np.ones(len(df), bool) if mascara is None else np.asarray(mascara)
    d = df[m]
    return avaliar(d.alvo, np.asarray(p)[m], d.channel_id, d.log_mediana_likes_ult20, datas=d.published_at)


@lru_cache(maxsize=1)
def contexto_registro() -> dict:
    """Tags de rastreabilidade de cada run: versão e hash da base e das features, e datas de corte."""
    s = json.loads(SPLITS.read_text())
    return {"versao_base": VERSAO_BASE, "hash_base": _hash(BASE), "hash_features": _hash(FEATURES),
            "corte_treino_validacao": s["corte_treino_validacao"], "corte_validacao_teste": s["corte_validacao_teste"]}


def _hash(arquivo: Path) -> str:
    return hashlib.sha256(arquivo.read_bytes()).hexdigest()[:12]


def importancia_shap(m: lgb.LGBMRegressor, x: pd.DataFrame) -> pd.Series:
    """Média do |SHAP| por coluna (TreeSHAP exato do próprio LightGBM)."""
    contrib = m.predict(x, pred_contrib=True)[:, :-1]
    return pd.Series(np.abs(contrib).mean(axis=0), index=x.columns).sort_values(ascending=False)


def registrar(nome: str, metricas: dict, params: dict | None = None, cols: list[str] | None = None,
              tags: dict | None = None, shap: pd.Series | None = None):
    with mlflow.start_run(run_name=nome):
        mlflow.set_tags({"avaliado_em": "validacao", **contexto_registro(), **(tags or {})})
        if params:
            mlflow.log_params({k: v for k, v in params.items() if k != "verbose"})
        if cols:
            mlflow.log_param("n_features", len(cols))
            mlflow.log_text("\n".join(cols), "features.txt")
        if shap is not None:
            mlflow.log_text(shap.round(5).to_string(), "shap_medio_abs.txt")
        mlflow.log_metrics({k: float(v) for k, v in metricas.items()
                            if isinstance(v, (int, float, np.number)) and not np.isnan(v)})


COLS_TABELA = ["n", "mae_log", "mape", "spearman", "spearman_intra_canal", "acuracia_par_a_par",
               "acuracia_par_a_par_7d", "f1_faixas"]


def _fmt_delta(b: dict, k: str, casas: int = 3) -> str:
    d, (lo, hi) = b[k]["diferenca"], b[k]["ic95"]
    return f"{d:+.{casas}f} [{lo:+.{casas}f}; {hi:+.{casas}f}]"


def conjuntos_de_features(df: pd.DataFrame) -> tuple[dict, dict]:
    """(escada cumulativa, variantes): "todos menos um" a partir do completo e a ablação dentro do bloco D
    (binárias × notas, hipótese H5)."""
    b = blocos(df)
    escada = {"A": b["A"], "A+B (manual)": b["A"] + BLOCO_B_MANUAL, "A+B": b["A"] + b["B"]}
    escada["A+B+C (thumb)"] = escada["A+B"] + b["C_thumb"]
    escada["A+B+C (thumb + vídeo)"] = escada["A+B+C (thumb)"] + b["C_video"]
    escada["A+B+C+D"] = escada["A+B+C (thumb + vídeo)"] + b["D"]
    completo = escada["A+B+C+D"]
    variantes = {f"completo sem {k}": [c for c in completo if c not in set(b[k])]
                 for k in ("B", "C_thumb", "C_video", "D")}
    sem_d = escada["A+B+C (thumb + vídeo)"]
    variantes["A+B+C + D binárias"] = sem_d + multimodal.COLUNAS_VLM_BINARIAS + multimodal.COLUNAS_VLM_CATEGORICAS
    variantes["A+B+C + D notas"] = sem_d + multimodal.COLUNAS_VLM_NOTAS
    variantes["A+B+C + ruído (controle negativo)"] = sem_d + COLUNAS_RUIDO
    return escada, variantes


COLUNAS_RUIDO = [f"ruido_{i}" for i in range(len(BLOCO_D))]


def com_ruido(df: pd.DataFrame) -> pd.DataFrame:
    """Controle negativo (aula 2.6): tantas colunas de ruído quanto o bloco D, só nos vídeos com VLM (como o
    D). O ruído depende só do video_id, então é o mesmo no treino e na avaliação."""
    z = np.stack([np.random.default_rng(int(hashlib.sha1(v.encode()).hexdigest()[:8], 16)).normal(
        size=len(COLUNAS_RUIDO)) for v in df.video_id])
    z[~df.tem_vlm.to_numpy()] = np.nan
    return df.assign(**dict(zip(COLUNAS_RUIDO, z.T, strict=True)))


# degraus: cada um contra a referência, nos vídeos em que o bloco novo existe ("amostra")
COMPARACOES = [
    ("A+B (manual)", "A", "inteiro"), ("A+B", "A+B (manual)", "inteiro"), ("A+B+C (thumb)", "A+B", "inteiro"),
    ("A+B+C (thumb + vídeo)", "A+B+C (thumb)", "com vídeo"), ("A+B+C+D", "A+B+C (thumb + vídeo)", "com VLM"),
    ("A+B+C+D", "A+B", "com VLM"), ("A+B+C+D", "A+B+C (thumb)", "inteiro"),
    ("A+B+C+D", "baseline_mediana_canal", "inteiro"), ("A+B+C+D", "baseline_linear", "inteiro"),
    ("A+B+C+D", "completo sem B", "inteiro"), ("A+B+C+D", "completo sem C_thumb", "inteiro"),
    ("A+B+C+D", "completo sem C_video", "com vídeo"), ("A+B+C+D", "completo sem D", "com VLM"),
    ("A+B+C + D binárias", "A+B+C + D notas", "com VLM"),
    ("A+B+C + ruído (controle negativo)", "A+B+C (thumb + vídeo)", "com VLM"),
]


def amostras_de(aval: pd.DataFrame) -> dict[str, np.ndarray]:
    return {"inteiro": np.ones(len(aval), bool), "com vídeo": aval.tem_video.to_numpy(),
            "com VLM": aval.tem_vlm.to_numpy()}


def baselines(treino: pd.DataFrame, aval: pd.DataFrame) -> dict[str, np.ndarray]:
    lin = LinearRegression().fit(treino[["log_mediana_likes_ult20"]], np.log1p(treino.likes))
    return {"baseline_mediana_canal": np.zeros(len(aval)),
            "baseline_linear": lin.predict(aval[["log_mediana_likes_ult20"]]) - aval.log_mediana_likes_ult20}


def rodar_ablacao(treino: pd.DataFrame, aval: pd.DataFrame, params: dict, experimento: str,
                  rotulo: str, arvores: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Treina a escada e as variantes no `treino` (PCAs já aplicados) e avalia em `aval`. Registra cada
    modelo no MLflow e devolve (métricas por amostra, diferenças com IC 95%). Com `arvores` (nome → nº de
    árvores, vindos da validação), cada modelo usa esse número em vez de um novo early stopping."""
    arvores = arvores or {}
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experimento)
    treino, aval = com_ruido(treino), com_ruido(aval)
    amostras = amostras_de(aval)
    prev = baselines(treino, aval)
    escada, variantes = conjuntos_de_features(treino)
    for nome, cols in {**escada, **variantes}.items():
        m, n_arvores = treinar_lgbm(treino, cols, params, n_arvores=arvores.get(nome))
        arvores[nome] = n_arvores
        prev[nome] = m.predict(matriz(aval, cols))
        shap = importancia_shap(m, matriz(aval, cols)) if nome == "A+B+C+D" else None
        registrar(nome, {**metricas_em(aval, prev[nome]), "n_arvores": n_arvores}, params, cols,
                  {"tipo": "escada" if nome in escada else "variante", "avaliado_em": rotulo}, shap)
        print(f"{nome}: {n_arvores} árvores", flush=True)
    for nome in ("baseline_mediana_canal", "baseline_linear"):
        registrar(nome, metricas_em(aval, prev[nome]), tags={"tipo": "baseline", "avaliado_em": rotulo})

    tabela = pd.DataFrame([{"amostra": a, "modelo": nome, "n_arvores": arvores.get(nome),
                            **metricas_em(aval, p, m)}
                           for a, m in amostras.items() for nome, p in prev.items()])
    linhas = []
    for novo, ref, amostra in COMPARACOES:
        m = amostras[amostra]
        v = aval[m]
        bd = bootstrap_diferenca(v.alvo, prev[novo][m], prev[ref][m], v.channel_id, v.published_at)
        linhas.append({"novo": novo, "referencia": ref, "amostra": amostra, "n": int(m.sum()),
                       **{k: _fmt_delta(bd, k, 4 if k == "mae_log" else 3) for k in bd}})
    return tabela, pd.DataFrame(linhas)


def imprimir(tabela: pd.DataFrame, deltas: pd.DataFrame):
    for amostra in tabela.amostra.unique():
        t = tabela[tabela.amostra == amostra].set_index("modelo")[COLS_TABELA]
        print(f"\n### {amostra}\n{t.round(4).to_string()}")
    print("\n### Diferenças (novo − referência) com IC 95% por bootstrap pareado de canais")
    print(deltas.to_string(index=False))


def main():
    df = carregar()
    treino, valid = df[df.split == "treino"].copy(), df[df.split == "validacao"].copy()
    pcas = ajustar_pcas(treino)
    treino, valid = aplicar_pcas(treino, pcas), aplicar_pcas(valid, pcas)
    tabela, deltas = rodar_ablacao(treino, valid, params_lgbm(), "galaxies_validacao", "validacao")
    RESULTADOS.mkdir(exist_ok=True)
    tabela.to_csv(RESULTADOS / "validacao_metricas.csv", index=False)
    deltas.to_csv(RESULTADOS / "validacao_ablacao.csv", index=False)
    imprimir(tabela, deltas)


if __name__ == "__main__":
    main()
