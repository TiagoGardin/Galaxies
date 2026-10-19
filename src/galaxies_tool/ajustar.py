"""Busca de hiperparâmetros do LightGBM com Optuna, na VALIDAÇÃO (o teste não participa).

Uso: uv run python -m galaxies_tool.ajustar [--tentativas 40]
Saída: configs/lgbm.yaml (hiperparâmetros escolhidos, lidos por modelo.params_lgbm) e um run no MLflow.

Objetivo: MAE-log na validação inteira, com o conjunto completo de features (A+B+C+D) e o mesmo protocolo de
modelo.treinar_lgbm (early stopping nos últimos 10% do treino, em ordem de tempo). O MAE-log foi escolhido
como objetivo por ser a métrica de erro mais estável; as métricas de ordenação são conferidas depois, na
ablação. Amostrador TPE com semente fixa: a busca é reprodutível.
"""

import argparse
from datetime import UTC, datetime

import mlflow
import numpy as np
import optuna
import yaml

from galaxies_tool import modelo
from galaxies_tool.metricas import avaliar

ESPACO = {  # nome: (mínimo, máximo, escala log?)
    "learning_rate": (0.01, 0.1, True),
    "num_leaves": (7, 127, True),
    "min_child_samples": (20, 300, True),
    "feature_fraction": (0.3, 1.0, False),
    "bagging_fraction": (0.5, 1.0, False),
    "lambda_l1": (1e-3, 10.0, True),
    "lambda_l2": (1e-3, 30.0, True),
}
INTEIROS = {"num_leaves", "min_child_samples"}


def sugerir(trial: optuna.Trial) -> dict:
    p = {}
    for nome, (lo, hi, log) in ESPACO.items():
        sugerir_valor = trial.suggest_int if nome in INTEIROS else trial.suggest_float
        p[nome] = sugerir_valor(nome, lo, hi, log=log)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tentativas", type=int, default=40)
    args = ap.parse_args()
    df = modelo.carregar()
    treino, valid = df[df.split == "treino"].copy(), df[df.split == "validacao"].copy()
    pcas = modelo.ajustar_pcas(treino)
    treino, valid = modelo.aplicar_pcas(treino, pcas), modelo.aplicar_pcas(valid, pcas)
    cols = modelo.colunas_completas(df)
    x_valid = modelo.matriz(valid, cols)
    arvores = {}

    def objetivo(trial: optuna.Trial) -> float:
        params = {**modelo.PARAMS_LGBM, **sugerir(trial)}
        m, n = modelo.treinar_lgbm(treino, cols, params)
        arvores[trial.number] = n
        p = m.predict(x_valid)
        r = avaliar(valid.alvo, p, valid.channel_id, valid.log_mediana_likes_ult20, datas=valid.published_at)
        for k in ("spearman_intra_canal", "acuracia_par_a_par_7d"):
            trial.set_user_attr(k, r[k])
        print(f"tentativa {trial.number}: MAE-log {r['mae_log']:.4f} | {n} árvores", flush=True)
        return r["mae_log"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudo = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=modelo.SEMENTE)
    )
    # o ponto de partida também concorre (lambda_l1 = 0 vira o mínimo do espaço, praticamente o mesmo)
    estudo.enqueue_trial({k: min(max(modelo.PARAMS_LGBM[k], lo), hi) for k, (lo, hi, _) in ESPACO.items()})
    estudo.optimize(objetivo, n_trials=args.tentativas)

    melhor = estudo.best_trial
    padrao = estudo.trials[0].value
    modelo.CONFIG.parent.mkdir(exist_ok=True)
    modelo.CONFIG.write_text(
        yaml.safe_dump(
            {
                "gerado_por": "uv run python -m galaxies_tool.ajustar",
                "gerado_em": datetime.now(UTC).isoformat(timespec="seconds"),
                "objetivo": "MAE-log na validação (A+B+C+D)",
                "tentativas": len(estudo.trials),
                "mae_log_validacao": {"padrao": round(padrao, 5), "escolhido": round(melhor.value, 5)},
                "arvores_no_early_stopping": arvores[melhor.number],
                "params": {
                    k: (int(v) if k in INTEIROS else float(round(v, 6))) for k, v in melhor.params.items()
                },
            },
            allow_unicode=True,
            sort_keys=False,
        )
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("galaxies_validacao")
    with mlflow.start_run(run_name="ajuste_optuna"):
        mlflow.set_tags({"avaliado_em": "validacao", "tipo": "ajuste", **modelo.contexto_registro()})
        mlflow.log_params(melhor.params)
        mlflow.log_metrics(
            {
                "mae_log_padrao": padrao,
                "mae_log_escolhido": melhor.value,
                "tentativas": len(estudo.trials),
                **{k: v for k, v in melhor.user_attrs.items() if not np.isnan(v)},
            }
        )
        mlflow.log_artifact(str(modelo.CONFIG))
    print(f"\nPadrão: {padrao:.4f} → escolhido: {melhor.value:.4f} (tentativa {melhor.number})")
    print(yaml.safe_dump(melhor.params))


if __name__ == "__main__":
    main()
