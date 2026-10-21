"""Backtest com origem móvel (rolling origin) dentro do período de desenvolvimento.

Uso: uv run python -m galaxies_tool.backtest
Saída: resultados/backtest.csv e resultados/backtest_diferencas.csv (+ MLflow, experimento galaxies_backtest)

Por que existe: os hiperparâmetros e os blocos foram escolhidos num único corte de validação (abr–jul/2026).
Aqui, a mesma receita (LightGBM v1, todos os blocos, `configs/lgbm.yaml`, PCAs ajustados em cada treino) é
reajustada em vários pontos do tempo, e cada modelo é avaliado nos 2 meses seguintes. Responde se o ganho
sobre a mediana do canal é estável no tempo ou depende de um período favorável.

Regras:
- só vídeos de treino + validação dos canais vistos (o teste temporal e os canais novos ficam intocados);
- lacuna de 30 dias entre o fim do treino e o início da avaliação (aula 2.3): as curtidas do fim do treino
  já estariam maduras no momento da primeira previsão;
- `mes` só entra se o treino cobrir 12 meses (mesma regra do modelo principal);
- só o LightGBM (o CatBoost multiplicaria o tempo por ~5; a média dos dois empatou com o LightGBM no teste).
"""

import mlflow
import numpy as np
import pandas as pd

from galaxies_tool import modelo
from galaxies_tool.metricas import bootstrap_diferenca

ORIGENS = ["2025-11-01", "2026-01-01", "2026-03-01", "2026-05-01"]
LACUNA = pd.Timedelta(days=30)
HORIZONTE = pd.DateOffset(months=2)


def main():
    df = modelo.carregar()
    dev = df[df.split.isin(["treino", "validacao"])].copy()
    params = modelo.params_lgbm()
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("galaxies_backtest")
    linhas, deltas = [], []
    for origem in ORIGENS:
        t0 = pd.Timestamp(origem, tz="UTC")
        treino = dev[dev.published_at < t0 - LACUNA].copy()
        aval = dev[(dev.published_at >= t0) & (dev.published_at < t0 + HORIZONTE)].copy()
        pcas = modelo.ajustar_pcas(treino)
        treino, aval = modelo.aplicar_pcas(treino, pcas), modelo.aplicar_pcas(aval, pcas)
        cols = modelo.colunas_completas(treino)
        meses = (treino.published_at.max() - treino.published_at.min()).days / 30.44
        if meses < 12:
            cols = [c for c in cols if c != "mes"]
        m, n_arvores = modelo.treinar_lgbm(treino, cols, params)
        p = m.predict(modelo.matriz(aval, cols))
        prev = {"modelo (LightGBM v1)": p, **modelo.baselines(treino, aval)}
        for nome, pr in prev.items():
            r = modelo.metricas_em(aval, np.asarray(pr))
            linhas.append({"origem": origem, "n_treino": len(treino), "meses_treino": round(meses, 1),
                           "n_aval": len(aval), "modelo": nome, **r})
        for ref in ("baseline_mediana_canal", "baseline_linear"):
            bd = bootstrap_diferenca(aval.alvo, p, np.asarray(prev[ref]), aval.channel_id, aval.published_at)
            deltas.append({"origem": origem, "referencia": ref,
                           **{k: modelo._fmt_delta(bd, k, 4 if k == "mae_log" else 3) for k in bd}})
        with mlflow.start_run(run_name=f"backtest_{origem}"):
            mlflow.set_tags({"avaliado_em": f"backtest {origem} (+2 meses)", **modelo.contexto_registro()})
            mlflow.log_params({"n_treino": len(treino), "n_aval": len(aval), "n_arvores": n_arvores,
                               "usa_mes": "mes" in cols})
            r = modelo.metricas_em(aval, p)
            mlflow.log_metrics({k: float(v) for k, v in r.items() if isinstance(v, (int, float))
                                and not np.isnan(v)})
        print(f"{origem}: treino {len(treino)} ({meses:.1f} meses), avaliação {len(aval)}, {n_arvores} árvores",
              flush=True)
    t, d = pd.DataFrame(linhas), pd.DataFrame(deltas)
    modelo.RESULTADOS.mkdir(exist_ok=True)
    t.to_csv(modelo.RESULTADOS / "backtest.csv", index=False)
    d.to_csv(modelo.RESULTADOS / "backtest_diferencas.csv", index=False)
    cols = ["origem", "modelo", "n_aval", "mae_log", "spearman_intra_canal", "acuracia_par_a_par",
            "acuracia_par_a_par_7d"]
    print(t[cols].round(4).to_string(index=False))
    print(d.to_string(index=False))


if __name__ == "__main__":
    main()
