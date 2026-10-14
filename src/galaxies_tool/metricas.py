"""Métricas do experimento, conforme a tabela de métricas de docs/EXPERIMENTO.md.

Todas recebem o alvo relativo real (`y`) e previsto (`p`) na escala log:
y = log1p(curtidas) − log1p(mediana do canal).
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

# Faixas: abaixo (< 1/1,5×), na média, acima (1,5× a 3×), viral (>= 3×) — razão (1+curtidas)/(1+mediana)
LIMITES_FAIXAS = (np.log(1 / 1.5), np.log(1.5), np.log(3.0))
NOMES_FAIXAS = ("abaixo", "na_media", "acima", "viral")
MIN_VIDEOS_CANAL = 5


def faixa(alvo: np.ndarray) -> np.ndarray:
    return np.asarray(NOMES_FAIXAS)[np.digitize(alvo, LIMITES_FAIXAS)]


def mae_log(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def mape(y, p) -> float:
    """Erro percentual sobre 1 + curtidas. Como (1 + curtidas) = (1 + mediana)·e^alvo, o erro relativo
    |(1+L̂) − (1+L)| / (1+L) é exatamente |e^(p − y) − 1|, sem depender da mediana do canal."""
    return float(np.mean(np.abs(np.exp(np.asarray(p) - np.asarray(y)) - 1)))


def spearman(y, p) -> float:
    if np.ptp(p) == 0:
        return float("nan")
    return float(spearmanr(y, p).statistic)


def spearman_intra_canal(y, p, canal) -> tuple[float, int]:
    """Média do Spearman por canal (canais com >= 5 vídeos e previsão não constante) e nº de canais usados."""
    df = pd.DataFrame({"y": y, "p": p, "c": canal})
    valores = [spearman(g.y, g.p) for _, g in df.groupby("c") if len(g) >= MIN_VIDEOS_CANAL]
    valores = [v for v in valores if not np.isnan(v)]
    return (float(np.mean(valores)) if valores else float("nan")), len(valores)


def acuracia_par_a_par(y, p, canal, datas=None, max_dias: float | None = None) -> tuple[float, int]:
    """Fração de pares do mesmo canal ordenados como no real (micro: soma sobre todos os pares).

    Pares empatados no real são ignorados; empate na previsão vale meio acerto. Com `datas` e
    `max_dias`, só entram pares publicados a até `max_dias` de distância (mesma época do canal).
    """
    acertos, pares = 0.0, 0
    df = pd.DataFrame({"y": y, "p": p, "c": canal})
    if max_dias is not None:
        df["t"] = pd.to_datetime(pd.Series(datas).to_numpy()).to_numpy("datetime64[s]").astype(np.int64) / 86400
    for _, g in df.groupby("c"):
        yy, pp = g.y.to_numpy(), g.p.to_numpy()
        i, j = np.triu_indices(len(yy), k=1)
        dy, dp = np.sign(yy[i] - yy[j]), np.sign(pp[i] - pp[j])
        valido = dy != 0
        if max_dias is not None:
            tt = g.t.to_numpy()
            valido &= np.abs(tt[i] - tt[j]) <= max_dias
        acertos += np.sum((dy == dp) & valido & (dp != 0)) + 0.5 * np.sum(valido & (dp == 0))
        pares += int(valido.sum())
    return (acertos / pares if pares else float("nan")), pares


def cobertura(y, baixo, alto) -> float:
    y = np.asarray(y)
    return float(np.mean((y >= np.asarray(baixo)) & (y <= np.asarray(alto))))


def f1_faixas(y, p) -> float:
    return float(f1_score(faixa(y), faixa(p), average="macro", labels=list(NOMES_FAIXAS), zero_division=0))


JANELA_MESMA_EPOCA_DIAS = 7


def avaliar(y, p, canal, log_mediana, baixo=None, alto=None, datas=None) -> dict:
    """Todas as métricas da tabela principal. `log_mediana` permite o Spearman em curtidas absolutas."""
    y, p, log_mediana, canal = np.asarray(y), np.asarray(p), np.asarray(log_mediana), np.asarray(canal)
    sp_intra, n_canais = spearman_intra_canal(y, p, canal)
    par, n_pares = acuracia_par_a_par(y, p, canal)
    r = {
        "n": len(y),
        "mae_log": mae_log(y, p),
        "mape": mape(y, p),
        "spearman": spearman(y, p),
        "spearman_absoluto": spearman(y + log_mediana, p + log_mediana),
        "spearman_intra_canal": sp_intra,
        "n_canais_intra": n_canais,
        "acuracia_par_a_par": par,
        "n_pares": n_pares,
        "f1_faixas": f1_faixas(y, p),
    }
    if datas is not None:
        r["acuracia_par_a_par_7d"], r["n_pares_7d"] = acuracia_par_a_par(
            y, p, canal, datas, JANELA_MESMA_EPOCA_DIAS)
    if baixo is not None:
        r["cobertura_80"] = cobertura(y, baixo, alto)
    return r
