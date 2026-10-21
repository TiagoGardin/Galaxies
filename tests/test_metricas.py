import numpy as np
import pytest

from galaxies_tool.metricas import (
    acuracia_par_a_par,
    avaliar,
    cobertura,
    f1_faixas,
    faixa,
    mae_log,
    mape,
    spearman_intra_canal,
)


def test_mape_equivale_ao_calculo_em_curtidas():
    mediana = np.array([100.0, 3.0, 0.0])
    likes = np.array([250.0, 0.0, 7.0])
    prev_likes = np.array([180.0, 2.0, 1.0])
    y = np.log1p(likes) - np.log1p(mediana)
    p = np.log1p(prev_likes) - np.log1p(mediana)
    direto = np.mean(np.abs((1 + prev_likes) - (1 + likes)) / (1 + likes))
    assert mape(y, p) == pytest.approx(direto)


def test_mae_log():
    assert mae_log([0, 1, -1], [0.5, 1, 0]) == pytest.approx(0.5)


def test_par_a_par_empates():
    canal = ["a", "a", "a", "b", "b"]
    y = [1.0, 2.0, 2.0, 5.0, 4.0]
    # canal a: pares (0,1) e (0,2) válidos, (1,2) empatado no real → ignorado; canal b: 1 par
    assert acuracia_par_a_par(y, [0, 1, 0, 1, 0], canal) == (pytest.approx((1 + 0.5 + 1) / 3), 3)
    # previsão constante (baseline): todo par válido vale meio acerto
    assert acuracia_par_a_par(y, [0, 0, 0, 0, 0], canal)[0] == pytest.approx(0.5)


def test_spearman_intra_ignora_canais_pequenos_e_previsao_constante():
    canal = ["a"] * 5 + ["b"] * 5 + ["c"] * 3
    y = list(range(5)) + list(range(5)) + [0, 1, 2]
    p = list(range(5)) + [0] * 5 + [2, 1, 0]
    valor, n = spearman_intra_canal(y, p, canal)
    assert n == 1 and valor == pytest.approx(1.0)   # só o canal "a" conta


def test_faixas_e_f1():
    assert faixa(np.log([0.5, 1.0, 2.0, 3.0, 10.0])).tolist() == ["abaixo", "na_media", "acima", "viral", "viral"]
    assert f1_faixas(np.log([0.5, 1, 2, 4]), np.log([0.5, 1, 2, 4])) == pytest.approx(1.0)


def test_cobertura():
    assert cobertura([0, 1, 2, 3], [-1, 1.5, 1, 0], [1, 2, 3, 2]) == pytest.approx(0.5)


def test_avaliar_baseline():
    rng = np.random.default_rng(0)
    y = rng.normal(size=60)
    canal = np.repeat(["a", "b", "c"], 20)
    r = avaliar(y, np.zeros(60), canal, np.full(60, 5.0))
    assert r["acuracia_par_a_par"] == pytest.approx(0.5) and np.isnan(r["spearman_intra_canal"])
    assert r["mae_log"] == pytest.approx(np.mean(np.abs(y)))


def test_par_a_par_restrito_a_mesma_epoca():
    import pandas as pd
    datas = pd.to_datetime(["2026-01-01", "2026-01-05", "2026-03-01"], utc=True)
    y, canal = [1.0, 2.0, 3.0], ["a"] * 3
    p = [0.0, -1.0, 5.0]   # erra o único par próximo (01/01 x 05/01), acerta os pares distantes
    assert acuracia_par_a_par(y, p, canal)[0] == pytest.approx(2 / 3)
    assert acuracia_par_a_par(y, p, canal, datas, 7) == (pytest.approx(0.0), 1)


def test_bootstrap_reproduz_a_diferenca_pontual_e_cobre_zero_para_modelos_iguais():
    import pandas as pd

    from galaxies_tool.metricas import bootstrap_diferenca

    rng = np.random.default_rng(1)
    n = 400
    canal = rng.integers(0, 30, n)
    datas = pd.Timestamp("2026-01-01", tz="UTC") + pd.to_timedelta(rng.integers(0, 60, n), "D")
    y = rng.normal(size=n)
    p1, p0 = y + rng.normal(scale=0.7, size=n), rng.normal(scale=0.01, size=n)
    b = bootstrap_diferenca(y, p1, p0, canal, datas, replicas=300)
    r1, r0 = avaliar(y, p1, canal, np.zeros(n), datas=datas), avaliar(y, p0, canal, np.zeros(n), datas=datas)
    for k in ("mae_log", "acuracia_par_a_par_7d", "spearman_intra_canal"):
        assert b[k]["diferenca"] == pytest.approx(r1[k] - r0[k])
    igual = bootstrap_diferenca(y, p1, p1, canal, datas, replicas=100)
    assert igual["mae_log"]["ic95"] == (0.0, 0.0)


def test_interval_score_penaliza_largura_e_o_que_fica_fora():
    from galaxies_tool.metricas import interval_score

    assert interval_score([0.0], [-1.0], [1.0]) == 2.0  # cobre: só a largura
    assert interval_score([2.0], [-1.0], [1.0]) == 2.0 + 10 * 1.0  # fora por 1: + 2/0,2 × 1
