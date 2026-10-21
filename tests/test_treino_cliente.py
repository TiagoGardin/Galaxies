"""Treino dinâmico (treino_cliente.py): esquema mínimo do TAPI e treino de ponta a ponta numa base sintética."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from galaxies_tool import treino_cliente as tc


def base_sintetica(n_canais: int = 8, por_canal: int = 160, semente: int = 0) -> pd.DataFrame:
    """Posts semanais; "tutorial" na legenda rende ~2,7x a mediana do canal (o sinal a ser aprendido)."""
    rng = np.random.default_rng(semente)
    linhas = []
    for c in range(n_canais):
        nivel = rng.uniform(3, 8)
        for k in range(por_canal):
            tutorial = rng.random() < 0.4
            likes = int(np.expm1(nivel + (1.0 if tutorial else 0.0) + rng.normal(0, 0.4)))
            linhas.append({
                "post_id": f"p{c}_{k}",
                "channel_id": f"canal{c}",
                "published_at": pd.Timestamp("2023-01-02") + pd.Timedelta(days=7 * k + c),
                "caption": ("tutorial de make passo a passo" if tutorial else "meu look de hoje") + f" {k}",
                "media_ref": "",
                "metric_target": likes,
                "channel_stats_at_collection": "",
                "sector": "beleza",
            })
    return pd.DataFrame(linhas)


def test_esquema_exige_as_colunas_do_tapi():
    df = base_sintetica(1, 3).drop(columns=["media_ref"])
    with pytest.raises(ValueError, match="media_ref"):
        tc.ler_base(df)


def test_data_sem_fuso_e_horario_de_brasilia_e_tags_viram_lista():
    df = base_sintetica(1, 2).assign(published_at="2026-10-20 18:00", tags="make|beleza")
    df["post_id"] = ["a", "b"]
    b = tc.ler_base(df)
    assert b.published_at.iloc[0] == pd.Timestamp("2026-10-20 21:00", tz="UTC")
    assert b.tags.iloc[0] == ["make", "beleza"]


def test_setor_fora_do_contrato_e_recusado():
    df = base_sintetica(1, 2).assign(sector="games")
    with pytest.raises(Exception):  # noqa: B017 — erro do pandera
        tc.ler_base(df)


def test_nome_do_modelo_e_validado():
    with pytest.raises(ValueError):
        tc.caminho("../fora")


@pytest.mark.slow
def test_treino_de_ponta_a_ponta_aprende_o_sinal(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "PASTA", tmp_path)
    monkeypatch.setattr(tc, "_registrar", lambda r: None)
    r = tc.treinar_modelo_cliente(base_sintetica(), "teste", coletado_em=datetime(2026, 10, 1))
    m = r.metricas_teste
    assert m["modelo"]["mae_log"] < m["baseline_mediana_canal"]["mae_log"]
    assert m["modelo"]["acuracia_par_a_par"] > 0.6
    assert (tmp_path / "teste.joblib").exists() and (tmp_path / "teste.json").exists()
    assert "Formato tutorial/dica declarado no texto" in r.principais_fatores[:5] or any(
        "legenda" in f for f in r.principais_fatores[:5]
    )
