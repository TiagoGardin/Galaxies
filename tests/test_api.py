"""API REST opcional (api.py): mesmas funções e contrato da tool, outro transporte."""

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from galaxies_tool.api import app  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
TEM_MODELO = (RAIZ / "models/lgbm_final.joblib").exists()
cliente = TestClient(app)


def test_saude_e_model_card():
    if not TEM_MODELO:
        pytest.skip("sem artefato")
    r = cliente.get("/saude")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    card = cliente.get("/modelo").json()
    assert card["versao"] == r.json()["modelo"]


def test_entrada_invalida_e_422():
    sem_canal = {"creative": "texto", "planned_datetime": "2026-10-20T18:00", "sector": "moda"}
    assert cliente.post("/predict", json=sem_canal).status_code == 422
    assert cliente.post("/rank", json=[{**sem_canal, "channel_id": "UC1"}]).status_code == 422


def test_treino_com_base_inexistente_e_400():
    r = cliente.post("/train", json={"caminho_base": "/nao/existe.csv", "nome": "x"})
    assert r.status_code == 400


@pytest.mark.slow
def test_predict_devolve_o_contrato():
    if not TEM_MODELO:
        pytest.skip("sem artefato")
    import pandas as pd

    b = pd.read_parquet(RAIZ / "data/processed/base_v1.parquet", columns=["channel_id", "sector"]).iloc[0]
    r = cliente.post("/predict", json={"creative": "Tutorial de contorno em 30 segundos #maquiagem",
                                       "planned_datetime": "2026-10-20T18:00", "channel_id": b.channel_id,
                                       "sector": b.sector})
    assert r.status_code == 200
    s = r.json()
    assert s["interval_low"] <= s["predicted_metric"] <= s["interval_high"] and s["explanation"]
