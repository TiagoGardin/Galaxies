import numpy as np
import pandas as pd
import pytest

from galaxies_tool import artefato


class _Const:
    def __init__(self, v):
        self.v = v

    def predict(self, x):
        return np.full(len(x), self.v, dtype=float)


def _art(baixo=-1.0, alto=1.0):
    return {"ponto": _Const(0.0), "ponto_catboost": _Const(0.0), "quantis": {0.1: _Const(baixo), 0.9: _Const(alto)},
            "cqr": 0.0}


def test_cqr_alarga_quando_os_quantis_subcobrem(monkeypatch):
    rng = np.random.default_rng(0)
    y = rng.normal(0, 2, 2000)  # quantis fixos em ±1 cobrem só ~38%
    calib = pd.DataFrame({"alvo": y, "published_at": pd.Timestamp("2026-05-01", tz="UTC")})
    art = _art()
    art["limiares"] = {}
    art["ood"] = None
    monkeypatch.setattr(artefato, "preparar", lambda df, a: pd.DataFrame(index=df.index))
    monkeypatch.setattr(artefato, "distancia_ood", lambda a, x: np.full(len(x), np.nan))
    artefato.calibrar(art, calib)
    p = artefato.prever(art, pd.DataFrame(index=range(len(y))))
    cob = np.mean((y >= p.alvo_baixo) & (y <= p.alvo_alto))
    assert art["cqr"] > 0 and cob == pytest.approx(0.80, abs=0.01)


def test_limites_nunca_cruzam_o_ponto():
    p = artefato.prever(_art(baixo=0.5, alto=-0.5), pd.DataFrame(index=range(3)))
    assert (p.alvo_baixo <= p.alvo_previsto).all() and (p.alvo_previsto <= p.alvo_alto).all()


def _art_flag():
    return {
        "limiares": {
            "min_historico": 10,
            "std_instavel": 1.0,
            "largura_larga": 2.0,
            "distancia_ood": 5.0,
            "ultima_data_vista": "2026-07-01",
            "max_dias_futuro": 180,
            "max_dias_passado": 365,
        },
        "setores_treino": ["beleza", "moda"],
    }


def test_motivos_de_baixa_confianca_com_codigos():
    linha = {
        "n_historico_total": 25,
        "std_log_likes_ult20": 0.5,
        "setor": "beleza",
        "published_at": pd.Timestamp("2026-10-20", tz="UTC"),
    }
    assert artefato.motivos(_art_flag(), linha, 1.0, 1.0, []) == []
    ruim = {
        "n_historico_total": 4,
        "std_log_likes_ult20": 1.5,
        "setor": "automotivo",
        "published_at": pd.Timestamp("2027-06-01", tz="UTC"),
    }
    codigos = [m["codigo"] for m in artefato.motivos(_art_flag(), ruim, 3.0, 9.0, ["vídeo"])]
    assert codigos == [
        "canal_pouco_historico",
        "canal_instavel",
        "intervalo_largo",
        "fora_da_distribuicao",
        "setor_fora_do_treino",
        "extracao_incompleta",
        "data_fora_do_periodo",
    ]
