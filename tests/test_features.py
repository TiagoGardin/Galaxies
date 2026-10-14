from datetime import date

import numpy as np
import pandas as pd
import pytest

from galaxies_tool.features import (
    atividade_de_postagem,
    eh_data_sazonal,
    features_tempo,
    features_texto,
    historico_do_alvo,
    janelas_sazonais,
)


def _ts(datas):
    return pd.to_datetime(datas, utc=True).as_unit("ns")


def test_historico_usa_so_videos_com_30_dias_de_antecedencia():
    base = pd.DataFrame({
        "channel_id": ["a"] * 5,
        "published_at": _ts(["2025-01-01", "2025-01-10", "2025-01-20", "2025-02-25", "2025-03-01"]),
        "likes": [10, 100, 1000, 999_999, 5],
    })
    alvo = pd.DataFrame({"channel_id": ["a"], "published_at": _ts(["2025-03-01"])}, index=[7])
    h = historico_do_alvo(base, alvo).loc[7]
    # 01/03 - 30 dias = 30/01: entram 01/01, 10/01, 20/01; 25/02 (viral) e o próprio alvo ficam de fora
    assert h.n_hist == 3
    assert h.log_mediana_likes_ult20 == pytest.approx(np.log1p(100))
    assert h.std_log_likes_ult20 == pytest.approx(np.std(np.log1p([10, 100, 1000]), ddof=1))
    assert h.tendencia == pytest.approx(0.0)  # com 3 vídeos, os "últimos 5" são os mesmos 3


def test_historico_vazio_e_janela_de_20():
    datas = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC").as_unit("ns")
    base = pd.DataFrame({"channel_id": "a", "published_at": datas, "likes": np.arange(30) * 10})
    alvos = pd.DataFrame({"channel_id": ["a", "a"], "published_at": _ts(["2024-01-15", "2024-04-01"])},
                         index=[1, 2])
    h = historico_do_alvo(base, alvos)
    assert h.loc[1, "n_hist"] == 0 and np.isnan(h.loc[1, "log_mediana_likes_ult20"])
    # em 01/04, todos os 30 vídeos têm >= 30 dias; a janela usa só os 20 últimos (likes 100..290)
    assert h.loc[2, "log_mediana_likes_ult20"] == pytest.approx(np.log1p(np.median(np.arange(10, 30) * 10)))
    assert h.loc[2, "tendencia"] == pytest.approx(np.log1p(270) - np.log1p(195))


def test_atividade_conta_so_shorts_anteriores_ao_alvo():
    shorts = pd.DataFrame({"channel_id": "a", "published_at": _ts(
        ["2025-01-01", "2025-02-20", "2025-03-01", "2025-03-10", "2025-03-10", "2025-03-11"])})
    alvos = pd.DataFrame({"channel_id": ["a", "a"], "published_at": _ts(["2025-03-10", "2025-01-01"])},
                         index=[1, 2])
    a = atividade_de_postagem(shorts, alvos)
    # em 10/03: últimos 30 dias = [08/02, 10/03) → 20/02 e 01/03; o próprio dia 10/03 não conta
    assert a.loc[1, "shorts_ult_30d"] == 2
    assert a.loc[1, "dias_desde_ultimo_post"] == pytest.approx(9.0)
    assert a.loc[2, "shorts_ult_30d"] == 0 and np.isnan(a.loc[2, "dias_desde_ultimo_post"])


def test_fuso_de_sao_paulo():
    t = features_tempo(pd.Series(_ts(["2025-06-01 02:00"])))
    # 02h UTC de domingo = 23h de sábado em São Paulo
    assert t.hora[0] == 23 and t.dia_semana[0] == 5 and bool(t.fim_de_semana[0]) and t.mes[0] == 5


def test_datas_comerciais():
    bf, maes, *_ = janelas_sazonais(2025)
    assert bf == (date(2025, 11, 21), date(2025, 12, 1))       # Black Friday 28/11/2025
    assert maes == (date(2025, 4, 27), date(2025, 5, 11))      # Dia das Mães 11/05/2025
    assert janelas_sazonais(2026)[0][0] == date(2026, 11, 20)  # Black Friday 27/11/2026
    assert eh_data_sazonal(date(2025, 6, 12)) and eh_data_sazonal(date(2025, 12, 24))
    assert not eh_data_sazonal(date(2025, 6, 13)) and not eh_data_sazonal(date(2025, 8, 1))


def test_features_texto():
    f = features_texto("TESTEI a TRANSFORMAÇÃO? 😍😍", "Comenta aqui! @maria https://loja.com #shorts", ["grwm"])
    assert f["fmt_antes_depois"] and f["fmt_review"] and f["fmt_grwm"]
    assert f["tem_cta"] and f["tem_hashtag_shorts"] and f["tem_link"]
    assert f["n_emojis"] == 2 and f["n_mencoes"] == 1 and f["n_hashtags"] == 1
    assert f["titulo_pergunta"] and not f["titulo_exclamacao"]
    assert f["prop_maiusculas"] == pytest.approx(19 / 20)  # 20 letras, só o "a" minúsculo
    vazio = features_texto("", "", None)
    assert vazio["titulo_n_chars"] == 0 and vazio["prop_maiusculas"] == 0 and not vazio["tem_cta"]
    assert not any(v for k, v in vazio.items() if k.startswith("fmt_"))


def test_formatos_nao_disparam_em_palavras_parecidas():
    f = features_texto("Comentei no salvador", "dicionário de resultados", [])
    assert not f["tem_cta"]                 # "comentei" e "salvador" não são CTA
    assert not f["fmt_antes_depois"]        # "resultados" ≠ "resultado" (fronteira de palavra)
    assert not f["fmt_tutorial"]            # "dicionario" não começa com "dica"
