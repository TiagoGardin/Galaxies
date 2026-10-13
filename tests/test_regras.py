import pandas as pd
import pytest

from galaxies_tool.processar import contar_historico
from galaxies_tool.regras import CORTE_HISTORICO, eh_short, texto_pos_publicacao
from galaxies_tool.splits import sortear_canais_novos
from galaxies_tool.youtube_api import duracao_segundos


@pytest.mark.parametrize("iso, esperado", [
    ("PT59S", 59), ("PT1M", 60), ("PT3M", 180), ("PT1H2M3S", 3723), ("P0D", 0), ("", -1), ("lixo", -1),
])
def test_duracao_segundos(iso, esperado):
    assert duracao_segundos(iso) == esperado


@pytest.mark.parametrize("duracao, publicado, esperado", [
    (60, "2024-10-14T23:59:59Z", True),    # antes da mudança: até 60 s
    (30, "2021-06-30T23:59:59Z", False),   # antes de existir Shorts
    (30, "2021-07-01T00:00:00Z", True),
    (61, "2024-10-14T23:59:59Z", False),
    (61, "2024-10-15T00:00:00Z", True),    # depois da mudança: até 180 s
    (180, "2025-06-01T00:00:00Z", True),
    (181, "2025-06-01T00:00:00Z", False),
    (0, "2025-06-01T00:00:00Z", False),    # live/estreia sem duração
    (-1, "2025-06-01T00:00:00Z", False),
])
def test_eh_short(duracao, publicado, esperado):
    assert eh_short(duracao, publicado) is esperado


def test_corte_historico():
    assert CORTE_HISTORICO == "2025-03-02T00:00:00Z"


@pytest.mark.parametrize("titulo, descricao", [
    ("ESSE VÍDEO VIRALIZOU 😱", ""),
    ("Meu vídeo bombou", ""),
    ("look do dia", "obrigada pelos 100 mil views!!"),
    ("make rápida", "Obrigado por mais de 1 milhão de visualizações"),
    ("1 milhão de views nesse tutorial", ""),
    ("2,5M views", ""),
])
def test_texto_pos_publicacao_detecta(titulo, descricao):
    assert texto_pos_publicacao(titulo, descricao)


@pytest.mark.parametrize("titulo, descricao", [
    ("Testei a base que viralizou no TikTok", ""),     # quem viralizou é o produto
    ("BLUSH VIRALIZADO DA SHEGLAM", ""),
    ("look do dia", "obrigada por assistir"),
    ("5 dicas de maquiagem", "mil beijos"),
    ("10 mil inscritos!!", ""),                         # marco do canal, não do vídeo
])
def test_texto_pos_publicacao_ignora(titulo, descricao):
    assert not texto_pos_publicacao(titulo, descricao)


def test_contar_historico_usa_so_videos_com_30_dias_de_antecedencia():
    df = pd.DataFrame({
        "channel_id": ["a"] * 4 + ["b"],
        "published_at": pd.to_datetime(["2025-01-01", "2025-01-20", "2025-02-01", "2025-03-03", "2025-03-03"],
                                       utc=True),
    })
    # 2025-03-03 menos 30 dias = 2025-02-01 (inclusive): conta 01/01, 20/01 e 01/02
    assert contar_historico(df).tolist() == [0, 0, 1, 3, 0]


def test_sorteio_de_canais_novos_e_estratificado_e_deterministico():
    canais = pd.DataFrame({
        "channel_id": [f"c{i:02d}" for i in range(20)],
        "sector": ["beleza"] * 10 + ["moda"] * 10,
        "channel_type": (["criador"] * 5 + ["marca"] * 5) * 2,
    })
    a = sortear_canais_novos(canais)
    assert a == sortear_canais_novos(canais.sample(frac=1, random_state=1))
    assert len(a) == 4
    estratos = canais.set_index("channel_id").loc[a].groupby(["sector", "channel_type"]).size()
    assert (estratos == 1).all() and len(estratos) == 4


def test_selecionar_alvos_espaca_no_tempo_e_respeita_teto():
    from galaxies_tool.processar import selecionar_alvos
    datas = pd.date_range("2025-04-01", periods=1000, freq="D", tz="UTC")
    df = pd.DataFrame({"channel_id": ["a"] * 1000 + ["b"] * 10,
                       "published_at": list(datas) + list(datas[:10])})
    idx = selecionar_alvos(df, teto=250)
    a = df.loc[idx][lambda d: d.channel_id == "a"]
    assert len(a) == 250 and len(df.loc[idx][lambda d: d.channel_id == "b"]) == 10
    assert a.published_at.min() == datas[0] and a.published_at.max() == datas[-1]  # cobre a janela toda
