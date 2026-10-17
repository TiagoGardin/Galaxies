from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from galaxies_tool.contrato import FUSO, EntradaCriativo
from galaxies_tool.extracao import tipo_do_criativo

TEM_DADOS = Path("models/lgbm_dev.joblib").exists() and Path("data/features/features_ab.parquet").exists()


def test_tipo_do_criativo_reconhece_urls_do_youtube():
    for url in (
        "https://www.youtube.com/shorts/aNEs_mSlsvY",
        "https://youtu.be/aNEs_mSlsvY",
        "https://www.youtube.com/watch?v=aNEs_mSlsvY&t=3",
    ):
        assert tipo_do_criativo(url) == ("youtube", "aNEs_mSlsvY")
    assert tipo_do_criativo("Tutorial de contorno #maquiagem") == ("texto", None)


def test_tipo_do_criativo_arquivo_inexistente_da_erro():
    with pytest.raises(FileNotFoundError):
        tipo_do_criativo("/nao/existe/video.mp4")


def test_entrada_exige_canal_e_assume_horario_de_brasilia():
    with pytest.raises(ValidationError):
        EntradaCriativo(creative="x", planned_datetime="2026-10-20T18:00", sector="moda")
    e = EntradaCriativo(creative="x", planned_datetime="2026-10-20T18:00", sector="moda", channel_id="UC1")
    assert e.planned_datetime == datetime(2026, 10, 20, 18, 0, tzinfo=FUSO)


def test_explicacao_padrao_so_usa_os_numeros_recebidos():
    from galaxies_tool.agente import explicacao_padrao

    f = {
        "curtidas_previstas": 1234,
        "intervalo_80": [500, 4000],
        "mediana_do_canal": 1000,
        "razao_vs_canal": 1.23,
        "faixa": "na média do canal",
        "baixa_confianca": ["canal com só 3 Shorts"],
        "fatores": [
            {"descricao": "Hora de publicação", "valor": "18h", "efeito": "aumenta", "impacto_pct": 7.4}
        ],
    }
    texto = explicacao_padrao(f)
    assert "1.234" in texto and "500" in texto and "4.000" in texto and "18h" in texto
    assert "Baixa confiança" in texto


def test_historico_do_cliente_igual_ao_da_base():
    """channel_stats com os mesmos posts da base produz as mesmas features de canal que channel_id."""
    if not TEM_DADOS:
        pytest.skip("sem dados locais")
    from galaxies_tool import canal

    base = canal._base()
    cid = base.channel_id.iloc[0]
    posts = base[base.channel_id == cid]
    planned = pd.Timestamp("2026-08-01", tz="UTC")
    a = canal.features_do_canal(planned, channel_id=cid)
    b = canal.features_do_canal(planned, posts=list(zip(posts.published_at, posts.likes, strict=True)))
    for k in ("log_mediana_likes_ult20", "std_log_likes_ult20", "tendencia", "n_shorts_anteriores"):
        assert a[k] == pytest.approx(b[k], nan_ok=True)


@pytest.mark.slow
def test_numero_do_agente_igual_ao_do_modelo():
    """Para vídeos da validação, o caminho do agente (extração + histórico + previsão) reproduz as features
    do treino e a previsão do modelo avaliado no experimento."""
    if not TEM_DADOS:
        pytest.skip("sem dados locais")
    from galaxies_tool import canal, extracao, modelo, previsao
    from galaxies_tool.agente import linha_do_modelo
    from galaxies_tool.artefato import preparar, prever

    df = modelo.carregar()
    art = previsao.carregar_artefato("models/lgbm_dev.joblib")
    amostra = df[df.split == "validacao"].sample(4, random_state=1)
    ref = prever(art, preparar(amostra, art)).alvo_previsto.to_numpy()
    obtido = []
    for _, r in amostra.iterrows():
        atr = extracao.extrair_isolado(f"https://www.youtube.com/shorts/{r.video_id}")
        ctx = {
            **canal.features_do_canal(r.published_at, channel_id=r.channel_id),
            **canal.features_de_tempo(r.published_at),
            "setor": r.setor,
            "tipo_canal": r.tipo_canal,
        }
        obtido.append(previsao.prever_linha(linha_do_modelo(atr, ctx), art)["alvo_previsto"])
    np.testing.assert_allclose(obtido, ref, atol=1e-6)


def test_agente_nao_importa_pytorch_no_processo_principal():
    """PyTorch e LightGBM no mesmo processo derrubam o Python no macOS (ver extracao.py)."""
    import subprocess
    import sys

    codigo = "import sys; import galaxies_tool.servidor_mcp; print('torch' in sys.modules)"
    r = subprocess.run(
        [sys.executable, "-c", codigo], check=False, capture_output=True, text=True, timeout=300
    )
    assert r.stdout.strip().endswith("False"), r.stderr[-2000:]
