import json

import numpy as np
import pandas as pd
import pytest

from galaxies_tool import multimodal


def test_categorias_e_modelo_iguais_aos_do_esquema_do_vlm():
    from galaxies_tool import vlm

    assert multimodal.MODELO_VLM == vlm.MODELO
    esquema = vlm.Rubrica.model_fields
    for campo, cats in multimodal.VLM_CATEGORICAS.items():
        assert [e.value for e in esquema[campo].annotation] == cats
    usados = set(multimodal.VLM_BINARIAS) | set(multimodal.VLM_NOTAS) | set(multimodal.VLM_CATEGORICAS)
    assert usados == set(esquema) - {"potencial_viral_0a10"}  # o LLM chutando o resultado nunca entra


def test_tabela_vlm_tipa_colunas_e_recusa_outro_modelo(tmp_path):
    rubrica = {
        "forca_hook_0a10": 8,
        "tem_rosto": True,
        "close_up_produto": False,
        "before_after_ou_transformacao": False,
        "try_on_ou_demonstracao": True,
        "swatch": False,
        "aplicacao_na_pele": True,
        "corpo_inteiro_vs_detalhe": "detalhe",
        "texto_na_tela": True,
        "densidade_texto_0a10": 2,
        "ritmo_cortes_0a10": 6,
        "tem_musica_de_fundo": True,
        "audio_original_ou_musica": "ambos",
        "tom_narracao": "animado",
        "estetica_geral_0a10": 7,
        "clareza_beneficio_produto_0a10": 9,
        "formato": "tutorial",
        "potencial_viral_0a10": 10,
    }
    (tmp_path / "a.json").write_text(
        json.dumps({"video_id": "a", "modelo": multimodal.MODELO_VLM, "rubrica": rubrica})
    )
    t = multimodal.tabela_vlm(tmp_path)
    assert t.vlm_tem_rosto.dtype == float and t.vlm_tem_rosto[0] == 1.0
    assert isinstance(t.vlm_formato.dtype, pd.CategoricalDtype) and t.vlm_formato[0] == "tutorial"
    assert "vlm_potencial_viral_0a10" not in t
    (tmp_path / "b.json").write_text(json.dumps({"video_id": "b", "modelo": "outro", "rubrica": rubrica}))
    with pytest.raises(AssertionError):
        multimodal.tabela_vlm(tmp_path)


def test_media_do_gancho_renormalizada():
    from galaxies_tool.thumbs import media_normalizada

    e = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float16)
    m = media_normalizada(e).astype(np.float32)
    assert np.linalg.norm(m) == pytest.approx(1, abs=1e-3) and m[0] == pytest.approx(m[1])
