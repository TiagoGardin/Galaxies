"""Interface opcional (app.py): o modo de respostas salvas abre sem rede e sem erro."""

from pathlib import Path

import pytest


def test_app_abre_as_respostas_salvas_da_demo():
    pytest.importorskip("streamlit")
    if not Path("resultados/demo/de_para.csv").exists():
        pytest.skip("rode python -m galaxies_tool.demo")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "src/galaxies_tool/app.py"), default_timeout=300).run()
    at.sidebar.radio[0].set_value("Respostas salvas da demonstração (sem rede)").run()
    assert not at.exception
    assert len(at.metric) >= 3 and len(at.get("plotly_chart")) >= 2
