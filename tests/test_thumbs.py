import numpy as np
import pytest
from PIL import Image

from galaxies_tool.thumbs import estatisticas_cor, recortar_quadro


def test_recorte_16x9_pega_o_quadro_central():
    im = Image.new("RGB", (1280, 720), (0, 0, 0))
    im.paste(Image.new("RGB", (405, 720), (255, 0, 0)), (437, 0))  # quadro vertical no centro
    q = recortar_quadro(im)
    assert q.size == (405, 720)
    assert np.asarray(q)[:, 2:-2].mean(axis=(0, 1)).tolist() == pytest.approx([255, 0, 0])


def test_recorte_4x3_remove_faixas_e_laterais():
    im = Image.new("RGB", (480, 360), (0, 0, 0))
    im.paste(Image.new("RGB", (480, 270), (0, 0, 255)), (0, 45))  # área 16:9 entre faixas pretas
    q = recortar_quadro(im)
    assert q.size == (152, 270)
    assert np.asarray(q).mean(axis=(0, 1)).tolist() == pytest.approx([0, 0, 255])


def test_estatisticas_de_cor():
    preto = estatisticas_cor(np.zeros((10, 10, 3), dtype=np.uint8))
    assert preto["thumb_brilho"] == 0 and preto["thumb_colorfulness"] == 0
    vermelho = estatisticas_cor(np.full((10, 10, 3), (255, 0, 0), dtype=np.uint8))
    assert vermelho["thumb_brilho"] == pytest.approx(1) and vermelho["thumb_saturacao"] == pytest.approx(1)
    assert vermelho["thumb_contraste"] == 0                 # imagem uniforme não tem contraste
    cinza = estatisticas_cor(np.full((10, 10, 3), 128, dtype=np.uint8))
    assert cinza["thumb_saturacao"] == 0 and cinza["thumb_colorfulness"] == 0
    assert vermelho["thumb_colorfulness"] > 0.3            # cor saturada é "colorida"
