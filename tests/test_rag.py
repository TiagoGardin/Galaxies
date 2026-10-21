"""Features de comparáveis (rag.py): só vizinhos maduros (>= 30 dias antes), nunca o próprio vídeo."""

import numpy as np
import pandas as pd

from galaxies_tool import rag


def _base():
    datas = pd.to_datetime(["2026-01-01", "2026-01-10", "2026-02-15", "2026-03-01", "2026-03-20"], utc=True)
    df = pd.DataFrame(
        {
            "video_id": list("abcde"),
            "channel_id": ["c1", "c1", "c1", "c2", "c1"],
            "setor": "beleza",
            "published_at": datas,
            "alvo": [1.0, -1.0, 0.5, 2.0, 0.0],
        }
    )
    v = np.eye(5, 8, dtype=np.float32)
    v[4] = v[0]  # "e" é idêntico a "a" no conteúdo
    return df, v


def test_so_entram_vizinhos_maduros_e_nunca_o_proprio_video():
    df, v = _base()
    f = rag.features_rag(df, v, df, v)
    assert f.loc[0].isna().all()  # "a" é o primeiro: não há histórico maduro
    assert np.isnan(f.loc[1, "rag_canal_alvo"])  # "b" é só 9 dias depois de "a"
    assert f.loc[2, "rag_canal_recente"] == 0.0  # "c": maduros = a (+1) e b (−1)
    # "e" (20/03): maduros do canal = a, b, c; o mais parecido é "a" (idêntico), similaridade 1
    assert f.loc[4, "rag_canal_sim"] == 1.0
    assert f.loc[4, "rag_canal_alvo"] > 0.5  # puxado para o alvo de "a" (+1)
    # setor: "e" só pode usar "d" de outro canal se "d" tiver 30 dias (01/03 → 20/03 são 19): não pode
    assert np.isnan(f.loc[4, "rag_setor_alvo"])


def test_vetores_sem_thumbnail_usam_so_o_texto():
    t = np.array([[1.0, 0.0]], dtype=np.float32)
    i = np.array([[np.nan, np.nan]], dtype=np.float32)
    assert np.allclose(rag.vetores(t, i), [[1.0, 0.0]])
