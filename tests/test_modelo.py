import numpy as np
import pandas as pd

from galaxies_tool import modelo


def test_embeddings_alinhados_por_id_e_nan_quando_falta(tmp_path):
    arq = tmp_path / "emb_x.npy"
    np.save(arq, np.array([[1, 1], [2, 2], [3, 3]], dtype=np.float16))
    (tmp_path / "emb_x_ids.txt").write_text("b\na\nc")
    tab = modelo._embeddings(arq, "e_", pd.Series(["a", "z", "b"]))
    assert tab.e_0.tolist()[0] == 2 and np.isnan(tab.e_0[1]) and tab.e_0[2] == 1


def test_pca_ajustado_so_no_treino_e_nan_preservado():
    rng = np.random.default_rng(0)
    cols = [f"e_{i}" for i in range(20)]
    treino = pd.DataFrame(rng.normal(size=(100, 20)), columns=cols)
    outro = pd.DataFrame(rng.normal(size=(5, 20)) + 1000, columns=cols)  # valores absurdos: não podem afetar o ajuste
    outro.iloc[0] = np.nan
    nomes, (t1, o1) = modelo.adicionar_pca(treino, [outro], "e_", "x")
    _, (t2, _) = modelo.adicionar_pca(treino, [outro.iloc[:1]], "e_", "x")
    assert len(nomes) == modelo.N_PCA
    np.testing.assert_allclose(t1[nomes], t2[nomes])          # o treino não depende dos outros conjuntos
    assert o1.loc[0, nomes].isna().all() and o1.loc[1:, nomes].notna().all().all()
