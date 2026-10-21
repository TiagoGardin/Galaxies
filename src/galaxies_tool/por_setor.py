"""Um modelo por setor × modelo global, só na validação (docs/EXPERIMENTO.md, "Um modelo por setor").

Uso: uv run python -m galaxies_tool.por_setor
Saída: resultados/validacao_por_setor.csv (métricas) e resultados/validacao_por_setor_diferencas.csv (IC 95%)

Mesma receita da v2 (LightGBM, `configs/lgbm.yaml`, PCA 32 + comparáveis), com os PCAs reajustados em cada
treino. O modelo global é treinado com os dois setores; cada modelo por setor, só com o próprio setor (sem a
coluna `setor`, que seria constante). Os dois são comparados na validação de cada setor, com IC 95% por
bootstrap pareado de canais. O teste e o teste prospectivo não são usados: os dois já foram avaliados.
"""

import numpy as np
import pandas as pd

from galaxies_tool import modelo
from galaxies_tool.metricas import bootstrap_diferenca

VARIANTE = "v2"
COLS_TABELA = ["n", "mae_log", "spearman_intra_canal", "acuracia_par_a_par", "acuracia_par_a_par_7d"]


def main():
    df = modelo.carregar(rag=True)
    treino, valid = df[df.split == "treino"].copy(), df[df.split == "validacao"].copy()
    params = modelo.params_lgbm()
    pcas = modelo.ajustar_pcas(treino, VARIANTE)
    tg, vg = modelo.aplicar_pcas(treino, pcas), modelo.aplicar_pcas(valid, pcas)
    cols = modelo.colunas_completas(tg, VARIANTE)
    global_, _ = modelo.treinar_lgbm(tg, cols, params)
    p_global = pd.Series(global_.predict(modelo.matriz(vg, cols)), index=vg.index)

    linhas, deltas = [], []
    for setor in ("beleza", "moda"):
        ts, vs = treino[treino.setor == setor].copy(), valid[valid.setor == setor].copy()
        ps = modelo.ajustar_pcas(ts, VARIANTE)
        ts, vs = modelo.aplicar_pcas(ts, ps), modelo.aplicar_pcas(vs, ps)
        cs = [c for c in cols if c != "setor"]
        m, n_arvores = modelo.treinar_lgbm(ts, cs, params)
        prev = {"global": p_global.loc[vs.index].to_numpy(), "por setor": m.predict(modelo.matriz(vs, cs))}
        for nome, p in prev.items():
            linhas.append({"setor": setor, "modelo": nome, "n_treino": len(ts) if nome == "por setor" else len(tg),
                           **modelo.metricas_em(vs, p)})
        bd = bootstrap_diferenca(vs.alvo, prev["por setor"], prev["global"], vs.channel_id, vs.published_at)
        deltas.append({"setor": setor, "comparacao": "por setor − global", "n": len(vs),
                       **{k: modelo._fmt_delta(bd, k, 4 if k == "mae_log" else 3) for k in bd}})
        print(f"{setor}: {len(ts)} vídeos de treino, {n_arvores} árvores", flush=True)
    t, d = pd.DataFrame(linhas), pd.DataFrame(deltas)
    t.to_csv(modelo.RESULTADOS / "validacao_por_setor.csv", index=False)
    d.to_csv(modelo.RESULTADOS / "validacao_por_setor_diferencas.csv", index=False)
    print(t.set_index(["setor", "modelo"])[COLS_TABELA].round(4).to_string())
    print(d.to_string(index=False))
    return t, d


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
