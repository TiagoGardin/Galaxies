"""Avaliação ÚNICA do teste prospectivo (protocolo pré-registrado em docs/EXPERIMENTO.md, "Teste prospectivo").

Uso: uv run python -m galaxies_tool.prospectivo avaliar --parte principal   # todos os vídeos, sem vídeo/VLM
     uv run python -m galaxies_tool.prospectivo avaliar --parte video       # subconjunto com vídeo e VLM

Cada parte roda uma vez (trava em resultados/.prospectivo_<parte>); --forcar só reproduz os números.

Compara, nos mesmos vídeos: a mediana do canal e a regressão linear (baselines), o modelo v1 (lgbm_final,
congelado e avaliado no teste temporal em 19/10) e a v2 (lgbm_v2: PCA 32 + features de comparáveis, escolhida
só na validação em 21/10). Os vídeos de canais vistos no treino e os de canais novos são avaliados à parte.
"""

import argparse
import json
from datetime import UTC, datetime

import mlflow
import numpy as np
import pandas as pd

from galaxies_tool import canal, modelo, multimodal, previsao, prospectivo, rag
from galaxies_tool.artefato import motivos_em_lote, preparar, prever
from galaxies_tool.metricas import avaliar, bootstrap_diferenca, faixa

RES = modelo.RESULTADOS
MODELOS = {"v1": "models/lgbm_final.joblib", "v2": "models/lgbm_v2.joblib"}
CORTE_VIDEO = datetime.fromisoformat("2026-10-21T09:00:00-03:00")  # pré-registrado: extração até 09h00
CONJUNTOS = {"canais vistos": "prospectivo", "canais novos": "prospectivo_canais_novos"}


def dados(com_video: bool) -> pd.DataFrame:
    """Vídeos do teste prospectivo com o bloco A, os atributos da Tool 1 e as features RAG (v2)."""
    t = prospectivo.tabela(com_video, CORTE_VIDEO if com_video else None)
    funil = t.attrs["funil"]
    if com_video:
        t = t[t.tem_video | t.tem_vlm].reset_index(drop=True)
    pool, vp = canal.pool_rag()
    vq = rag.vetores_do_df(t)
    t = multimodal.tipar(pd.concat([t, rag.features_rag(t, vq, pool, vp)], axis=1))
    t.attrs["funil"] = funil
    return t


def completar(df: pd.DataFrame, art: dict) -> pd.DataFrame:
    """Colunas que o modelo espera e a extração não produziu (ex.: vídeo e VLM na parte principal) = NaN,
    como em previsao.prever_linha."""
    brutas = [c for p in art["pcas"].values() for c in p["colunas"]]
    geradas = {n for p in art["pcas"].values() for n in p["nomes"]}
    faltam = [c for c in dict.fromkeys([*art["colunas"], *brutas]) if c not in df and c not in geradas]
    return pd.concat([df, pd.DataFrame(np.nan, index=df.index, columns=faltam)], axis=1) if faltam else df


def _metricas(d: pd.DataFrame, p, baixo=None, alto=None) -> dict:
    return avaliar(d.alvo, p, d.channel_id, d.log_mediana_likes_ult20, baixo, alto, datas=d.published_at)


def _delta(d: pd.DataFrame, novo, ref) -> dict:
    bd = bootstrap_diferenca(d.alvo, np.asarray(novo), np.asarray(ref), d.channel_id, d.published_at)
    return {k: modelo._fmt_delta(bd, k, 4 if k == "mae_log" else 3) for k in bd}


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--parte", choices=["principal", "video"], default="principal")
    ap.add_argument("--forcar", action="store_true")
    args, _ = ap.parse_known_args(argv)
    trava = RES / f".prospectivo_{args.parte}"
    if trava.exists() and not args.forcar:
        raise SystemExit(f"Esta parte do teste prospectivo já foi avaliada ({trava.read_text().strip()}).")
    df = dados(args.parte == "video")
    hist = pd.read_parquet(modelo.FEATURES)
    treino = hist[hist.split.isin(["treino", "validacao"])]  # baselines ajustadas como no modelo final
    arts = {nome: previsao.carregar_artefato(caminho) for nome, caminho in MODELOS.items()}
    prev, intervalos, flags = {}, {}, {}
    for nome, art in arts.items():
        d = completar(df, art)
        x = preparar(d, art)
        p = prever(art, x)
        prev[nome], intervalos[nome] = p.alvo_previsto.to_numpy(), p
        flags[nome] = motivos_em_lote(art, d.assign(esperava_video=args.parte == "video"), x, p)
    base = modelo.baselines(treino, df)
    resumo = {
        "parte": args.parte,
        "avaliado_em": datetime.now(UTC).isoformat(timespec="seconds"),
        "coleta": str(prospectivo._coleta_mais_recente().name),
        "funil": df.attrs.get("funil"),
        "periodo": [str(df.published_at.min()), str(df.published_at.max())],
        "n_com_video": int(df.tem_video.sum()),
        "n_com_vlm": int(df.tem_vlm.sum()),
    }
    linhas = []
    for nome_conj, split in CONJUNTOS.items():
        m = (df.split == split).to_numpy()
        d = df[m]
        if len(d) == 0:
            continue
        r = {"n": int(m.sum()), "n_canais": int(d.channel_id.nunique()), "modelos": {}, "diferencas": {}}
        for nome_b, pb in base.items():
            r["modelos"][nome_b] = _metricas(d, pb[m])
        for nome, p in intervalos.items():
            r["modelos"][nome] = _metricas(d, p.alvo_previsto[m], p.alvo_baixo[m], p.alvo_alto[m])
            r["modelos"][nome]["vies_mediano"] = float(np.median(p.alvo_previsto[m] - d.alvo))
            r["modelos"][nome]["taxa_baixa_confianca"] = float(np.mean([bool(f) for f in np.array(
                flags[nome], dtype=object)[m]]))
        for novo, ref in (("v1", "baseline_mediana_canal"), ("v2", "baseline_mediana_canal"),
                          ("v1", "baseline_linear"), ("v2", "v1")):
            pn = prev[novo][m] if novo in prev else base[novo][m]
            pr = prev[ref][m] if ref in prev else base[ref][m]
            r["diferencas"][f"{novo} − {ref}"] = _delta(d, pn, pr)
        resumo[nome_conj] = r
        for nome_m, met in r["modelos"].items():
            linhas.append({"conjunto": nome_conj, "modelo": nome_m, **met})

    # recortes da v1 nos canais vistos: setor, tipo de canal, faixa real e idade do vídeo na coleta
    vistos = df[df.split == "prospectivo"].copy()
    pv = intervalos["v1"][(df.split == "prospectivo").to_numpy()]
    vistos["erro"] = pv.alvo_previsto.to_numpy() - vistos.alvo.to_numpy()
    vistos["coberto"] = (vistos.alvo >= pv.alvo_baixo.to_numpy()) & (vistos.alvo <= pv.alvo_alto.to_numpy())
    coleta = pd.Timestamp(
        pd.read_parquet(prospectivo._coleta_mais_recente() / "videos.parquet").collected_at_hoje.iloc[0]
    )
    vistos["idade_dias"] = (coleta - vistos.published_at).dt.days
    vistos["faixa_real"] = faixa(vistos.alvo.to_numpy())
    vistos["baixa_confianca"] = [bool(f) for f in np.array(flags["v1"], dtype=object)[
        (df.split == "prospectivo").to_numpy()]]
    recortes = []
    for col in ("setor", "tipo_canal", "faixa_real", "idade_dias", "baixa_confianca"):
        for valor, g in vistos.groupby(col, observed=True):
            recortes.append({"recorte": col, "valor": str(valor), "n": len(g),
                             "mae_log": float(g.erro.abs().mean()), "vies_mediano": float(g.erro.median()),
                             "mae_log_baseline": float(g.alvo.abs().mean()), "cobertura": float(g.coberto.mean())})
    pd.DataFrame(recortes).to_csv(RES / f"prospectivo_{args.parte}_recortes.csv", index=False)
    pd.DataFrame(linhas).to_csv(RES / f"prospectivo_{args.parte}.csv", index=False)
    (RES / f"prospectivo_{args.parte}_resumo.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2, default=str))

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("galaxies_prospectivo")
    for nome in MODELOS:
        with mlflow.start_run(run_name=f"prospectivo_{args.parte}_{nome}"):
            mlflow.set_tags({"avaliado_em": f"teste prospectivo ({args.parte})", "modelo": MODELOS[nome]})
            met = resumo["canais vistos"]["modelos"][nome]
            mlflow.log_metrics({k: float(v) for k, v in met.items() if isinstance(v, (int, float))
                                and not np.isnan(v)})
    trava.write_text(f"avaliado em {resumo['avaliado_em']}\n")
    cols = ["n", "mae_log", "spearman_intra_canal", "acuracia_par_a_par", "acuracia_par_a_par_7d",
            "cobertura_80", "vies_mediano"]
    t = pd.DataFrame(linhas).set_index(["conjunto", "modelo"])
    print(t.reindex(columns=cols).round(4).to_string())
    for conj in CONJUNTOS:
        if conj in resumo:
            print(f"\n### {conj}: diferenças com IC 95%")
            print(pd.DataFrame(resumo[conj]["diferencas"]).T.to_string())
    print("\n### Recortes (v1, canais vistos)")
    print(pd.DataFrame(recortes).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
