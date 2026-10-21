"""Avaliação final, ÚNICA, no teste temporal e no teste em canais novos (protocolo de docs/EXPERIMENTO.md).

Uso: uv run python -m galaxies_tool.avaliacao_final
Pré-requisitos: configs/lgbm.yaml e models/lgbm_final.joblib (artefato.py), todas as escolhas congeladas
em docs/EXPERIMENTO.md. Grava resultados/teste_*.csv e resultados/teste_resumo.json, registra no MLflow
(experimento galaxies_teste) e cria resultados/.teste_avaliado: uma segunda execução é recusada, porque o
teste é avaliado uma única vez (--forcar existe só para reproduzir os números, nunca para escolher modelo).

O que sai daqui:
1. Tabela principal (H1, H2): baselines × modelo final, com cobertura do intervalo e IC 95% das diferenças.
2. Ablação por bloco (H3, H5): a mesma escada e as mesmas variantes da validação, treinadas em treino +
   validação, como o modelo final.
3. Recortes por setor, tipo de canal e histórico (H4), cobertura por grupo e canais novos (H4).
4. Avaliação do confidence_flag: erro dos sinalizados × não sinalizados, por motivo.
5. Análise de erros (roteiro do EXPERIMENTO): por faixa real, porte, histórico, instabilidade e os 20
   maiores erros com os fatores do SHAP.
6. Extensão: LLM zero-shot (potencial_viral_0a10 do VLM) × modelo, na amostra com VLM.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from galaxies_tool import modelo, previsao
from galaxies_tool.artefato import contribuicoes, motivos_em_lote, preparar, prever
from galaxies_tool.metricas import avaliar, bootstrap_diferenca, faixa

RES = modelo.RESULTADOS
TRAVA = RES / ".teste_avaliado"
BASE = Path("data/processed/base_v1.parquet")


def _metricas(df, p, baixo=None, alto=None) -> dict:
    return avaliar(df.alvo, p, df.channel_id, df.log_mediana_likes_ult20, baixo, alto, datas=df.published_at)


def _delta(df, p_novo, p_ref) -> dict:
    bd = bootstrap_diferenca(df.alvo, p_novo, p_ref, df.channel_id, df.published_at)
    return {k: modelo._fmt_delta(bd, k, 4 if k == "mae_log" else 3) for k in bd}


def prever_com_artefato(art: dict, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = preparar(df, art)
    return x, prever(art, x)


def por_grupo(df: pd.DataFrame, p: pd.DataFrame, coluna: str) -> pd.DataFrame:
    linhas = []
    for g, idx in df.groupby(coluna, observed=True).groups.items():
        d, q = df.loc[idx], p.loc[idx]
        r = _metricas(d, q.alvo_previsto, q.alvo_baixo, q.alvo_alto)
        r0 = _metricas(d, np.zeros(len(d)))
        linhas.append(
            {
                "grupo": coluna,
                "valor": str(g),
                **r,
                "largura_media": float((q.alvo_alto - q.alvo_baixo).mean()),
                "mae_log_baseline": r0["mae_log"],
            }
        )
    return pd.DataFrame(linhas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forcar", action="store_true")
    args = ap.parse_args()
    if TRAVA.exists() and not args.forcar:
        raise SystemExit(f"O teste já foi avaliado ({TRAVA.read_text().strip()}). Ver docstring.")
    # sempre a v1: é o artefato que o protocolo congelou para o teste (a v2 foi escolhida depois dele e só é
    # medida no teste prospectivo); --forcar reproduz os números de 19/10, nunca avalia outro modelo
    art = previsao.carregar_artefato(str(previsao.V1))
    params = art["params"]
    df = modelo.carregar()
    treino = df[df.split.isin(["treino", "validacao"])].copy()  # como o modelo final (artefato.py)
    teste = df[df.split == "teste"].copy()
    novos = df[df.split == "teste_canais_novos"].copy()
    teste, novos = (modelo.aplicar_pcas(d, art["pcas"]).reset_index(drop=True) for d in (teste, novos))
    inscritos = pd.read_parquet(BASE, columns=["video_id", "channel_subscribers_at_collection", "url"])
    teste = teste.merge(inscritos, on="video_id", how="left")
    RES.mkdir(exist_ok=True)
    resumo = {
        "avaliado_em": datetime.now(UTC).isoformat(timespec="seconds"),
        "modelo": art["versao"],
        "n_teste": len(teste),
        "n_teste_com_video": int(teste.tem_video.sum()),
        "n_teste_com_vlm": int(teste.tem_vlm.sum()),
        "n_canais_novos": len(novos),
    }

    # 1. tabela principal
    x, p = prever_com_artefato(art, teste)
    base = modelo.baselines(treino, teste)
    principal = {nome: _metricas(teste, prev) for nome, prev in base.items()}
    principal["modelo final (A+B+C+D)"] = _metricas(teste, p.alvo_previsto, p.alvo_baixo, p.alvo_alto)
    principal["modelo final (A+B+C+D)"]["largura_media"] = float((p.alvo_alto - p.alvo_baixo).mean())
    pd.DataFrame(principal).T.to_csv(RES / "teste_principal.csv")
    resumo["principal"] = principal
    resumo["final_vs_baselines"] = {
        nome: _delta(teste, p.alvo_previsto.to_numpy(), prev) for nome, prev in base.items()
    }

    # 2. ablação: mesmos conjuntos, hiperparâmetros e nº de árvores da validação, reajustados em treino +
    # validação, como o modelo final
    treino_pca = modelo.aplicar_pcas(treino, art["pcas"])
    val = pd.read_csv(RES / "validacao_metricas.csv").dropna(subset=["n_arvores"])
    arvores = dict(zip(val.modelo, val.n_arvores.astype(int), strict=True))
    tabela, deltas = modelo.rodar_ablacao(treino_pca, teste, params, "galaxies_teste", "teste", arvores)
    tabela.to_csv(RES / "teste_ablacao_metricas.csv", index=False)
    deltas.to_csv(RES / "teste_ablacao.csv", index=False)
    modelo.imprimir(tabela, deltas)

    # 3. recortes, cobertura por grupo e canais novos
    teste["faixa_historico"] = pd.cut(
        teste.n_shorts_anteriores, [0, 9, 19, np.inf], labels=["5–9", "10–19", "20+"]
    )
    teste["porte"] = pd.cut(
        teste.channel_subscribers_at_collection,
        [0, 20e3, 100e3, 500e3, np.inf],
        labels=["<20k", "20–100k", "100–500k", ">500k"],
    )
    teste["instabilidade"] = pd.qcut(
        teste.std_log_likes_ult20, 4, labels=["Q1 (estável)", "Q2", "Q3", "Q4 (instável)"]
    )
    teste["faixa_real"] = faixa(teste.alvo.to_numpy())
    recortes = pd.concat(
        [
            por_grupo(teste, p, c)
            for c in ("setor", "tipo_canal", "faixa_historico", "porte", "instabilidade", "faixa_real")
        ]
    )
    recortes.to_csv(RES / "teste_recortes.csv", index=False)
    xn, pn = prever_com_artefato(art, novos)
    base_n = modelo.baselines(treino, novos)
    canais_novos = {nome: _metricas(novos, prev) for nome, prev in base_n.items()}
    canais_novos["modelo final (A+B+C+D)"] = _metricas(novos, pn.alvo_previsto, pn.alvo_baixo, pn.alvo_alto)
    pd.DataFrame(canais_novos).T.to_csv(RES / "teste_canais_novos.csv")
    resumo["canais_novos"] = canais_novos
    resumo["canais_novos_vs_baseline"] = _delta(
        novos, pn.alvo_previsto.to_numpy(), base_n["baseline_mediana_canal"]
    )

    # 4. confidence_flag
    flags = motivos_em_lote(
        art, teste.assign(n_historico_total=teste.n_shorts_anteriores, esperava_video=True), x, p
    )
    teste["baixa_confianca"] = [bool(m) for m in flags]
    teste["erro_abs"] = (teste.alvo - p.alvo_previsto).abs()
    teste["coberto"] = (teste.alvo >= p.alvo_baixo) & (teste.alvo <= p.alvo_alto)
    linhas = [
        {
            "grupo": "sinalizados" if b else "não sinalizados",
            "n": int(len(g)),
            "mae_log": g.erro_abs.mean(),
            "cobertura": g.coberto.mean(),
        }
        for b, g in teste.groupby("baixa_confianca")
    ]
    for codigo in sorted({m["codigo"] for ms in flags for m in ms}):
        tem = np.array([any(m["codigo"] == codigo for m in ms) for ms in flags])
        linhas.append(
            {
                "grupo": f"motivo: {codigo}",
                "n": int(tem.sum()),
                "mae_log": teste.erro_abs[tem].mean(),
                "cobertura": teste.coberto[tem].mean(),
            }
        )
    pd.DataFrame(linhas).to_csv(RES / "teste_confidence_flag.csv", index=False)
    resumo["confidence_flag"] = linhas

    # 5. análise de erros: os 20 maiores, com os 3 fatores de maior |SHAP|
    contrib = contribuicoes(art, x)[:, :-1]
    shap = pd.DataFrame(contrib, columns=art["colunas"]).T.groupby(previsao.grupo).sum().T
    teste["erro"] = p.alvo_previsto - teste.alvo
    piores = teste.reindex(teste.erro.abs().sort_values(ascending=False).index[:20])
    linhas = []
    for i, r in piores.iterrows():
        top = shap.loc[i].reindex(shap.loc[i].abs().sort_values(ascending=False).index[:3])
        linhas.append(
            {
                "url": r.url,
                "setor": r.setor,
                "tipo_canal": r.tipo_canal,
                "curtidas": int(r.likes),
                "mediana_canal": round(float(np.expm1(r.log_mediana_likes_ult20))),
                "razao_real": round(float(np.exp(r.alvo)), 2),
                "razao_prevista": round(float(np.exp(p.alvo_previsto[i])), 2),
                "baixa_confianca": r.baixa_confianca,
                "top_fatores": "; ".join(f"{g} ({np.expm1(v) * 100:+.0f}%)" for g, v in top.items()),
                "classificacao_manual": "",
            }
        )
    pd.DataFrame(linhas).to_csv(RES / "teste_maiores_erros.csv", index=False)

    # 6. extensão: LLM zero-shot × modelo (o potencial_viral nunca entrou no modelo)
    vlm = teste[teste.tem_vlm].copy()
    if len(vlm):
        from galaxies_tool.multimodal import PASTA_VLM

        nota = {
            v: json.loads((PASTA_VLM / f"{v}.json").read_text())["rubrica"]["potencial_viral_0a10"]
            for v in vlm.video_id
        }
        z = vlm.video_id.map(nota).astype(float).to_numpy()
        z = z + np.random.default_rng(0).normal(0, 1e-6, len(z))  # desempate: notas inteiras empatam muito
        idx = vlm.index
        resumo["llm_zero_shot_vs_modelo"] = {
            "n": len(vlm),
            "llm_zero_shot": _metricas(vlm, z - z.mean()),
            "modelo": _metricas(vlm, p.alvo_previsto[idx]),
            "nota": "só a ordenação é comparável: a nota 0–10 não está na escala do alvo (MAE não se aplica)",
        }

    (RES / "teste_resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2, default=str))
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("galaxies_teste")
    with mlflow.start_run(run_name=f"avaliacao_final_{art['versao']}"):
        mlflow.set_tags({"avaliado_em": "teste", **modelo.contexto_registro()})
        mlflow.log_metrics(
            {
                k: float(v)
                for k, v in principal["modelo final (A+B+C+D)"].items()
                if isinstance(v, (int, float)) and not np.isnan(v)
            }
        )
        for arq in sorted(RES.glob("teste_*")):
            mlflow.log_artifact(str(arq))
    TRAVA.write_text(f"avaliado em {resumo['avaliado_em']} com {art['versao']}\n")
    print(
        "\n### Principal\n"
        + pd.DataFrame(principal).T[modelo.COLS_TABELA + ["cobertura_80"]].round(4).to_string()
    )
    print("\n### Canais novos\n" + pd.DataFrame(canais_novos).T[modelo.COLS_TABELA].round(4).to_string())
    print("\n### confidence_flag\n" + pd.DataFrame(resumo["confidence_flag"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
