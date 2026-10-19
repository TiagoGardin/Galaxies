"""Demonstração executável: o agente em criativos de canais FORA do treino, com o de-para (previsto × real).

Uso: uv run python -m galaxies_tool.demo [--sem-video] [--salvas]
Saída: resultados/demo/<canal>.json (respostas completas da tool, o plano B se a rede cair na apresentação) e
resultados/demo/de_para.csv.

Regra de escolha, fixada ANTES de ver qualquer previsão (sem cherry-picking): para cada um dos 4 canais de
demonstração (fora de treino, validação e teste; ver docs/EXPERIMENTO.md), os 3 Shorts elegíveis mais recentes
da base. Cada trio passa pela Tool 5 (ranking), como a persona faria com os criativos de uma campanha, com a
data real de publicação como data planejada e o título como legenda planejada. O agente nunca vê as curtidas
desses vídeos: elas só aparecem aqui, depois, para o de-para.

`--salvas` só reimprime as respostas já salvas (sem rede, sem modelo).
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from galaxies_tool.contrato import FUSO

PASTA = Path("resultados/demo")
POR_CANAL = 3


def criativos_da_demo() -> pd.DataFrame:
    base = pd.read_parquet("data/processed/base_v1.parquet")
    demo = base[base.is_demo_channel & base.elegivel_treino].sort_values("published_at")
    canais = pd.read_csv("data/canais.csv", dtype=str).set_index("channel_id")
    escolhidos = demo.groupby("channel_id").tail(POR_CANAL).copy()
    escolhidos["canal"] = escolhidos.channel_id.map(canais.titulo)
    return escolhidos


def imprimir(canal: str, resposta: dict, reais: dict[str, int]):
    print(f"\n=== {canal}")
    for r in resposta["ranking"]:
        p = next(x for x in resposta["previsoes"] if x["id"] == r["id"])
        flag = " [baixa confiança]" if r["confidence_flag"] else ""
        print(
            f"  {r['posicao']}º {r['id']}: previsto {r['predicted_metric']:,} (80%: {r['interval_low']:,}–"
            f"{r['interval_high']:,}) | real {reais[r['id']]:,}{flag}".replace(",", ".")
        )
        fatores = "; ".join(
            f"{d['descricao']} = {d['valor']} ({d['impacto_pct']:+.0f}%)" for d in p["top_drivers"][:3]
        )
        print(f"     fatores: {fatores}")
    print(f"  explicação: {resposta['explanation']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sem-video", action="store_true", help="não baixa o vídeo nem chama o VLM (mais rápido)"
    )
    ap.add_argument("--salvas", action="store_true", help="só reimprime as respostas salvas (plano B)")
    args = ap.parse_args()
    PASTA.mkdir(parents=True, exist_ok=True)
    escolhidos = criativos_da_demo()
    reais = dict(zip(escolhidos.video_id, escolhidos.likes.astype(int), strict=True))
    linhas = []
    for cid, g in escolhidos.groupby("channel_id", sort=False):
        nome = g.canal.iloc[0]
        arquivo = PASTA / f"{cid}.json"
        if args.salvas:
            resposta = json.loads(arquivo.read_text())
        else:
            from galaxies_tool.agente import ordenar_criativos

            entradas = [
                {
                    "id": r.video_id,
                    "creative": r.url,
                    "caption": r.title,
                    "description": r.description,
                    "tags": list(r.tags) if r.tags is not None else [],
                    "planned_datetime": r.published_at.tz_convert(FUSO).isoformat(),
                    "channel_id": cid,
                    "sector": r.sector,
                    "analisar_video": not args.sem_video,
                }
                for r in g.itertuples()
            ]
            resposta = ordenar_criativos(entradas).model_dump(mode="json")
            arquivo.write_text(json.dumps(resposta, ensure_ascii=False, indent=2))
        imprimir(nome, resposta, reais)
        for p in resposta["previsoes"]:
            linhas.append(
                {
                    "canal": nome,
                    "video_id": p["id"],
                    "url": f"https://www.youtube.com/shorts/{p['id']}",
                    "previsto": p["predicted_metric"],
                    "baixo": p["interval_low"],
                    "alto": p["interval_high"],
                    "real": reais[p["id"]],
                    "baixa_confianca": p["confidence_flag"],
                }
            )
    t = pd.DataFrame(linhas)
    t["dentro_do_intervalo"] = (t.real >= t.baixo) & (t.real <= t.alto)
    t.to_csv(PASTA / "de_para.csv", index=False)
    ordem = [spearmanr(g.previsto, g.real).statistic for _, g in t.groupby("canal")]
    print(
        f"\nDe-para em {len(t)} criativos de {t.canal.nunique()} canais fora do treino: real dentro do "
        f"intervalo de 80% em {t.dentro_do_intervalo.sum()} de {len(t)}; Spearman previsto × real dentro de "
        f"cada canal: {', '.join(f'{v:+.2f}' for v in ordem)}. É uma demonstração, não uma avaliação: "
        f"com {len(t)} vídeos, os números oscilam muito (a avaliação está em docs/EXPERIMENTO.md)."
    )


if __name__ == "__main__":
    main()
