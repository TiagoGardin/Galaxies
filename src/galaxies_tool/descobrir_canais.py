"""Descobre canais candidatos via search.list (Shorts recentes, Brasil, pt).

Uso: uv run python -m galaxies_tool.descobrir_canais
Custo: 100 unidades por consulta. Saída: data/canais_descobertos.csv
"""

from collections import Counter, defaultdict

import pandas as pd

from galaxies_tool.regras import INICIO_JANELA_ALVO
from galaxies_tool.youtube_api import YouTube

CONSULTAS = {
    "beleza": [
        "maquiagem", "tutorial de maquiagem", "make rápida", "skincare rotina", "resenha base",
        "batom", "perfume feminino", "cabelo cacheado", "unhas decoradas", "arrume-se comigo",
        "maquiagem pele negra", "cuidados com a pele",
    ],
    "moda": [
        "look do dia", "provador", "comprinhas", "haul shein", "moda feminina", "moda masculina",
        "como combinar roupa", "tendência moda", "look trabalho", "moda plus size", "sapatos femininos",
        "estilo masculino",
    ],
}


def main():
    yt = YouTube()
    vistos = Counter()
    setores = defaultdict(Counter)
    for setor, consultas in CONSULTAS.items():
        for q in consultas:
            for item in yt.buscar_shorts(q, INICIO_JANELA_ALVO):
                cid = item["snippet"]["channelId"]
                vistos[cid] += 1
                setores[cid][setor] += 1
            print(f"{q!r}: {len(vistos)} canais únicos até agora (cota {yt.cota_usada})")

    canais = yt.canais_por_ids(list(vistos))
    linhas = []
    for c in canais:
        cid = c["id"]
        linhas.append({
            "channel_id": cid,
            "handle": c["snippet"].get("customUrl", ""),
            "titulo": c["snippet"]["title"],
            "pais": c["snippet"].get("country", ""),
            "inscritos": int(c["statistics"].get("subscriberCount", 0)),
            "n_videos": int(c["statistics"].get("videoCount", 0)),
            "aparicoes_busca": vistos[cid],
            "setor_sugerido": setores[cid].most_common(1)[0][0],
        })
    df = pd.DataFrame(linhas).sort_values(["aparicoes_busca", "inscritos"], ascending=False)
    df.to_csv("data/canais_descobertos.csv", index=False)
    print(f"{len(df)} canais salvos em data/canais_descobertos.csv. Cota usada: {yt.cota_usada}")


if __name__ == "__main__":
    main()
