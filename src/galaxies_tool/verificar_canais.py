"""Verifica canais candidatos contra os critérios do data card.

Uso: uv run python -m galaxies_tool.verificar_canais
Entradas: data/canais_semente.csv, data/canais_descobertos.csv
Saída: data/canais_candidatos.csv (para revisão manual -> data/canais.csv)

Lê só as páginas mais recentes de uploads (até cobrir a janela-alvo ou MAX_PAGINAS),
então n_shorts_janela é um limite inferior quando janela_coberta=False.
"""

import json
import re
import unicodedata
from pathlib import Path
from statistics import median

import pandas as pd

from galaxies_tool.regras import (
    INICIO_JANELA_ALVO,
    MIN_SHORTS_NA_JANELA,
    MIN_SHORTS_NA_JANELA_MARCA,
    eh_short,
    fim_janela_alvo,
)
from galaxies_tool.youtube_api import YouTube, duracao_segundos

MAX_PAGINAS = 4  # 200 uploads por canal na verificação
MIN_INSCRITOS_DESCOBERTOS = 5_000
MIN_VIDEOS_DESCOBERTOS = 50
CACHE = Path("data/cache/verificacao")
MIN_ADERENCIA_TEMA = 0.3

# Só para triagem de canais (aderência ao tema), nunca como feature do modelo.
TERMOS_TEMA = {
    "beleza": r"maquiage\w*|\bmake\b|makeup|skin ?care|\bpele\b|batom|\bbase\b|rimel|mascara de cilios|cilios|"
              r"sobrancelha|blush|corretivo|gloss|po compacto|iluminador|cabelo|\bunhas?\b|esmalte|perfume|"
              r"beleza|serum|hidratante|protetor solar|cachos|cacheado|autocuidado|grwm|arrume-se",
    "moda": r"\blooks?\b|roupas?|\bmoda\b|outfit|provador|provando|vestido|calca|\btenis\b|sapatos?|"
            r"bolsas?|"
            r"\bestilo\b|comprinhas|\bhaul\b|shein|jeans|blusa|\bsaia\b|combinar|tendencia|acessorios|ootd|"
            r"lookbook|brecho|alfaiataria",
}
_REGEX_TEMA = {s: re.compile(p) for s, p in TERMOS_TEMA.items()}


def _normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sem_acento.lower()


def aderencia_tema(videos: list[dict]) -> dict[str, float]:
    """Fração dos vídeos cujo título+descrição+tags cita termos de cada setor."""
    if not videos:
        return {s: 0.0 for s in TERMOS_TEMA}
    textos = [_normalizar(" ".join([v["snippet"]["title"], v["snippet"].get("description", ""),
                                    " ".join(v["snippet"].get("tags", []))])) for v in videos]
    return {s: sum(bool(rx.search(t)) for t in textos) / len(textos) for s, rx in _REGEX_TEMA.items()}


def carregar_candidatos(yt: YouTube) -> list[dict]:
    sementes = pd.read_csv("data/canais_semente.csv", dtype=str).fillna("")
    linhas = []
    for s in sementes.itertuples():
        c = yt.canal_por_handle(s.handle)
        if c is None:
            print(f"handle não encontrado: {s.handle}")
            continue
        linhas.append({"canal": c, "setor": s.setor, "tipo": s.tipo, "origem": "semente",
                       "aparicoes_busca": 0})

    desc = pd.read_csv("data/canais_descobertos.csv", dtype={"pais": str}).fillna({"pais": ""})
    desc = desc[desc.pais.isin(["BR", ""]) & (desc.inscritos >= MIN_INSCRITOS_DESCOBERTOS)
                & (desc.n_videos >= MIN_VIDEOS_DESCOBERTOS)]
    ja = {x["canal"]["id"] for x in linhas}
    desc = desc[~desc.channel_id.isin(ja)]
    for c in yt.canais_por_ids(desc.channel_id.tolist()):
        d = desc.loc[desc.channel_id == c["id"]].iloc[0]
        linhas.append({"canal": c, "setor": d.setor_sugerido, "tipo": "", "origem": "descoberta",
                       "aparicoes_busca": int(d.aparicoes_busca)})
    return linhas


def videos_recentes(yt: YouTube, canal: dict) -> list[dict]:
    arq = CACHE / f"{canal['id']}.json"
    if arq.exists():
        return json.loads(arq.read_text())
    playlist = canal["contentDetails"]["relatedPlaylists"]["uploads"]
    ids = [i["contentDetails"]["videoId"]
           for i in yt.uploads(playlist, max_paginas=MAX_PAGINAS, parar_antes=INICIO_JANELA_ALVO)]
    vids = yt.videos(ids)
    arq.write_text(json.dumps(vids, ensure_ascii=False))
    return vids


def resumir(linha: dict, vids: list[dict], fim: str) -> dict:
    c = linha["canal"]
    shorts_janela, idiomas = [], []
    for v in vids:
        pub = v["snippet"]["publishedAt"]
        if not (INICIO_JANELA_ALVO <= pub <= fim):
            continue
        if not eh_short(duracao_segundos(v["contentDetails"].get("duration", "")), pub):
            continue
        shorts_janela.append(v)
        idiomas.append(v["snippet"].get("defaultAudioLanguage") or v["snippet"].get("defaultLanguage") or "")
    com_likes = [int(v["statistics"]["likeCount"]) for v in shorts_janela if "likeCount" in v["statistics"]]
    mais_antigo = min((v["snippet"]["publishedAt"] for v in vids), default="")
    pais = c["snippet"].get("country", "")

    motivos = []
    if pais not in ("BR", ""):
        motivos.append(f"país {pais}")
    minimo = MIN_SHORTS_NA_JANELA_MARCA if linha["tipo"] == "marca" else MIN_SHORTS_NA_JANELA
    if len(com_likes) < minimo:
        motivos.append(f"só {len(com_likes)} Shorts com curtidas na janela")
    idioma_top = pd.Series(idiomas).replace("", pd.NA).dropna().mode()
    tema = aderencia_tema(shorts_janela)
    return {
        "channel_id": c["id"],
        "handle": c["snippet"].get("customUrl", ""),
        "titulo": c["snippet"]["title"],
        "setor": linha["setor"],
        "tipo": linha["tipo"],
        "origem": linha["origem"],
        "aparicoes_busca": linha["aparicoes_busca"],
        "pais": pais,
        "idioma_declarado": idioma_top.iloc[0] if len(idioma_top) else "",
        "inscritos": int(c["statistics"].get("subscriberCount", 0)),
        "n_videos": int(c["statistics"].get("videoCount", 0)),
        "uploads_lidos": len(vids),
        "janela_coberta": mais_antigo != "" and mais_antigo < INICIO_JANELA_ALVO,
        "n_shorts_janela": len(shorts_janela),
        "n_shorts_janela_com_likes": len(com_likes),
        "mediana_likes_janela": median(com_likes) if com_likes else None,
        "ultimo_short": max((v["snippet"]["publishedAt"] for v in shorts_janela), default=""),
        "aderencia_beleza": round(tema["beleza"], 2),
        "aderencia_moda": round(tema["moda"], 2),
        "setor_conteudo": max(tema, key=tema.get) if max(tema.values()) >= MIN_ADERENCIA_TEMA else "",
        # sementes foram escolhidas por serem de moda/beleza; a triagem por texto vale só para a descoberta
        "tema_provavel": linha["origem"] == "semente" or max(tema.values()) >= MIN_ADERENCIA_TEMA,
        "passa_criterios": not motivos,
        "motivo": "; ".join(motivos),
        "incluir": "",
    }


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    yt = YouTube()
    fim = fim_janela_alvo()
    linhas = carregar_candidatos(yt)
    print(f"{len(linhas)} candidatos a verificar (cota até aqui: {yt.cota_usada})")
    saida = []
    for i, linha in enumerate(linhas, 1):
        saida.append(resumir(linha, videos_recentes(yt, linha["canal"]), fim))
        if i % 25 == 0:
            print(f"{i}/{len(linhas)} verificados (cota {yt.cota_usada})")
    df = pd.DataFrame(saida).sort_values(["passa_criterios", "tema_provavel", "setor", "inscritos"],
                                         ascending=[False, False, True, False])
    df.to_csv("data/canais_candidatos.csv", index=False)
    print(f"Salvo data/canais_candidatos.csv: {df.passa_criterios.sum()} de {len(df)} passam. "
          f"Cota usada: {yt.cota_usada}")


if __name__ == "__main__":
    main()
