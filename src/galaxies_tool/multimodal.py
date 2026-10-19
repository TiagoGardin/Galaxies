"""Colunas do modelo vindas da parte de vídeo do bloco C (video.py) e do VLM (bloco D, vlm.py).

Converte os caches por vídeo (data/features/video/<id>.json e data/features/vlm/<id>.json) em colunas, com as
MESMAS funções no treino (modelo.carregar) e no agente (extracao.py). Módulo leve de propósito: não importa
OpenCV, PyTorch nem o SDK do Gemini, para poder conviver com o LightGBM no mesmo processo.
"""

import json
from pathlib import Path

import pandas as pd

PASTA_VIDEO = Path("data/features/video")
PASTA_VLM = Path("data/features/vlm")
MODELO_VLM = "gemini-3.1-flash-lite"  # o mesmo de vlm.MODELO (há um teste que confere)

# Bloco C, parte do vídeo. Largura, altura e "vertical" não entram (quase todo Short é 360x640) e a duração
# já está no bloco A (duracao_s, da API).
COLUNAS_VIDEO = ["cortes_por_segundo", "movimento_medio", "tem_voz", "voz_por_segundo", "energia_rms"]

# Bloco D. `potencial_viral_0a10` nunca entra (é o LLM chutando o resultado; ver docs/FEATURES.md).
VLM_BINARIAS = [
    "tem_rosto",
    "close_up_produto",
    "before_after_ou_transformacao",
    "try_on_ou_demonstracao",
    "swatch",
    "aplicacao_na_pele",
    "texto_na_tela",
    "tem_musica_de_fundo",
]
VLM_NOTAS = [
    "forca_hook_0a10",
    "densidade_texto_0a10",
    "ritmo_cortes_0a10",
    "estetica_geral_0a10",
    "clareza_beneficio_produto_0a10",
]
# categorias fixas pelo esquema da rubrica (vlm.py), não aprendidas dos dados
VLM_CATEGORICAS = {
    "corpo_inteiro_vs_detalhe": ["corpo_inteiro", "detalhe", "nao_se_aplica"],
    "audio_original_ou_musica": ["fala", "musica", "ambos", "ambiente"],
    "tom_narracao": ["ausente", "calmo", "animado", "instrucional"],
    "formato": ["tutorial", "grwm", "haul", "review", "trend", "humor", "outro"],
}
PREFIXO_VLM = "vlm_"
COLUNAS_VLM_BINARIAS = [PREFIXO_VLM + c for c in VLM_BINARIAS]
COLUNAS_VLM_NOTAS = [PREFIXO_VLM + c for c in VLM_NOTAS]
COLUNAS_VLM_CATEGORICAS = [PREFIXO_VLM + c for c in VLM_CATEGORICAS]
COLUNAS_VLM = COLUNAS_VLM_BINARIAS + COLUNAS_VLM_NOTAS + COLUNAS_VLM_CATEGORICAS


def features_video(medidas: dict) -> dict:
    """Colunas do bloco C (vídeo) a partir das medidas de video.medir."""
    return {c: medidas[c] for c in COLUNAS_VIDEO}


def features_vlm(rubrica: dict) -> dict:
    """Colunas do bloco D a partir da rubrica validada (prefixo vlm_ para não colidir com outros blocos)."""
    return {PREFIXO_VLM + c: rubrica[c] for c in VLM_BINARIAS + VLM_NOTAS + list(VLM_CATEGORICAS)}


def tipar(df: pd.DataFrame) -> pd.DataFrame:
    """Binárias como float (NaN = ausente) e categóricas com as categorias fixas do esquema."""
    df = df.copy()
    for c in COLUNAS_VIDEO + COLUNAS_VLM_BINARIAS + COLUNAS_VLM_NOTAS:
        if c in df:
            df[c] = pd.to_numeric(df[c].astype(object), errors="coerce").astype(float)
    for c, cats in VLM_CATEGORICAS.items():
        if PREFIXO_VLM + c in df:
            df[PREFIXO_VLM + c] = pd.Categorical(df[PREFIXO_VLM + c].astype(object), categories=cats)
    return df


def _ler_jsons(pasta: Path) -> list[dict]:
    return [json.loads(a.read_text()) for a in sorted(pasta.glob("*.json"))]


def tabela_video(pasta: Path = PASTA_VIDEO) -> pd.DataFrame:
    linhas = [{"video_id": d["video_id"], **features_video(d)} for d in _ler_jsons(pasta)]
    return tipar(pd.DataFrame(linhas, columns=["video_id", *COLUNAS_VIDEO]))


def tabela_vlm(pasta: Path = PASTA_VLM) -> pd.DataFrame:
    dados = _ler_jsons(pasta)
    assert all(d["modelo"] == MODELO_VLM for d in dados), "rubricas de outro modelo misturadas"
    linhas = [{"video_id": d["video_id"], **features_vlm(d["rubrica"])} for d in dados]
    return tipar(pd.DataFrame(linhas, columns=["video_id", *COLUNAS_VLM]))
