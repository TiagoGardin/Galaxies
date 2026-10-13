"""Regras da base, espelhando docs/DATA_CARD.md. Mudou aqui, mude lá."""

import re
from datetime import UTC, datetime, timedelta

INICIO_JANELA_ALVO = "2025-04-01T00:00:00Z"
IDADE_MINIMA_DIAS = 30
MUDANCA_LIMITE_SHORTS = "2024-10-15T00:00:00Z"  # Shorts passam de 60 s para 180 s
INICIO_SHORTS = "2021-07-01T00:00:00Z"  # lançamento global; vídeo curto anterior não era Short
MIN_SHORTS_NA_JANELA = 30
MIN_SHORTS_NA_JANELA_MARCA = 10  # marcas postam menos Shorts; mais marcas = base mais próxima do cliente
TETO_ALVOS_POR_CANAL = 250

# Histórico: o primeiro vídeo-alvo precisa de até 20 Shorts publicados >= 30 dias antes dele.
JANELA_HISTORICO = 20
CORTE_HISTORICO = (datetime.fromisoformat(INICIO_JANELA_ALVO) - timedelta(days=IDADE_MINIMA_DIAS)).strftime(
    "%Y-%m-%dT%H:%M:%SZ")

# Texto editado depois da publicação para comentar o desempenho do PRÓPRIO vídeo.
# Não usa "viralizou" sozinho: "a base que viralizou" fala do produto, não do vídeo.
TEXTO_POS_PUBLICACAO = re.compile(
    r"(esse|este|meu|nosso|o) v[ií]deo (viralizou|bombou|estourou)"
    r"|obrigad[ao]s? (pel[oa]s?|por) (\d|mais de|quase|um|1)[^\n]{0,20}(mil|milh|views|visualiza)"
    r"|\d+([.,]\d+)? ?(mil|milh[õo]es|milh[ãa]o|k|m)( de)? (views|visualiza)",
    re.IGNORECASE,
)


def eh_short(duracao_s: int, publicado_em: str) -> bool:
    if publicado_em < INICIO_SHORTS:
        return False
    limite = 180 if publicado_em >= MUDANCA_LIMITE_SHORTS else 60
    return 0 < duracao_s <= limite


def fim_janela_alvo(coleta: datetime | None = None) -> str:
    coleta = coleta or datetime.now(UTC)
    return (coleta - timedelta(days=IDADE_MINIMA_DIAS)).strftime("%Y-%m-%dT%H:%M:%SZ")


def texto_pos_publicacao(titulo: str, descricao: str) -> bool:
    return bool(TEXTO_POS_PUBLICACAO.search(f"{titulo}\n{descricao}"))
