"""Regras da base, espelhando docs/DATA_CARD.md. Mudou aqui, mude lá."""

from datetime import UTC, datetime, timedelta

INICIO_JANELA_ALVO = "2025-04-01T00:00:00Z"
IDADE_MINIMA_DIAS = 30
MUDANCA_LIMITE_SHORTS = "2024-10-15T00:00:00Z"  # Shorts passam de 60 s para 180 s
MIN_SHORTS_NA_JANELA = 30
MIN_SHORTS_NA_JANELA_MARCA = 10  # marcas postam menos Shorts; mais marcas = base mais próxima do cliente
TETO_ALVOS_POR_CANAL = 250


def eh_short(duracao_s: int, publicado_em: str) -> bool:
    limite = 180 if publicado_em >= MUDANCA_LIMITE_SHORTS else 60
    return 0 < duracao_s <= limite


def fim_janela_alvo(coleta: datetime | None = None) -> str:
    coleta = coleta or datetime.now(UTC)
    return (coleta - timedelta(days=IDADE_MINIMA_DIAS)).strftime("%Y-%m-%dT%H:%M:%SZ")
