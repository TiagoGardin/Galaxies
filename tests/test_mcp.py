"""Teste de ponta a ponta do servidor MCP com um cliente real (stdio), como a persona da Galaxies o usaria."""

import asyncio
import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
TEM_MODELO = (RAIZ / "models/lgbm_final.joblib").exists()


async def _sessao(fn):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=sys.executable, args=["-m", "galaxies_tool.servidor_mcp"], cwd="/")
    async with stdio_client(params) as (leitura, escrita), ClientSession(leitura, escrita) as s:
        await s.initialize()
        return await fn(s)


def test_servidor_lista_tools_e_recurso_do_model_card():
    async def fn(s):
        tools = {t.name for t in (await s.list_tools()).tools}
        recursos = {str(r.uri) for r in (await s.list_resources()).resources}
        return tools, recursos, s.instructions if hasattr(s, "instructions") else ""

    tools, recursos, _ = asyncio.run(_sessao(fn))
    assert tools == {"prever_criativo", "ordenar_criativos", "what_if_horarios"}
    assert "modelo://card" in recursos


@pytest.mark.slow
def test_numero_dito_pelo_mcp_e_o_do_modelo():
    """Criativo só em texto (sem rede): a resposta da tool via MCP traz o contrato completo, o intervalo
    ordenado, a versão do modelo e o motivo de baixa confiança por falta de imagem e vídeo."""
    if not TEM_MODELO:
        pytest.skip("sem models/lgbm_final.joblib")
    import pandas as pd

    base = pd.read_parquet(RAIZ / "data/processed/base_v1.parquet", columns=["channel_id", "sector"])
    cid, setor = base.iloc[0].channel_id, base.iloc[0].sector
    entrada = {
        "creative": "Tutorial de contorno em 30 segundos #maquiagem",
        "caption": "Contorno em 30 s",
        "planned_datetime": "2026-10-20T18:00",
        "channel_id": cid,
        "sector": setor,
    }

    async def fn(s):
        r = await s.call_tool("prever_criativo", {"entrada": entrada})
        card = await s.read_resource("modelo://card")
        return r, card

    r, card = asyncio.run(_sessao(fn))
    assert not r.is_error if hasattr(r, "is_error") else not r.isError
    saida = r.structured_content if hasattr(r, "structured_content") else r.structuredContent
    saida = saida.get("result", saida)
    assert saida["interval_low"] <= saida["predicted_metric"] <= saida["interval_high"]
    assert saida["confidence_flag"] and any(
        m["codigo"] == "extracao_incompleta" for m in saida["confidence_reasons"]
    )
    assert saida["modelo_versao"] == json.loads(card.contents[0].text)["versao"]
    assert str(saida["predicted_metric"]) in saida["explanation"].replace(".", "")
