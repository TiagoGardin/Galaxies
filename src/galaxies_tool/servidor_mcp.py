"""Servidor MCP (MCPServer do SDK oficial, o antigo FastMCP): expõe o agente como tool para a persona
sintética da Galaxies.

Uso: uv run python -m galaxies_tool.servidor_mcp            (transporte stdio, o padrão dos clientes MCP)
     uv run python -m galaxies_tool.servidor_mcp --http     (HTTP streamable em http://127.0.0.1:8000/mcp)

Configuração num cliente MCP (ex.: Claude Desktop, Claude Code ou o agente da Galaxies):
    {"command": "uv", "args": ["run", "--directory", "<repositório>", "python", "-m",
                               "galaxies_tool.servidor_mcp"]}

Tools: prever_criativo, ordenar_criativos e what_if_horarios (mesmo contrato de galaxies_tool.contrato).
"""

import os
import sys
from datetime import datetime
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from galaxies_tool import agente
from galaxies_tool.contrato import EntradaCriativo, PrevisaoCriativo, RankingCriativos

os.chdir(Path(__file__).resolve().parents[2])  # dados e modelos usam caminhos relativos à raiz do repositório

mcp = MCPServer(
    "galaxies-preditor",
    instructions=(
        "Prevê, antes da publicação, as curtidas de um YouTube Short de moda ou beleza em relação ao "
        "padrão do "
        "canal, com intervalo de 80%, fatores que pesaram e criativos comparáveis. O número vem de um modelo "
        "LightGBM treinado em ~22 mil Shorts brasileiros; use-o para comparar criativos, não como promessa. "
        "Se confidence_flag for true, comunique a incerteza."
    ),
)


@mcp.tool()
def prever_criativo(entrada: EntradaCriativo) -> PrevisaoCriativo:
    """Prevê as curtidas de um criativo (URL do YouTube, vídeo/imagem local ou texto) antes de publicar.

    Informe o canal por `channel_id` (canal da base) ou `channel_stats` (histórico de Shorts com curtidas).
    Retorna previsão, intervalo de 80%, sinal de baixa confiança, desempenho relativo ao canal, principais
    fatores (SHAP), justificativa em linguagem natural e criativos comparáveis do histórico.
    """
    return agente.prever_criativo(entrada)


@mcp.tool()
def ordenar_criativos(entradas: list[EntradaCriativo]) -> RankingCriativos:
    """Ordena vários criativos (ex.: os vídeos de uma campanha) pelas curtidas previstas, com justificativa.

    Dê um `id` a cada criativo para identificá-lo no ranking.
    """
    return agente.ordenar_criativos(entradas)


@mcp.tool()
def what_if_horarios(entrada: EntradaCriativo, horarios: list[datetime]) -> list[dict]:
    """Re-pontua o mesmo criativo em outras datas/horas de publicação (sem fuso = horário de Brasília).

    O primeiro item é o horário original da entrada.
    """
    return agente.what_if_horarios(entrada, horarios)


if __name__ == "__main__":
    mcp.run(transport="streamable-http" if "--http" in sys.argv else "stdio")
