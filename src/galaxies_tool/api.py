"""API REST opcional (TAPI, seção 8: "Função Python + servidor MCP (FastMCP); FastAPI opcional").

Uso (na raiz do repositório): uv run --group api uvicorn galaxies_tool.api:app --port 8001
Documentação interativa: http://127.0.0.1:8001/docs

São as MESMAS funções da tool e do servidor MCP (agente.py, treino_cliente.py), com o mesmo contrato Pydantic
(contrato.py); a API só muda o transporte. Endpoints:
  GET  /saude              servidor no ar e modelo servido
  GET  /modelo             model card (versão, calibração, métricas), igual ao recurso modelo://card do MCP
  POST /predict            um criativo → PrevisaoCriativo
  POST /rank               vários criativos → RankingCriativos
  POST /what-if/horarios   o mesmo criativo em outras datas/horas
  POST /what-if/legendas   o mesmo criativo com outras legendas
  POST /train              base de um cliente (esquema mínimo do TAPI) → RelatorioTreino (treino dinâmico)

Entradas inválidas voltam como 422 (validação do Pydantic); erros de domínio (ex.: canal fora da base sem
`channel_stats`) como 400.
"""

import json
import os
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from galaxies_tool import agente, previsao
from galaxies_tool.contrato import EntradaCriativo, PrevisaoCriativo, RankingCriativos, RelatorioTreino

os.chdir(Path(__file__).resolve().parents[2])  # dados e modelos usam caminhos relativos à raiz do repositório

app = FastAPI(
    title="Galaxies · preditor de curtidas de YouTube Shorts",
    description="Previsão antes da publicação, com intervalo de 80%, fatores (SHAP), comparáveis e explicação. "
    "O número vem do modelo treinado; o LLM só redige a explicação. Mesmo contrato da tool MCP.",
    version="2.0",
)


class PedidoHorarios(BaseModel):
    entrada: EntradaCriativo
    horarios: list[datetime] = Field(min_length=1, description="Sem fuso = horário de Brasília")


class PedidoLegendas(BaseModel):
    entrada: EntradaCriativo
    legendas: list[str] = Field(min_length=1)


class PedidoTreino(BaseModel):
    caminho_base: str = Field(description="CSV ou Parquet no esquema mínimo, no computador onde a API roda")
    nome: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    usar_midia: bool = False
    coletado_em: datetime | None = None


def _executar(fn, *args):
    """Erros de domínio (ValueError, arquivo inexistente) viram 400 com a mensagem; o resto, 500."""
    try:
        return fn(*args)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/saude")
def saude() -> dict:
    caminho = previsao.caminho_padrao()
    return {"status": "ok", "modelo": caminho.stem, "aviso": previsao.aviso_modelo()}


@app.get("/modelo")
def modelo_card() -> dict:
    from galaxies_tool.servidor_mcp import model_card

    return json.loads(model_card())


@app.post("/predict")
def predict(entrada: EntradaCriativo) -> PrevisaoCriativo:
    return _executar(agente.prever_criativo, entrada)


@app.post("/rank")
def rank(entradas: list[EntradaCriativo]) -> RankingCriativos:
    if len(entradas) < 2:
        raise HTTPException(status_code=422, detail="envie pelo menos 2 criativos para ordenar")
    return _executar(agente.ordenar_criativos, entradas)


@app.post("/what-if/horarios")
def what_if_horarios(pedido: PedidoHorarios) -> list[dict]:
    return _executar(agente.what_if_horarios, pedido.entrada, pedido.horarios)


@app.post("/what-if/legendas")
def what_if_legendas(pedido: PedidoLegendas) -> list[dict]:
    return _executar(agente.what_if_legendas, pedido.entrada, pedido.legendas)


@app.post("/train")
def train(pedido: PedidoTreino) -> RelatorioTreino:
    from galaxies_tool.treino_cliente import treinar_modelo_cliente

    return _executar(treinar_modelo_cliente, pedido.caminho_base, pedido.nome, pedido.usar_midia,
                     pedido.coletado_em)
