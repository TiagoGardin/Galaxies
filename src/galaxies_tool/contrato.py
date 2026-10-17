"""Contrato da tool (seção 7 do TAPI): entrada e saída validadas com Pydantic.

O esquema é agnóstico à base: o canal pode vir por `channel_id` (um canal da base coletada) ou por
`channel_stats` (histórico enviado pelo próprio cliente), e o criativo por URL do YouTube, caminho de
vídeo ou imagem, ou só texto.
"""

from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, field_validator, model_validator

FUSO = ZoneInfo("America/Sao_Paulo")
Setor = Literal["moda", "beleza", "automotivo"]


class PostHistorico(BaseModel):
    published_at: datetime
    likes: int = Field(ge=0)

    @field_validator("published_at")
    @classmethod
    def _com_fuso(cls, v: datetime) -> datetime:
        return v if v.tzinfo else v.replace(tzinfo=FUSO)


class ChannelStats(BaseModel):
    """Histórico do canal enviado pelo cliente: Shorts anteriores com curtidas (quanto mais, melhor; >= 5)."""

    tipo_canal: Literal["criador", "marca"]
    posts: list[PostHistorico] = Field(min_length=1)
    inscritos: int | None = Field(default=None, ge=0, description="Informativo: não entra no modelo")


class EntradaCriativo(BaseModel):
    creative: str = Field(description="URL do YouTube, caminho de vídeo (.mp4/.mov) ou imagem, ou só texto")
    caption: str = Field(default="", description="Legenda/título planejado")
    description: str = Field(default="", description="Descrição planejada (opcional)")
    tags: list[str] = Field(default_factory=list)
    planned_datetime: datetime = Field(description="Data e hora previstas; sem fuso = horário de Brasília")
    channel_id: str | None = None
    channel_stats: ChannelStats | None = None
    sector: Setor
    id: str | None = Field(default=None, description="Rótulo do criativo para o ranking (opcional)")

    @field_validator("planned_datetime")
    @classmethod
    def _com_fuso(cls, v: datetime) -> datetime:
        return v if v.tzinfo else v.replace(tzinfo=FUSO)

    @model_validator(mode="after")
    def _canal_informado(self):
        if not self.channel_id and not self.channel_stats:
            raise ValueError("informe channel_id ou channel_stats")
        return self


class Driver(BaseModel):
    atributo: str
    descricao: str
    valor: str
    efeito: Literal["aumenta", "reduz"]
    impacto_pct: float = Field(description="Efeito multiplicativo nas curtidas previstas, em %")
    bloco: str


class Comparavel(BaseModel):
    video_id: str
    url: str
    titulo: str
    publicado_em: str
    curtidas: int
    desempenho_vs_canal: float = Field(description="Curtidas / mediana do canal na época")
    similaridade: float


class RelativoCanal(BaseModel):
    mediana_canal: float = Field(description="Mediana de curtidas dos últimos (até) 20 Shorts maduros")
    razao_prevista: float = Field(description="Curtidas previstas / mediana do canal")
    faixa: Literal["abaixo", "na_media", "acima", "viral"]


class PrevisaoCriativo(BaseModel):
    id: str | None = None
    metric: Literal["curtidas"] = "curtidas"
    predicted_metric: int
    interval_low: int
    interval_high: int
    interval_level: float = 0.8
    confidence_flag: bool = Field(description="True = baixa confiança")
    confidence_reasons: list[str]
    relative_to_channel: RelativoCanal
    top_drivers: list[Driver]
    contribuicao_por_bloco: dict[str, float] = Field(description="Soma do SHAP por bloco, em % de efeito")
    explanation: str
    comparables: list[Comparavel]
    atributos_extraidos: dict[str, float | int | str | bool | None]
    modelo_versao: str


class ItemRanking(BaseModel):
    posicao: int
    id: str
    predicted_metric: int
    interval_low: int
    interval_high: int
    razao_prevista: float
    faixa: str
    confidence_flag: bool


class RankingCriativos(BaseModel):
    ranking: list[ItemRanking]
    previsoes: list[PrevisaoCriativo]
    explanation: str
