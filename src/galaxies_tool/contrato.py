"""Contrato da tool (seção 7 do TAPI): entrada e saída validadas com Pydantic.

O esquema é agnóstico à base: o canal pode vir por `channel_id` (um canal da base coletada) ou por
`channel_stats` (histórico enviado pelo próprio cliente), e o criativo por URL do YouTube, caminho de
vídeo ou imagem, ou só texto.
"""

from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FUSO = ZoneInfo("America/Sao_Paulo")
Setor = Literal["moda", "beleza", "automotivo"]


class PostHistorico(BaseModel):
    model_config = ConfigDict(extra="forbid")

    published_at: datetime
    likes: int = Field(ge=0)

    @field_validator("published_at")
    @classmethod
    def _com_fuso(cls, v: datetime) -> datetime:
        return v if v.tzinfo else v.replace(tzinfo=FUSO)


class ChannelStats(BaseModel):
    """Histórico do canal enviado pelo cliente: Shorts anteriores com curtidas (quanto mais, melhor; >= 5)."""

    model_config = ConfigDict(extra="forbid")

    tipo_canal: Literal["criador", "marca"]
    posts: list[PostHistorico] = Field(min_length=1)
    inscritos: int | None = Field(default=None, ge=0, description="Informativo: não entra no modelo")


class EntradaCriativo(BaseModel):
    model_config = ConfigDict(extra="forbid")  # campo desconhecido é erro, não é ignorado em silêncio

    creative: str = Field(description="URL do YouTube, caminho de vídeo (.mp4/.mov) ou imagem, ou só texto")
    caption: str = Field(default="", description="Legenda/título planejado")
    description: str = Field(default="", description="Descrição planejada (opcional)")
    tags: list[str] = Field(default_factory=list)
    planned_datetime: datetime = Field(description="Data e hora previstas; sem fuso = horário de Brasília")
    channel_id: str | None = None
    channel_stats: ChannelStats | None = None
    sector: Setor
    id: str | None = Field(default=None, description="Rótulo do criativo para o ranking (opcional)")
    modelo: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$",
        description="Modelo especialista treinado com a base de um cliente (treinar_modelo_cliente); "
        "vazio = modelo principal (YouTube Shorts de moda e beleza)",
    )
    analisar_video: bool = Field(
        default=True,
        description="Vídeo (URL ou arquivo): baixa/mede o vídeo e chama o VLM (~1 min no 1º uso do vídeo). "
        "False = só texto e imagem de capa, mais rápido e com menos informação (a confiança cai).",
    )

    @field_validator("planned_datetime")
    @classmethod
    def _com_fuso(cls, v: datetime) -> datetime:
        return v if v.tzinfo else v.replace(tzinfo=FUSO)

    @model_validator(mode="after")
    def _canal_informado(self):
        if not self.channel_id and not self.channel_stats:
            raise ValueError("informe channel_id ou channel_stats")
        return self


CodigoMotivo = Literal[
    "canal_pouco_historico",
    "canal_instavel",
    "intervalo_largo",
    "fora_da_distribuicao",
    "setor_fora_do_treino",
    "extracao_incompleta",
    "data_fora_do_periodo",
]


class MotivoConfianca(BaseModel):
    codigo: CodigoMotivo
    texto: str


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
    confidence_reasons: list[MotivoConfianca] = Field(
        description="Por que a confiança é baixa (vazio = normal)"
    )
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
    prob_superar_seguinte: float | None = Field(
        default=None,
        description="P(este criativo render mais que o próximo do ranking), supondo erros independentes e "
        "normais em log, com o desvio tirado do intervalo de 80%. Criativos do mesmo canal costumam errar "
        "juntos, então a suposição é conservadora (puxa a probabilidade para 50%).",
    )


class RankingCriativos(BaseModel):
    ranking: list[ItemRanking]
    previsoes: list[PrevisaoCriativo]
    explanation: str


class RelatorioTreino(BaseModel):
    """Saída do treino dinâmico (treino_cliente.py): o que entrou, como foi dividido e quanto o modelo do
    cliente erra no teste do próprio cliente, contra a mediana do canal."""

    nome: str
    modelo: str = Field(description="Caminho do artefato; use `modelo: <nome>` em prever_criativo")
    criado_em: str
    n_posts: int
    n_canais: int
    funil: dict[str, int] = Field(description="Posts → maduros (>= 30 dias) → com 5 de histórico no canal")
    periodo: list[str]
    cortes: dict[str, str] = Field(description="Datas dos cortes temporais treino | validação | teste")
    n_por_split: dict[str, int]
    features: list[str]
    metricas_teste: dict[str, dict[str, float | None]] = Field(
        description="Modelo e baselines no teste do cliente (avaliado uma vez)"
    )
    ganho_vs_mediana_do_canal: dict[str, str] = Field(
        description="Diferença modelo − mediana do canal, com IC 95% por bootstrap de canais"
    )
    principais_fatores: list[str]
    cobertura_crua_na_validacao: float
    avisos: list[str]
