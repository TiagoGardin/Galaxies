"""Agente orquestrador (LangGraph): criativo + contexto → previsão, intervalo, explicação e ranking.

Uso como função Python:
    from galaxies_tool.agente import prever_criativo, ordenar_criativos, what_if_horarios
    prever_criativo({"creative": "https://www.youtube.com/shorts/<id>", "caption": "...",
                     "planned_datetime": "2026-10-20T18:00", "channel_id": "UC...", "sector": "beleza"})

Grafo de um criativo (cada nó é uma tool da arquitetura do TAPI):
    extrair_atributos → historico_canal → prever → explicar → montar_saida
      Tool 1              Tool 2           Tool 3   Tool 4
Ranking (Tool 5): avaliar cada criativo com o grafo acima → ordenar → explicar_ranking.

O número vem sempre do modelo treinado (previsao.py). O LLM aparece só no subagente que redige a
explicação a partir dos fatos calculados, e é instruído a não criar números; se ele falhar ou não houver
chave, a explicação é montada por um texto padrão com os mesmos fatos.
"""

import json
import os
from datetime import datetime
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from galaxies_tool import canal, extracao, previsao
from galaxies_tool.contrato import (
    FUSO,
    EntradaCriativo,
    ItemRanking,
    PrevisaoCriativo,
    RankingCriativos,
)

# LLM do subagente explicador, em ordem de preferência (o 2º cobre sobrecarga do 1º). São modelos
# diferentes do
# VLM, então não gastam a cota diária dele (a cota do free tier é por modelo).
MODELOS_LLM = os.environ.get("AGENTE_MODELO", "gemini-3.5-flash,gemini-flash-latest").split(",")
SETORES_TREINO = ("moda", "beleza")
NOMES_FAIXA = {
    "abaixo": "abaixo da média do canal",
    "na_media": "na média do canal",
    "acima": "acima da média do canal",
    "viral": "muito acima da média do canal (faixa viral)",
}

# ---------------------------------------------------------------- estado e nós do grafo de um criativo


class Estado(TypedDict, total=False):
    entrada: EntradaCriativo
    atributos: dict
    contexto: dict
    comparaveis: list[dict]
    previsao: dict
    explicacao: str
    saida: PrevisaoCriativo


def no_extrair_atributos(estado: Estado) -> Estado:
    """Tool 1: atributos do criativo (texto, imagem), com as mesmas funções do treino."""
    e = estado["entrada"]
    return {"atributos": extracao.extrair_isolado(e.creative, e.caption, e.description, e.tags)}


def no_historico_canal(estado: Estado) -> Estado:
    """Tool 2: bloco A (histórico do canal e tempo) e criativos comparáveis."""
    e, atr = estado["entrada"], estado["atributos"]
    planned = pd.Timestamp(e.planned_datetime).tz_convert("UTC")
    avisos = list(atr.get("_avisos", []))
    if e.channel_id:
        if not canal.canal_conhecido(e.channel_id):
            raise ValueError(
                f"canal {e.channel_id} fora da base coletada: envie o histórico em channel_stats"
            )
        hist = canal.features_do_canal(planned, channel_id=e.channel_id)
        tipo = canal.tipo_do_canal(e.channel_id)
        categoria = atr.get("categoria_id") or canal.categoria_mais_comum(e.channel_id)
    else:
        posts = [(pd.Timestamp(p.published_at), p.likes) for p in e.channel_stats.posts]
        hist = canal.features_do_canal(planned, posts=posts)
        tipo, categoria = e.channel_stats.tipo_canal, atr.get("categoria_id")
    if hist["n_historico_total"] == 0:
        raise ValueError(
            "o canal precisa de pelo menos 1 Short publicado 30 dias ou mais antes da data planejada"
        )
    if e.sector not in SETORES_TREINO:
        avisos.append(f"setor '{e.sector}' fora da base de treino (só moda e beleza)")
    contexto = {
        **hist,
        **canal.features_de_tempo(planned),
        "setor": e.sector,
        "tipo_canal": tipo,
        "categoria_id": categoria,
        "_avisos": avisos,
    }
    return {"contexto": contexto, "comparaveis": canal.comparaveis(atr, planned, e.sector, e.channel_id)}


def linha_do_modelo(atributos: dict, contexto: dict) -> dict:
    return {
        **{k: v for k, v in atributos.items() if not k.startswith("_")},
        **{k: v for k, v in contexto.items() if not k.startswith("_")},
    }


def no_prever(estado: Estado) -> Estado:
    """Tool 3 (+ SHAP da Tool 4): previsão, intervalo de 80%, faixa e atribuição por atributo e bloco."""
    p = previsao.prever_linha(linha_do_modelo(estado["atributos"], estado["contexto"]))
    p["motivos_baixa_confianca"] += estado["contexto"]["_avisos"]
    return {"previsao": p}


def fatos_para_explicar(estado: Estado) -> dict:
    p, e = estado["previsao"], estado["entrada"]
    return {
        "legenda": estado["atributos"].get("_titulo", ""),
        "curtidas_previstas": p["curtidas_previstas"],
        "intervalo_80": [p["curtidas_baixo"], p["curtidas_alto"]],
        "mediana_do_canal": round(p["mediana_canal"]),
        "razao_vs_canal": p["razao_prevista"],
        "faixa": NOMES_FAIXA[p["faixa"]],
        "fatores": [
            {k: d[k] for k in ("descricao", "valor", "efeito", "impacto_pct")}
            for d in p["drivers"][: previsao.N_DRIVERS]
        ],
        "efeito_por_bloco_pct": p["contribuicao_por_bloco"],
        "baixa_confianca": p["motivos_baixa_confianca"],
        "comparaveis": [
            {k: c[k] for k in ("titulo", "curtidas", "desempenho_vs_canal")}
            for c in estado["comparaveis"][:3]
        ],
        "setor": e.sector,
    }


INSTRUCOES_EXPLICADOR = """Você é o subagente que explica a previsão de um modelo de machine learning
para um time de marketing de moda e beleza. Você recebe os fatos já calculados (JSON) e escreve a
justificativa.
Regras:
- Use SOMENTE os números e fatores do JSON. Nunca crie, recalcule ou arredonde para outro valor.
- Comece pelo veredito: curtidas previstas, intervalo e a posição em relação à média do canal.
- Explique os 2 a 4 fatores de maior impacto, dizendo se aumentam ou reduzem o resultado, em linguagem
  de marketing ("o gancho visual...", "postar às 18h..."). Fatores de canal (patamar, tendência) são
  contexto: diga que vêm do histórico, não do criativo.
- Se houver motivos de baixa confiança, diga isso claramente em uma frase.
- Se houver comparáveis, cite um como evidência.
- 4 a 6 frases, português do Brasil, sem jargão (não use "SHAP", "log", "embedding", "PCA")."""


def _cliente_llm():
    load_dotenv(".env")
    chave = os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not chave:
        return None
    from google import genai
    from google.genai import types

    return genai.Client(api_key=chave, http_options=types.HttpOptions(timeout=60_000))


def explicacao_padrao(f: dict) -> str:
    """Explicação sem LLM, com os mesmos fatos (usada se o LLM falhar ou não houver chave)."""

    def fmt(n):
        return f"{n:,}".replace(",", ".")

    texto = (
        f"Previsão: {fmt(f['curtidas_previstas'])} curtidas (intervalo de 80%: {fmt(f['intervalo_80'][0])} "
        f"a {fmt(f['intervalo_80'][1])}), {f['razao_vs_canal']:.2f}× a mediana recente do canal "
        f"({fmt(f['mediana_do_canal'])}), ou seja, {f['faixa']}. Principais fatores: "
    )
    texto += (
        "; ".join(
            f"{d['descricao'].lower()} = {d['valor']} ({d['efeito']} ~{abs(d['impacto_pct']):.0f}%)"
            for d in f["fatores"][:4]
        )
        + "."
    )
    if f["baixa_confianca"]:
        texto += " Baixa confiança: " + "; ".join(f["baixa_confianca"]) + "."
    return texto


def redigir(instrucoes: str, fatos: dict) -> str | None:
    """Chama o LLM; devolve None em qualquer falha (a explicação padrão assume)."""
    cliente = _cliente_llm()
    if cliente is None:
        return None
    from google.genai import types

    for modelo in MODELOS_LLM:
        try:
            r = cliente.models.generate_content(
                model=modelo.strip(),
                contents=json.dumps(fatos, ensure_ascii=False, default=str),
                config=types.GenerateContentConfig(system_instruction=instrucoes, temperature=0.2),
            )
            if texto := (r.text or "").strip():
                return texto
        except Exception:  # noqa: BLE001, S112 — cota, rede ou modelo indisponível: tenta o próximo
            continue
    return None


def no_explicar(estado: Estado) -> Estado:
    """Tool 4 (subagente): transforma fatores e números em justificativa em linguagem natural."""
    f = fatos_para_explicar(estado)
    return {"explicacao": redigir(INSTRUCOES_EXPLICADOR, f) or explicacao_padrao(f)}


def _valor_simples(v):
    if isinstance(v, (np.generic,)):
        v = v.item()
    if isinstance(v, float) and np.isnan(v):
        return None
    return v if isinstance(v, (int, float, str, bool)) or v is None else str(v)


def no_montar_saida(estado: Estado) -> Estado:
    p, e = estado["previsao"], estado["entrada"]
    inspecao = {
        k: _valor_simples(v)
        for k, v in linha_do_modelo(estado["atributos"], estado["contexto"]).items()
        if not k.startswith(("embtxt_", "embimg_"))
    }
    inspecao["tipo_criativo"] = estado["atributos"]["_tipo"]
    saida = PrevisaoCriativo(
        id=e.id,
        predicted_metric=p["curtidas_previstas"],
        interval_low=p["curtidas_baixo"],
        interval_high=p["curtidas_alto"],
        confidence_flag=bool(p["motivos_baixa_confianca"]),
        confidence_reasons=p["motivos_baixa_confianca"],
        relative_to_channel={
            "mediana_canal": round(p["mediana_canal"], 1),
            "razao_prevista": p["razao_prevista"],
            "faixa": p["faixa"],
        },
        top_drivers=p["drivers"][: previsao.N_DRIVERS],
        contribuicao_por_bloco=p["contribuicao_por_bloco"],
        explanation=estado["explicacao"],
        comparables=estado["comparaveis"],
        atributos_extraidos=inspecao,
        modelo_versao=p["modelo_versao"],
    )
    return {"saida": saida}


def construir_grafo():
    g = StateGraph(Estado)
    for nome, fn in (
        ("extrair_atributos", no_extrair_atributos),
        ("historico_canal", no_historico_canal),
        ("prever", no_prever),
        ("explicar", no_explicar),
        ("montar_saida", no_montar_saida),
    ):
        g.add_node(nome, fn)
    g.add_edge(START, "extrair_atributos")
    g.add_edge("extrair_atributos", "historico_canal")
    g.add_edge("historico_canal", "prever")
    g.add_edge("prever", "explicar")
    g.add_edge("explicar", "montar_saida")
    g.add_edge("montar_saida", END)
    return g.compile()


GRAFO = construir_grafo()


# ---------------------------------------------------------------- ranking (Tool 5)


class EstadoRanking(TypedDict, total=False):
    entradas: list[EntradaCriativo]
    previsoes: list[PrevisaoCriativo]
    ranking: list[ItemRanking]
    explicacao: str


def no_avaliar_todos(estado: EstadoRanking) -> EstadoRanking:
    return {"previsoes": [GRAFO.invoke({"entrada": e})["saida"] for e in estado["entradas"]]}


def no_ordenar(estado: EstadoRanking) -> EstadoRanking:
    """Ordena pelas curtidas previstas. Para criativos do mesmo canal e mesma época, é o mesmo que ordenar
    pelo desempenho relativo, que é o que o modelo prevê."""
    ordem = sorted(estado["previsoes"], key=lambda p: p.predicted_metric, reverse=True)
    return {
        "ranking": [
            ItemRanking(
                posicao=i,
                id=p.id,
                predicted_metric=p.predicted_metric,
                interval_low=p.interval_low,
                interval_high=p.interval_high,
                razao_prevista=p.relative_to_channel.razao_prevista,
                faixa=p.relative_to_channel.faixa,
                confidence_flag=p.confidence_flag,
            )
            for i, p in enumerate(ordem, 1)
        ]
    }


INSTRUCOES_RANKING = """Você explica a um time de marketing o ranking de criativos feito por um modelo
de machine learning. Use SOMENTE os números e fatores do JSON. Em 3 a 5 frases, em português do Brasil:
diga qual criativo deve render mais e por quanto em relação aos outros; aponte os fatores que mais
separam o primeiro dos demais; se os intervalos se sobrepõem muito, diga que a diferença é incerta.
Sem jargão técnico."""


def no_explicar_ranking(estado: EstadoRanking) -> EstadoRanking:
    por_id = {p.id: p for p in estado["previsoes"]}
    fatos = [
        {
            "posicao": r.posicao,
            "id": r.id,
            "curtidas_previstas": r.predicted_metric,
            "intervalo_80": [r.interval_low, r.interval_high],
            "razao_vs_canal": r.razao_prevista,
            "baixa_confianca": r.confidence_flag,
            "fatores": [
                {"descricao": d.descricao, "valor": d.valor, "efeito": d.efeito, "impacto_pct": d.impacto_pct}
                for d in por_id[r.id].top_drivers[:4]
            ],
        }
        for r in estado["ranking"]
    ]
    texto = redigir(INSTRUCOES_RANKING, {"ranking": fatos})
    if not texto:
        r = estado["ranking"]
        texto = " > ".join(f"{x.id} ({x.predicted_metric:,} curtidas)".replace(",", ".") for x in r) + "."
        if len(r) > 1 and r[1].interval_high >= r[0].interval_low:
            texto += " Os intervalos se sobrepõem: a diferença entre os primeiros é incerta."
    return {"explicacao": texto}


def construir_grafo_ranking():
    g = StateGraph(EstadoRanking)
    g.add_node("avaliar_todos", no_avaliar_todos)
    g.add_node("ordenar", no_ordenar)
    g.add_node("explicar_ranking", no_explicar_ranking)
    g.add_edge(START, "avaliar_todos")
    g.add_edge("avaliar_todos", "ordenar")
    g.add_edge("ordenar", "explicar_ranking")
    g.add_edge("explicar_ranking", END)
    return g.compile()


GRAFO_RANKING = construir_grafo_ranking()


# ---------------------------------------------------------------- funções públicas


def prever_criativo(entrada: EntradaCriativo | dict[str, Any]) -> PrevisaoCriativo:
    """Previsão de curtidas de UM criativo antes da publicação, com intervalo, fatores e justificativa."""
    entrada = EntradaCriativo.model_validate(entrada)
    return GRAFO.invoke({"entrada": entrada})["saida"]


def ordenar_criativos(entradas: list[EntradaCriativo | dict[str, Any]]) -> RankingCriativos:
    """Ranking de vários criativos (ex.: 5 vídeos de uma campanha) pelas curtidas previstas."""
    validadas = [EntradaCriativo.model_validate(e) for e in entradas]
    validadas = [e.model_copy(update={"id": e.id or f"criativo_{i}"}) for i, e in enumerate(validadas, 1)]
    if len({e.id for e in validadas}) != len(validadas):
        raise ValueError("ids dos criativos repetidos")
    r = GRAFO_RANKING.invoke({"entradas": validadas})
    return RankingCriativos(ranking=r["ranking"], previsoes=r["previsoes"], explanation=r["explicacao"])


def what_if_horarios(entrada: EntradaCriativo | dict[str, Any], horarios: list[datetime | str]) -> list[dict]:
    """O mesmo criativo em outras datas/horas: re-pontua só o contexto (atributos extraídos 1 vez)."""
    entrada = EntradaCriativo.model_validate(entrada)
    estado: Estado = {"entrada": entrada, **no_extrair_atributos({"entrada": entrada})}
    saida = []
    for h in [entrada.planned_datetime, *horarios]:
        alt = entrada.model_copy(
            update={
                "planned_datetime": EntradaCriativo.model_validate(
                    {**entrada.model_dump(), "planned_datetime": h}
                ).planned_datetime
            }
        )
        est = {**estado, "entrada": alt}
        est.update(no_historico_canal(est))
        p = no_prever(est)["previsao"]
        saida.append(
            {
                "planned_datetime": alt.planned_datetime.astimezone(FUSO).isoformat(timespec="minutes"),
                "predicted_metric": p["curtidas_previstas"],
                "interval_low": p["curtidas_baixo"],
                "interval_high": p["curtidas_alto"],
                "razao_prevista": p["razao_prevista"],
            }
        )
    return saida
