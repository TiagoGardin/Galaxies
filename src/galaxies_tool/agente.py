"""Agente orquestrador (LangGraph): criativo + contexto → previsão, intervalo, explicação e ranking.

Uso como função Python:
    from galaxies_tool.agente import prever_criativo, ordenar_criativos, what_if_horarios, what_if_legendas
    prever_criativo({"creative": "https://www.youtube.com/shorts/<id>", "caption": "...",
                     "planned_datetime": "2026-10-20T18:00", "channel_id": "UC...", "sector": "beleza"})

Grafo de um criativo (cada nó é uma tool da arquitetura do TAPI):
    extrair_atributos → historico_canal → prever → explicar → montar_saida
      Tool 1              Tool 2           Tool 3   Tool 4
Ranking (Tool 5): avaliar cada criativo com o grafo acima → ordenar → explicar_ranking.

O número vem sempre do modelo treinado (previsao.py). O LLM aparece só no subagente que redige a
explicação a partir de um JSON fechado de fatos (aula 5.7 da trilha): a saída é estruturada, um validador
determinístico confere que todo número citado está no JSON e que todo fator citado está entre os fatores do
modelo, e o LLM tem uma chance de corrigir. Se falhar de novo, ou se não houver chave, a explicação é um
texto padrão montado só com os fatos (sempre fiel).
"""

import json
import os
import re
import time
from datetime import datetime
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from galaxies_tool import canal, extracao, previsao
from galaxies_tool.contrato import (
    FUSO,
    EntradaCriativo,
    ItemRanking,
    PrevisaoCriativo,
    RankingCriativos,
)

# LLM do subagente explicador, em ordem de preferência (o 2º cobre sobrecarga do 1º). São modelos
# diferentes do VLM, então não gastam a cota diária dele (a cota do free tier é por modelo).
# Free tier: o 3.5-flash permite só 20 pedidos/dia; o 3.1-flash-lite (o modelo do VLM, cuja coleta terminou)
# permite 500. Se os dois falharem, o texto padrão (sempre fiel) assume, sem travar a tool.
MODELOS_LLM_PADRAO = "gemini-3.5-flash,gemini-3.1-flash-lite"
TENTATIVAS_LLM = 2  # por modelo, só em sobrecarga momentânea (5xx ou limite por minuto)
NIVEL_INTERVALO_PCT = 80
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
    explicacao_fonte: str
    explicacao_problemas: list[str]
    saida: PrevisaoCriativo


def no_extrair_atributos(estado: Estado) -> Estado:
    """Tool 1: atributos do criativo (texto, imagem, vídeo e rubrica do VLM), com as funções do treino."""
    e = estado["entrada"]
    # um modelo de cliente sem vídeo nem VLM não precisa baixar o vídeo (a etapa mais lenta)
    partes = previsao.partes_usadas(previsao.artefato_da_entrada(e.modelo))
    usa_video = bool(partes & {"vídeo", "rubrica do VLM"})
    return {
        "atributos": extracao.extrair_isolado(
            e.creative, e.caption, e.description, e.tags, e.analisar_video and usa_video
        )
    }


def no_historico_canal(estado: Estado) -> Estado:
    """Tool 2: bloco A (histórico do canal e tempo) e criativos comparáveis."""
    e, atr = estado["entrada"], estado["atributos"]
    planned = pd.Timestamp(e.planned_datetime).tz_convert("UTC")
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
    contexto = {
        **hist,
        **canal.features_de_tempo(planned),
        "setor": e.sector,
        "tipo_canal": tipo,
        "categoria_id": categoria,
        "published_at": planned,
    }
    return {"contexto": contexto, "comparaveis": canal.comparaveis(atr, planned, e.sector, e.channel_id)}


def linha_do_modelo(atributos: dict, contexto: dict) -> dict:
    return {
        **{k: v for k, v in atributos.items() if not k.startswith("_")},
        **{k: v for k, v in contexto.items() if not k.startswith("_")},
    }


def no_prever(estado: Estado) -> Estado:
    """Tool 3 (+ SHAP da Tool 4): previsão, intervalo de 80%, faixa, atribuição e sinal de confiança."""
    e = estado["entrada"]
    linha = linha_do_modelo(estado["atributos"], estado["contexto"])
    art = previsao.artefato_da_entrada(e.modelo)
    contexto = estado["contexto"]
    if "rag_canal_alvo" in art["colunas"]:  # variante v2: desempenho dos comparáveis vira feature
        rag = canal.features_rag(estado["atributos"], contexto["published_at"], e.sector, e.channel_id)
        linha.update(rag)
        contexto = {**contexto, **rag}  # aparece na inspeção (atributos_extraidos)
    faltando = [p for p in estado["atributos"]["_faltando"] if p in previsao.partes_usadas(art)]
    return {"previsao": previsao.prever_linha(linha, art=art, extracao_incompleta=faltando), "contexto": contexto}


# ---------------------------------------------------------------- Tool 4: explicador ancorado


def fatos_para_explicar(estado: Estado) -> dict:
    """O JSON fechado que o explicador recebe: tudo o que ele pode citar, e nada além disso."""
    p, e = estado["previsao"], estado["entrada"]
    return {
        "legenda": estado["atributos"].get("_titulo", ""),
        "curtidas_previstas": p["curtidas_previstas"],
        "intervalo": {"nivel_pct": NIVEL_INTERVALO_PCT, "de": p["curtidas_baixo"], "ate": p["curtidas_alto"]},
        "mediana_do_canal": round(p["mediana_canal"]),
        "razao_vs_canal": p["razao_prevista"],
        "faixa": NOMES_FAIXA[p["faixa"]],
        "fatores": [
            {k: d[k] for k in ("descricao", "valor", "efeito", "impacto_pct")}
            for d in p["drivers"][: previsao.N_DRIVERS]
        ],
        "baixa_confianca": [m["texto"] for m in p["motivos_baixa_confianca"]],
        "comparaveis": [
            {k: c[k] for k in ("titulo", "curtidas", "desempenho_vs_canal")}
            for c in estado["comparaveis"][:3]
        ],
        "setor": e.sector,
    }


class FatorExplicado(BaseModel):
    fator: str = Field(description="descrição EXATA de um item de 'fatores' do JSON")
    efeito: str = Field(description="o efeito em linguagem de marketing, citando o impacto_pct do JSON")


class Explicacao(BaseModel):
    resumo: str = Field(description="2 ou 3 frases: curtidas previstas, intervalo e posição vs. o canal")
    fatores: list[FatorExplicado] = Field(description="de 2 a 4 fatores, na ordem do JSON")
    ressalvas: list[str] = Field(
        description="baixa confiança (se houver) e limites; a 1ª é a baixa confiança"
    )


INSTRUCOES_EXPLICADOR = """Você é o subagente que explica a previsão de um modelo de machine learning
para um time de marketing de moda e beleza. Você recebe os fatos já calculados (JSON) e preenche a
explicação. Regras:
1. Use SOMENTE fatos do JSON. Não invente causas, tendências, públicos nem números.
2. Cite os números exatamente como estão no JSON, com ponto de milhar nas curtidas.
   Conversões permitidas: razao_vs_canal em vezes ou em porcentagem a mais/menos, e impacto_pct
   arredondado para inteiro. Nenhuma outra conta.
3. No resumo, cite curtidas_previstas, o intervalo (de e ate) e a posição em relação ao canal.
4. Em `fatores`, use a descrição EXATA de 2 a 4 itens de "fatores", na mesma ordem. Fatores do canal
   (patamar, tendência, instabilidade) são contexto do histórico, não do criativo: diga isso.
5. Se "baixa_confianca" não estiver vazia, a primeira ressalva começa com "Baixa confiança:" e cita os
   motivos. Se houver comparáveis, use um como evidência numa ressalva.
6. Os fatores são associações aprendidas pelo modelo, não causas comprovadas.
7. Português do Brasil, sem jargão (não use "SHAP", "log", "embedding", "PCA", "feature")."""

_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def _para_float(txt: str) -> float:
    """'4.380' → 4380; '1,33' → 1.33 (pt-BR); '1.5' → 1.5."""
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", txt):
        return float(txt.replace(".", ""))
    if "," in txt:
        return float(txt.replace(".", "").replace(",", "."))
    return float(txt)


def numeros_permitidos(fatos) -> set[float]:
    """Todo número do JSON (inclusive dentro de textos) e as conversões que as instruções autorizam."""
    ok: set[float] = set()

    def visitar(x):
        if isinstance(x, bool):
            return
        if isinstance(x, (int, float)):
            v = float(x)
            ok.update({round(v, 2), round(abs(v), 2), float(round(v)), float(round(abs(v)))})
            if 0 < v < 20:  # razão em vezes → % a mais/menos (1,23× = 23% a mais; 0,8× = 20% a menos)
                ok.update({float(round(abs(v - 1) * 100)), round(abs(v - 1) * 100, 1), round(v * 100)})
        elif isinstance(x, str):
            ok.update(round(_para_float(n), 2) for n in _NUM.findall(x))
        elif isinstance(x, dict):
            for v in x.values():
                visitar(v)
        elif isinstance(x, list):
            for v in x:
                visitar(v)

    visitar(fatos)
    return ok


def validar_explicacao(exp: dict, fatos: dict) -> list[str]:
    """Checagem determinística de fidelidade (aula 5.7). Lista vazia = passou."""
    problemas = []
    texto = json.dumps(exp, ensure_ascii=False)
    citados = {round(_para_float(n), 2) for n in _NUM.findall(texto)}
    for n in sorted(citados - numeros_permitidos(fatos)):
        problemas.append(f"número que não está no JSON: {n:g}")
    nomes = {f["descricao"] for f in fatos["fatores"]}
    for f in exp.get("fatores", []):
        if f["fator"] not in nomes:
            problemas.append(f"fator que não está no JSON: {f['fator']}")
    for campo in ("de", "ate"):
        if float(fatos["intervalo"][campo]) not in citados:
            problemas.append(f"o intervalo não foi citado ({campo} = {fatos['intervalo'][campo]})")
    if fatos["baixa_confianca"] and not (exp.get("ressalvas") or [""])[0].startswith("Baixa confiança"):
        problemas.append("há motivos de baixa confiança, mas a 1ª ressalva não começa com 'Baixa confiança'")
    return problemas


def renderizar(exp: dict) -> str:
    fatores = "; ".join(f"{f['fator'].lower()}: {f['efeito'].rstrip('.')}" for f in exp["fatores"])
    partes = [exp["resumo"].strip(), f"Principais fatores: {fatores}." if fatores else "", *exp["ressalvas"]]
    return " ".join(p.strip() for p in partes if p.strip())


def _fmt(n) -> str:
    return f"{n:,}".replace(",", ".")


def explicacao_padrao(f: dict) -> str:
    """Explicação sem LLM, só com os fatos (se o LLM falhar, reprovar duas vezes ou não houver chave)."""
    i, razao = f["intervalo"], f"{f['razao_vs_canal']:.2f}".replace(".", ",")
    texto = (
        f"Previsão: {_fmt(f['curtidas_previstas'])} curtidas (intervalo de {i['nivel_pct']}%: "
        f"{_fmt(i['de'])} a {_fmt(i['ate'])}), {razao}× a mediana recente do canal "
        f"({_fmt(f['mediana_do_canal'])}), ou seja, {f['faixa']}. Principais fatores: "
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
    return texto + " Os fatores são associações aprendidas pelo modelo, não causas comprovadas."


def _cliente_llm():
    load_dotenv(".env")
    chave = os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not chave:
        return None
    from google import genai
    from google.genai import types

    return genai.Client(api_key=chave, http_options=types.HttpOptions(timeout=60_000))


def _chamar_llm(instrucoes: str, conteudo: str, esquema=None) -> str | None:
    """Resposta do LLM (texto ou JSON do `esquema`), tentando os modelos em ordem; None se todos falharem."""
    cliente = _cliente_llm()
    if cliente is None:
        return None
    from google.genai import types

    # sem "pensamento" interno: a tarefa é redigir a partir de fatos prontos (cai de ~30 s para ~2 s), e a
    # fidelidade é garantida pelo validador, não pelo raciocínio do modelo
    config = {"system_instruction": instrucoes, "temperature": 0.0,
              "thinking_config": types.ThinkingConfig(thinking_budget=0)}
    if esquema is not None:
        config.update(response_mime_type="application/json", response_schema=esquema)
    for nome in os.environ.get("AGENTE_MODELO", MODELOS_LLM_PADRAO).split(","):  # lido depois do .env
        for tentativa in range(TENTATIVAS_LLM):
            try:
                r = cliente.models.generate_content(
                    model=nome.strip(), contents=conteudo, config=types.GenerateContentConfig(**config)
                )
                if texto := (r.text or "").strip():
                    return texto
                break
            except Exception as e:  # noqa: BLE001 — cota, rede ou modelo indisponível
                texto_erro = str(e).lower()
                cota_do_dia = "perday" in texto_erro or "per day" in texto_erro
                temporario = getattr(e, "code", None) in (429, 500, 503, 504) and not cota_do_dia
                if not temporario or tentativa == TENTATIVAS_LLM - 1:
                    break  # tenta o próximo modelo
                time.sleep(2)
    return None


def redigir_explicacao(fatos: dict, tentativas: int = 2) -> tuple[str, str, list[str]]:
    """(texto, fonte, problemas). Fonte: "llm" (aprovado pelo validador na 1ª tentativa), "llm_corrigido"
    (aprovado na 2ª) ou "padrao" (texto montado sem LLM); `problemas` = o que o validador reprovou."""
    conteudo = json.dumps(fatos, ensure_ascii=False, default=str)
    todos = []
    for t in range(tentativas):
        bruto = _chamar_llm(INSTRUCOES_EXPLICADOR, conteudo, Explicacao)
        if bruto is None:
            todos.append("LLM indisponível (sem chave, cota ou rede)")
            break
        try:
            exp = Explicacao.model_validate_json(bruto).model_dump()
        except Exception:  # noqa: BLE001 — JSON fora do esquema: conta como tentativa reprovada
            todos.append("resposta fora do esquema")
            continue
        problemas = validar_explicacao(exp, fatos)
        if not problemas:
            return renderizar(exp), ("llm" if t == 0 else "llm_corrigido"), todos
        todos += problemas
        conteudo = (
            f"{json.dumps(fatos, ensure_ascii=False, default=str)}\n\nSua resposta anterior:\n{bruto}"
            f"\n\nCorrija estes problemas e responda de novo: {'; '.join(problemas)}"
        )
    return explicacao_padrao(fatos), "padrao", todos


def no_explicar(estado: Estado) -> Estado:
    """Tool 4 (subagente): transforma fatores e números em justificativa em linguagem natural."""
    texto, fonte, problemas = redigir_explicacao(fatos_para_explicar(estado))
    return {"explicacao": texto, "explicacao_fonte": fonte, "explicacao_problemas": problemas}


def _valor_simples(v):
    if isinstance(v, (np.generic,)):
        v = v.item()
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, pd.Timestamp):
        return v.tz_convert(FUSO).isoformat(timespec="minutes")
    return v if isinstance(v, (int, float, str, bool)) or v is None else str(v)


def no_montar_saida(estado: Estado) -> Estado:
    p, e = estado["previsao"], estado["entrada"]
    inspecao = {
        k: _valor_simples(v)
        for k, v in linha_do_modelo(estado["atributos"], estado["contexto"]).items()
        if not k.startswith(("embtxt_", "embimg_", "embgancho_"))
    }
    inspecao["tipo_criativo"] = estado["atributos"]["_tipo"]
    inspecao["partes_nao_analisadas"] = ", ".join(estado["atributos"]["_faltando"]) or None
    inspecao["explicacao_fonte"] = estado["explicacao_fonte"]
    inspecao["aviso_modelo"] = None if e.modelo else previsao.aviso_modelo()
    inspecao["explicacao_reprovacoes"] = "; ".join(estado["explicacao_problemas"]) or None
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


Z_80 = 1.2816  # quantil 0,9 da normal padrão: o intervalo de 80% tem ±1,2816 desvios


def prob_superar(a: PrevisaoCriativo, b: PrevisaoCriativo) -> float:
    """P(a rende mais que b), com cada previsão vista como normal em log1p(curtidas): centro na previsão e
    desvio = largura do intervalo de 80% / (2 × 1,2816). Supõe erros independentes (conservador)."""
    from scipy.stats import norm

    def centro_e_desvio(p):
        lo, hi = np.log1p(p.interval_low), np.log1p(p.interval_high)
        return np.log1p(p.predicted_metric), max((hi - lo) / (2 * Z_80), 1e-6)

    (ma, sa), (mb, sb) = centro_e_desvio(a), centro_e_desvio(b)
    return round(float(norm.cdf((ma - mb) / np.hypot(sa, sb))), 2)


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
                prob_superar_seguinte=prob_superar(p, ordem[i]) if i < len(ordem) else None,
            )
            for i, p in enumerate(ordem, 1)
        ]
    }


INSTRUCOES_RANKING = """Você explica a um time de marketing o ranking de criativos feito por um modelo
de machine learning. Use SOMENTE os números e fatores do JSON, sem fazer contas. Em 3 a 5 frases, em
português do Brasil: diga qual criativo deve render mais, citando as curtidas previstas e os intervalos;
aponte os fatores que mais separam o primeiro dos demais; se os intervalos se sobrepõem, diga que a
diferença é incerta; se algum criativo tem baixa confiança, diga isso. Sem jargão técnico."""


def ranking_padrao(r: list[ItemRanking]) -> str:
    texto = " > ".join(f"{x.id} ({_fmt(x.predicted_metric)} curtidas)" for x in r) + "."
    if len(r) > 1:
        texto += (
            f" Chance de {r[0].id} render mais que {r[1].id}: {round(r[0].prob_superar_seguinte * 100)}%."
        )
        if r[1].interval_high >= r[0].interval_low:
            texto += " Os intervalos se sobrepõem: a diferença entre os primeiros é incerta."
    if baixa := [x.id for x in r if x.confidence_flag]:
        texto += f" Baixa confiança em: {', '.join(baixa)}."
    return texto


def no_explicar_ranking(estado: EstadoRanking) -> EstadoRanking:
    por_id = {p.id: p for p in estado["previsoes"]}
    fatos = {
        "ranking": [
            {
                "posicao": r.posicao,
                "id": r.id,
                "curtidas_previstas": r.predicted_metric,
                "intervalo_80": [r.interval_low, r.interval_high],
                "razao_vs_canal": r.razao_prevista,
                "chance_de_superar_o_seguinte": r.prob_superar_seguinte,
                "baixa_confianca": [m.texto for m in por_id[r.id].confidence_reasons],
                "fatores": [
                    {
                        "descricao": d.descricao,
                        "valor": d.valor,
                        "efeito": d.efeito,
                        "impacto_pct": d.impacto_pct,
                    }
                    for d in por_id[r.id].top_drivers[:4]
                ],
            }
            for r in estado["ranking"]
        ]
    }
    texto = _chamar_llm(INSTRUCOES_RANKING, json.dumps(fatos, ensure_ascii=False))
    if texto:
        citados = {round(_para_float(n), 2) for n in _NUM.findall(texto)}
        if citados - numeros_permitidos(fatos):  # número inventado: o texto padrão assume
            texto = None
    return {"explicacao": texto or ranking_padrao(estado["ranking"])}


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
                "confidence_flag": bool(p["motivos_baixa_confianca"]),
            }
        )
    return saida


def what_if_legendas(entrada: EntradaCriativo | dict[str, Any], legendas: list[str]) -> list[dict]:
    """O mesmo criativo com outras legendas: imagem, vídeo e VLM são extraídos uma vez; só o texto (bloco B)
    é recalculado, com as mesmas funções do treino. O primeiro item é a legenda original."""
    entrada = EntradaCriativo.model_validate(entrada)
    estado: Estado = {"entrada": entrada, **no_extrair_atributos({"entrada": entrada})}
    estado.update(no_historico_canal(estado))
    original = estado["atributos"]["_titulo"]
    saida = []
    for legenda in [original, *legendas]:
        texto = estado["atributos"] if legenda == original else extracao.texto_isolado(
            legenda, entrada.description, entrada.tags
        )
        atr = {**estado["atributos"], **texto, "_titulo": legenda}
        p = no_prever({**estado, "atributos": atr})["previsao"]
        saida.append(
            {
                "legenda": legenda,
                "predicted_metric": p["curtidas_previstas"],
                "interval_low": p["curtidas_baixo"],
                "interval_high": p["curtidas_alto"],
                "razao_prevista": p["razao_prevista"],
                "efeito_do_texto_pct": p["contribuicao_por_bloco"].get("B (texto)"),
                "confidence_flag": bool(p["motivos_baixa_confianca"]),
            }
        )
    return saida
