"""Interface opcional (TAPI, seção 3: "Streamlit/Gradio opcional") sobre o mesmo agente da tool.

Uso: uv sync --group app && uv run streamlit run src/galaxies_tool/app.py

Mostra o agente trabalhando: cada tool do grafo (extração, histórico, previsão, explicação) aparece quando
termina, via `stream` do LangGraph. O número vem sempre do modelo; a tela só apresenta a saída do contrato
(PrevisaoCriativo / RankingCriativos), com os fatores, os comparáveis e a inspeção dos atributos.
Modo "respostas salvas": reabre as respostas da demonstração (resultados/demo/) sem rede nem modelo.
"""

import json
import os
from datetime import datetime, time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

os.chdir(Path(__file__).resolve().parents[2])  # dados e modelos usam caminhos relativos à raiz

# paleta validada (skill de dataviz): azul = aumenta, vermelho = reduz; cinza neutro no zero
AZUL, VERMELHO, NEUTRO = "#2a78d6", "#e34948", "#8a8984"
ETAPAS = {
    "extrair_atributos": "Tool 1 · atributos do criativo (texto, capa, vídeo, VLM)",
    "historico_canal": "Tool 2 · histórico do canal e comparáveis",
    "prever": "Tool 3 · previsão, intervalo e SHAP",
    "explicar": "Tool 4 · explicação (subagente com validador)",
    "montar_saida": "Saída validada (contrato Pydantic)",
}

st.set_page_config(page_title="Galaxies · preditor de Shorts", layout="wide")


def fmt(n) -> str:
    return f"{int(n):,}".replace(",", ".")


@st.cache_data
def canais() -> pd.DataFrame:
    c = pd.read_csv("data/canais.csv", dtype=str).fillna("")
    return c[c.incluir == "sim"].sort_values("titulo")


def grafico_fatores(p: dict) -> go.Figure:
    d = pd.DataFrame(p["top_drivers"]).iloc[::-1]
    rotulo = d.descricao + " = " + d.valor.astype(str)
    fig = go.Figure(go.Bar(
        x=d.impacto_pct, y=rotulo, orientation="h",
        marker_color=[AZUL if v > 0 else VERMELHO for v in d.impacto_pct], marker_line_width=0,
        text=[f"{v:+.0f}%" for v in d.impacto_pct], textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>efeito: %{x:+.1f}% nas curtidas previstas<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=NEUTRO, line_width=1)
    fig.update_layout(height=60 + 34 * len(d), margin=dict(l=0, r=40, t=10, b=10), bargap=0.35,
                      xaxis=dict(title="efeito nas curtidas previstas (%)", showgrid=False, zeroline=False),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def grafico_ranking(previsoes: list[dict], reais: dict | None = None) -> go.Figure:
    d = pd.DataFrame([{"id": p["id"], "prev": p["predicted_metric"], "lo": p["interval_low"],
                       "hi": p["interval_high"]} for p in previsoes]).sort_values("prev")
    fig = go.Figure(go.Scatter(
        x=d.prev, y=d.id, mode="markers", marker=dict(size=11, color=AZUL), name="previsto",
        error_x=dict(type="data", symmetric=False, array=d.hi - d.prev, arrayminus=d.prev - d.lo,
                     color=AZUL, thickness=2, width=6),
        hovertemplate="%{y}: %{x:,} curtidas previstas<extra></extra>",
    ))
    if reais:
        fig.add_trace(go.Scatter(x=[reais[i] for i in d.id], y=d.id, mode="markers", name="real",
                                 marker=dict(size=11, symbol="diamond-open", color=VERMELHO, line_width=2),
                                 hovertemplate="%{y}: %{x:,} curtidas reais<extra></extra>"))
    fig.update_layout(height=80 + 50 * len(d), margin=dict(l=0, r=10, t=10, b=10), showlegend=bool(reais),
                      legend=dict(orientation="h", y=1.15),
                      xaxis=dict(type="log", title="curtidas (escala log; barra = intervalo de 80%)"),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def mostrar_previsao(p: dict, real: int | None = None):
    rel = p["relative_to_channel"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Curtidas previstas", fmt(p["predicted_metric"]))
    c2.metric("Intervalo de 80%", f"{fmt(p['interval_low'])} – {fmt(p['interval_high'])}")
    c3.metric("vs. mediana do canal", f"{rel['razao_prevista']:.2f}×".replace(".", ","),
              help=f"Mediana recente do canal: {fmt(rel['mediana_canal'])} curtidas")
    c4.metric("Faixa", rel["faixa"].replace("_", " "))
    if real is not None:
        st.caption(f"Real (só para o de-para; o agente não viu): **{fmt(real)}** curtidas")
    if p["confidence_flag"]:
        st.warning("**Baixa confiança:** " + "; ".join(m["texto"] for m in p["confidence_reasons"]))
    st.markdown(f"**Por quê:** {p['explanation']}")
    st.plotly_chart(grafico_fatores(p), width="stretch")
    with st.expander("Contribuição por bloco, comparáveis e inspeção dos atributos"):
        st.write("Efeito somado de cada bloco de features (%):", p["contribuicao_por_bloco"])
        if p["comparables"]:
            st.dataframe(pd.DataFrame(p["comparables"]), hide_index=True, width="stretch")
        st.json(p["atributos_extraidos"], expanded=False)
        st.caption(f"Modelo: {p['modelo_versao']}")


def rodar_criativo(entrada: dict) -> dict:
    """Roda o grafo de um criativo mostrando cada tool quando ela termina."""
    from galaxies_tool.agente import GRAFO
    from galaxies_tool.contrato import EntradaCriativo

    e = EntradaCriativo.model_validate(entrada)
    with st.status(f"Analisando {e.id or 'criativo'}…", expanded=True) as status:
        saida = None
        for passo in GRAFO.stream({"entrada": e}, stream_mode="updates"):
            for no, valor in passo.items():
                st.write("✓ " + ETAPAS.get(no, no))
                if no == "montar_saida":
                    saida = valor["saida"]
        status.update(label=f"{e.id or 'criativo'}: pronto", state="complete", expanded=False)
    return saida.model_dump(mode="json")


st.title("Preditor de curtidas de YouTube Shorts (moda e beleza)")
st.caption("Antes de publicar: previsão em relação ao padrão do canal, intervalo de 80%, fatores e comparáveis. "
           "O número vem de um modelo treinado (LightGBM + CatBoost); o LLM só redige a explicação.")
modo = st.sidebar.radio("Modo", ["Ao vivo", "Respostas salvas da demonstração (sem rede)"])

if modo.startswith("Respostas"):
    arquivos = sorted(Path("resultados/demo").glob("UC*.json"))
    de_para = pd.read_csv("resultados/demo/de_para.csv")
    reais = dict(zip(de_para.video_id, de_para.real, strict=True))
    nomes = dict(zip(de_para.video_id, de_para.canal, strict=True))
    escolha = st.sidebar.selectbox("Canal (fora do treino)", arquivos,
                                   format_func=lambda a: nomes[json.loads(a.read_text())["ranking"][0]["id"]])
    r = json.loads(escolha.read_text())
    st.subheader("Ranking dos criativos")
    st.plotly_chart(grafico_ranking(r["previsoes"], reais), width="stretch")
    st.markdown(r["explanation"])
    for p in sorted(r["previsoes"], key=lambda p: -p["predicted_metric"]):
        st.divider()
        st.markdown(f"#### [{p['id']}](https://www.youtube.com/shorts/{p['id']})")
        mostrar_previsao(p, reais.get(p["id"]))
else:
    c = canais()
    canal = st.sidebar.selectbox("Canal da base", c.channel_id, format_func=lambda i: c.set_index(
        "channel_id").titulo[i])
    setor = c.set_index("channel_id").setor[canal]
    dia = st.sidebar.date_input("Data planejada", datetime.now().date())
    hora = st.sidebar.time_input("Hora (Brasília)", time(18, 0))
    analisar = st.sidebar.checkbox("Analisar o vídeo (download + Whisper + VLM, ~1 min por vídeo novo)", True)
    n = st.sidebar.number_input("Quantos criativos", 1, 5, 2)
    entradas = []
    for i in range(int(n)):
        a, b = st.columns([3, 2])
        criativo = a.text_input(f"Criativo {i + 1}: URL do YouTube, caminho de vídeo/imagem ou texto",
                                key=f"c{i}")
        legenda = b.text_input("Legenda planejada (vazio = a publicada, se for URL)", key=f"l{i}")
        if criativo:
            entradas.append({"id": f"criativo_{i + 1}", "creative": criativo, "caption": legenda,
                             "planned_datetime": datetime.combine(dia, hora).isoformat(), "channel_id": canal,
                             "sector": setor, "analisar_video": analisar})
    if st.button("Prever", type="primary", disabled=not entradas):
        from galaxies_tool.agente import no_explicar_ranking, no_ordenar
        from galaxies_tool.contrato import PrevisaoCriativo

        try:
            previsoes = [rodar_criativo(e) for e in entradas]
            ranking = None
            if len(previsoes) > 1:
                objs = [PrevisaoCriativo.model_validate(p) for p in previsoes]
                estado = {"previsoes": objs, **no_ordenar({"previsoes": objs})}
                ranking = no_explicar_ranking(estado)["explicacao"]
            st.session_state["resultado"] = (previsoes, ranking)
        except Exception as e:  # noqa: BLE001 — erro de entrada ou de rede aparece na tela
            st.error(f"{type(e).__name__}: {e}")
    if res := st.session_state.get("resultado"):
        previsoes, ranking = res
        if ranking:
            st.subheader("Ranking")
            st.plotly_chart(grafico_ranking(previsoes), width="stretch")
            st.markdown(ranking)
        for p in sorted(previsoes, key=lambda p: -p["predicted_metric"]):
            st.divider()
            st.markdown(f"#### {p['id']}")
            mostrar_previsao(p)
