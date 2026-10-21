"""Interface opcional (TAPI, seção 3: "Streamlit/Gradio opcional") sobre o mesmo agente da tool.

Uso (na raiz do repositório): uv run --group app streamlit run src/galaxies_tool/app.py

Mostra o agente trabalhando: cada tool do grafo (extração, histórico, previsão, explicação) aparece quando
termina, via `stream` do LangGraph. O número vem sempre do modelo; a tela só apresenta a saída do contrato
(PrevisaoCriativo / RankingCriativos), com os fatores, os comparáveis e a inspeção dos atributos.
Modo "respostas salvas": reabre as respostas da demonstração (resultados/demo/) sem rede nem modelo.
"""

import html
import json
import os
import re
from datetime import datetime, time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

os.chdir(Path(__file__).resolve().parents[2])  # dados e modelos usam caminhos relativos à raiz

# paleta validada (skill de dataviz): azul = aumenta, vermelho = reduz; cinza neutro no zero
AZUL, VERMELHO, NEUTRO = "#2a78d6", "#e34948", "#8a8984"
RX_YOUTUBE = re.compile(r"(?:youtube\.com/(?:shorts/|watch\?v=|embed/)|youtu\.be/)([\w-]{11})")
ETAPAS = {
    "extrair_atributos": "Tool 1 · atributos do criativo (texto, capa, vídeo, VLM)",
    "historico_canal": "Tool 2 · histórico do canal e comparáveis",
    "prever": "Tool 3 · previsão, intervalo e fatores (SHAP)",
    "explicar": "Tool 4 · explicação (subagente com validador)",
    "montar_saida": "Saída validada (contrato Pydantic)",
}
FAIXAS = {"abaixo": "abaixo da média", "na_media": "na média", "acima": "acima da média", "viral": "viral"}

st.set_page_config(page_title="Galaxies · preditor de Shorts", page_icon="📈", layout="wide")
st.markdown(
    """
<style>
.block-container {padding-top: 3.2rem; max-width: 1180px;}
[data-testid="stMetric"] {background: rgba(128,128,128,.06); border: 1px solid rgba(128,128,128,.16);
  border-radius: 12px; padding: 10px 14px;}
[data-testid="stMetricLabel"] p {font-size: .78rem; opacity: .75;}
[data-testid="stMetricValue"] {font-size: 1.45rem;}
.gx-eyebrow {font-size: .75rem; letter-spacing: .12em; text-transform: uppercase; opacity: .6; margin: 0;}
.gx-title {font-size: 2rem; font-weight: 700; line-height: 1.15; margin: .15rem 0 .35rem 0;}
.gx-sub {opacity: .75; margin: 0 0 .6rem 0; max-width: 820px;}
.gx-chip {display: inline-block; padding: 2px 10px; margin: 0 6px 6px 0; border-radius: 999px;
  font-size: .78rem; border: 1px solid rgba(128,128,128,.3);}
.gx-badge {display: inline-block; padding: 3px 10px; border-radius: 8px; font-size: .82rem; margin: 2px 0 8px 0;}
.gx-ok {background: rgba(12,163,12,.12);}
.gx-alerta {background: rgba(250,178,25,.20);}
.gx-pos {font-size: 1.6rem; font-weight: 700; color: #2a78d6; line-height: 1;}
.gx-muted {opacity: .65; font-size: .85rem;}
.gx-comp {font-size: .8rem; line-height: 1.25;}
.gx-comp img {width: 100%; border-radius: 8px; margin-bottom: 4px; aspect-ratio: 16/9; object-fit: cover;}
</style>
""",
    unsafe_allow_html=True,
)


def fmt(n) -> str:
    return f"{int(round(n)):,}".replace(",", ".")


def br(x: float, casas: int = 2) -> str:
    return f"{x:.{casas}f}".replace(".", ",")


def id_youtube(texto: str | None) -> str | None:
    if not texto:
        return None
    if m := RX_YOUTUBE.search(texto):
        return m.group(1)
    return texto if re.fullmatch(r"[\w-]{11}", texto) else None


@st.cache_data
def canais() -> pd.DataFrame:
    c = pd.read_csv("data/canais.csv", dtype=str).fillna("")
    return c[c.incluir == "sim"].sort_values("titulo")


# ---------------------------------------------------------------- gráficos


def _layout(fig: go.Figure, altura: int, **kw) -> go.Figure:
    fig.update_layout(height=altura, margin=dict(l=0, r=16, t=8, b=8), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(size=13), hoverlabel=dict(font_size=13), **kw)
    return fig


def grafico_fatores(p: dict) -> go.Figure:
    d = pd.DataFrame(p["top_drivers"]).iloc[::-1]
    rotulo = [f"{a} <span style='color:{NEUTRO}'>· {v}</span>" if v != "—" else a
              for a, v in zip(d.descricao, d.valor.astype(str), strict=True)]
    fig = go.Figure(go.Bar(
        x=d.impacto_pct, y=rotulo, orientation="h",
        marker=dict(color=[AZUL if v > 0 else VERMELHO for v in d.impacto_pct], cornerradius=4),
        text=[f"{v:+.0f}%" for v in d.impacto_pct], textposition="outside", cliponaxis=False,
        customdata=d.efeito, hovertemplate="%{y}<br>%{customdata} as curtidas previstas em %{x:+.1f}%<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=NEUTRO, line_width=1)
    lim = max(d.impacto_pct.abs().max() * 1.35, 10)
    return _layout(fig, 50 + 36 * len(d), bargap=0.4, showlegend=False,
                   xaxis=dict(range=[-lim, lim], showgrid=False, zeroline=False, ticksuffix="%",
                              title=dict(text="efeito nas curtidas previstas", font=dict(size=12))),
                   yaxis=dict(automargin=True))


def grafico_ranking(previsoes: list[dict], reais: dict | None = None, nomes: dict | None = None) -> go.Figure:
    nomes = nomes or {}
    d = pd.DataFrame([{"id": p["id"], "rot": nomes.get(p["id"], p["id"]), "prev": p["predicted_metric"],
                       "lo": p["interval_low"], "hi": p["interval_high"]} for p in previsoes]).sort_values("prev")
    fig = go.Figure()
    for r in d.itertuples():  # intervalo de 80% como faixa larga e leve; previsão como ponto
        fig.add_trace(go.Scatter(x=[r.lo, r.hi], y=[r.rot, r.rot], mode="lines", showlegend=False,
                                 line=dict(color="rgba(42,120,214,.25)", width=12), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=d.prev, y=d.rot, mode="markers+text", name="previsto", marker=dict(size=13, color=AZUL),
        text=[fmt(v) for v in d.prev], textposition="top center", textfont=dict(size=12),
        customdata=d[["lo", "hi"]].to_numpy(),
        hovertemplate="%{y}<br>previsto: %{x:,.0f} curtidas<br>intervalo de 80%: %{customdata[0]:,.0f} a "
                      "%{customdata[1]:,.0f}<extra></extra>",
    ))
    if reais:
        fig.add_trace(go.Scatter(
            x=[reais[i] for i in d.id], y=d.rot, mode="markers", name="real (de-para)",
            marker=dict(size=13, symbol="diamond-open", color=VERMELHO, line_width=2.5),
            hovertemplate="%{y}<br>real: %{x:,.0f} curtidas<extra></extra>"))
    mediana = previsoes[0]["relative_to_channel"]["mediana_canal"]
    fig.add_trace(go.Scatter(x=[mediana, mediana], y=[d.rot.iloc[0], d.rot.iloc[-1]], mode="lines",
                             name=f"mediana do canal ({fmt(mediana)})", line=dict(color=NEUTRO, dash="dot", width=1.5),
                             hoverinfo="skip"))
    lo, hi = min(d.lo.min(), mediana, *(reais or {0: d.lo.min()}).values()), max(d.hi.max(), mediana)
    ticks = [m * 10**k for k in range(0, 9) for m in (1, 2, 5) if lo / 1.5 <= m * 10**k <= hi * 1.5]
    return _layout(fig, 110 + 64 * len(d), showlegend=True,
                   legend=dict(orientation="h", y=-0.25, x=0),
                   xaxis=dict(type="log", showgrid=True, gridcolor="rgba(128,128,128,.15)", tickvals=ticks,
                              ticktext=[fmt(t) for t in ticks],
                              title=dict(text="curtidas (escala log) · faixa clara = intervalo de 80%",
                                         font=dict(size=12))),
                   yaxis=dict(automargin=True))


# ---------------------------------------------------------------- blocos da tela


def cabecalho():
    st.markdown(
        """
<p class="gx-eyebrow">Galaxies · agente preditivo</p>
<div class="gx-title">Quantas curtidas este Short deve ter?</div>
<p class="gx-sub">Previsão antes da publicação, em relação ao padrão do canal, com intervalo de 80%, os fatores que
pesaram e Shorts comparáveis. O número vem de um modelo treinado (LightGBM + CatBoost); o LLM só redige a
explicação, e um validador confere cada número citado.</p>
<span class="gx-chip">YouTube Shorts · moda e beleza</span><span class="gx-chip">~60% de acerto ao ordenar
criativos do mesmo canal</span><span class="gx-chip">intervalo de 80% calibrado</span><span class="gx-chip">mesma
tool do servidor MCP</span>
""",
        unsafe_allow_html=True,
    )


def selo_confianca(p: dict):
    if p["confidence_flag"]:
        motivos = "; ".join(html.escape(m["texto"]) for m in p["confidence_reasons"])
        st.markdown(f'<span class="gx-badge gx-alerta">⚠ <b>Baixa confiança</b> · {motivos}</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="gx-badge gx-ok">✓ Confiança normal</span>', unsafe_allow_html=True)


def selo_faixa(p: dict):
    f = p["relative_to_channel"]["faixa"]
    st.markdown(f'<span class="gx-chip">Faixa prevista: <b>{FAIXAS[f]}</b></span>', unsafe_allow_html=True)


def comparaveis(p: dict):
    comps = p["comparables"][:4]
    if not comps:
        return
    st.markdown("**Shorts parecidos do histórico** (evidência: como foram de verdade)")
    for col, c in zip(st.columns(len(comps)), comps, strict=False):
        titulo = html.escape(c["titulo"][:70] + ("…" if len(c["titulo"]) > 70 else ""))
        col.markdown(
            f'<div class="gx-comp"><a href="{c["url"]}" target="_blank"><img src="https://i.ytimg.com/vi/'
            f'{c["video_id"]}/mqdefault.jpg" onerror="this.style.display=\'none\'"></a>{titulo}<br>'
            f'<span class="gx-muted">{fmt(c["curtidas"])} curtidas · {br(c["desempenho_vs_canal"], 1)}× a '
            f'mediana · {c["publicado_em"]}</span></div>',
            unsafe_allow_html=True,
        )


def cartao(p: dict, posicao: int | None = None, criativo: str | None = None, real: int | None = None,
           mostrar_video: bool = True):
    rel = p["relative_to_channel"]
    vid = id_youtube(criativo or p.get("id"))
    with st.container(border=True):
        cab = f'<span class="gx-pos">{posicao}º</span>&nbsp;&nbsp;' if posicao else ""
        nome = html.escape(p["id"] or "criativo")
        link = f'<a href="https://www.youtube.com/shorts/{vid}" target="_blank">{nome}</a>' if vid else nome
        st.markdown(f"{cab}<span style='font-size:1.15rem;font-weight:600'>{link}</span>", unsafe_allow_html=True)
        esq, dir_ = st.columns([1, 2.6], gap="large") if (vid and mostrar_video) else (None, st.container())
        if esq is not None:
            with esq:
                # player embutido do YouTube em pé (o st.video não aceita links /shorts/)
                st.iframe(f"https://www.youtube.com/embed/{vid}?rel=0", height=470)
        with dir_:
            a, b = st.columns([1, 1])
            with a:
                selo_confianca(p)
            with b:
                selo_faixa(p)
            c1, c2, c3 = st.columns(3)
            c1.metric("Curtidas previstas", fmt(p["predicted_metric"]))
            c2.metric("Intervalo de 80%", f"{fmt(p['interval_low'])} – {fmt(p['interval_high'])}")
            c3.metric("vs. mediana do canal", f"{br(rel['razao_prevista'])}×",
                      help=f"Mediana recente do canal: {fmt(rel['mediana_canal'])} curtidas")
            if real is not None:
                dentro = p["interval_low"] <= real <= p["interval_high"]
                st.markdown(f'<span class="gx-muted">De-para (o agente não viu): <b>{fmt(real)}</b> curtidas reais '
                            f'· {"dentro" if dentro else "fora"} do intervalo</span>', unsafe_allow_html=True)
            st.markdown(f"**Por quê:** {p['explanation']}")
        st.markdown("**O que mais pesou** · azul aumenta, vermelho reduz")
        st.plotly_chart(grafico_fatores(p), width="stretch", config={"displayModeBar": False})
        comparaveis(p)
        with st.expander("Inspeção: contribuição por bloco e todos os atributos que entraram no modelo"):
            blocos = pd.DataFrame([{"bloco": b, "efeito (%)": v} for b, v in p["contribuicao_por_bloco"].items()])
            st.dataframe(blocos, hide_index=True, width="stretch")
            st.json(p["atributos_extraidos"], expanded=False)
            st.caption(f"Modelo: {p['modelo_versao']}")


def resumo_ranking(ranking: list[dict], nomes: dict | None = None):
    nomes = nomes or {}
    linhas = []
    for r in ranking:
        chance = r.get("prob_superar_seguinte")
        linhas.append({
            "posição": f"{r['posicao']}º", "criativo": nomes.get(r["id"], r["id"]),
            "curtidas previstas": fmt(r["predicted_metric"]),
            "intervalo de 80%": f"{fmt(r['interval_low'])} – {fmt(r['interval_high'])}",
            "vs. canal": f"{br(r['razao_prevista'])}×",
            "chance de superar o próximo": f"{round(chance * 100)}%" if chance is not None else "—",
            "confiança": "⚠ baixa" if r["confidence_flag"] else "✓ normal",
        })
    st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch")


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


# ---------------------------------------------------------------- tema claro/escuro

TEMAS = {  # mesmas cores da paleta validada, com os passos de cada modo
    "Claro": {"base": "light", "primaryColor": "#2a78d6", "backgroundColor": "#fcfcfb",
              "secondaryBackgroundColor": "#f3f2ef", "textColor": "#1b1b1a"},
    "Escuro": {"base": "dark", "primaryColor": "#3987e5", "backgroundColor": "#1a1a19",
               "secondaryBackgroundColor": "#262624", "textColor": "#f0efec"},
}


def aplicar_tema(tema: str):
    """Troca o tema da sessão (opções de tema do Streamlit) e recarrega a página uma vez."""
    if st.session_state.get("tema_aplicado") == tema:
        return
    for chave, valor in TEMAS[tema].items():
        st._config.set_option(f"theme.{chave}", valor)
    st.session_state["tema"] = st.session_state["tema_aplicado"] = tema
    st.rerun()


# ---------------------------------------------------------------- página

with st.sidebar:
    st.markdown("### Modo")
    modo = st.segmented_control(
        "Modo", ["Ao vivo", "Demo salva"], default="Ao vivo", label_visibility="collapsed", width="stretch",
        format_func=lambda m: {"Ao vivo": ":material/bolt: Ao vivo", "Demo salva": ":material/inventory_2: Demo salva"}[m],
    ) or "Ao vivo"
    st.caption("Ao vivo roda o agente agora; Demo salva reabre as respostas da demonstração, sem rede.")
    st.markdown("### Exibição")
    tema = st.segmented_control(
        "Tema", ["Claro", "Escuro"], default=st.session_state.get("tema", "Claro"), width="stretch",
        format_func=lambda t: {"Claro": ":material/light_mode: Claro", "Escuro": ":material/dark_mode: Escuro"}[t],
    ) or "Claro"
    mostrar_video = st.segmented_control(
        "Vídeos", ["Mostrar", "Ocultar"], default="Mostrar", width="stretch",
        format_func=lambda v: {"Mostrar": ":material/smart_display: Mostrar", "Ocultar": ":material/hide_source: Ocultar"}[v],
        help="O player do YouTube precisa de internet",
    ) != "Ocultar"
aplicar_tema(tema)
cabecalho()

if modo == "Demo salva":
    arquivos = sorted(Path("resultados/demo").glob("UC*.json"))
    de_para = pd.read_csv("resultados/demo/de_para.csv")
    reais = dict(zip(de_para.video_id, de_para.real, strict=True))
    canal_de = dict(zip(de_para.video_id, de_para.canal, strict=True))
    with st.sidebar:
        escolha = st.selectbox("Canal (fora do treino)", arquivos,
                               format_func=lambda a: canal_de[json.loads(a.read_text())["ranking"][0]["id"]])
        st.caption("Os 3 Shorts mais recentes de canais que o modelo nunca viu, com a data real como data "
                   "planejada. As curtidas reais só aparecem aqui, para o de-para.")
    r = json.loads(escolha.read_text())
    prev = {p["id"]: p for p in r["previsoes"]}
    dentro = sum(prev[i]["interval_low"] <= reais[i] <= prev[i]["interval_high"] for i in prev)
    st.divider()
    st.markdown(f"### {canal_de[r['ranking'][0]['id']]}: qual dos 3 Shorts rende mais?")
    k1, k2, k3 = st.columns(3)
    k1.metric("Criativos comparados", len(prev))
    k2.metric("Real dentro do intervalo de 80%", f"{dentro} de {len(prev)}")
    k3.metric("Mediana recente do canal", f"{fmt(r['previsoes'][0]['relative_to_channel']['mediana_canal'])} curtidas")
    st.plotly_chart(grafico_ranking(r["previsoes"], reais), width="stretch", config={"displayModeBar": False})
    resumo_ranking(r["ranking"])
    st.info(r["explanation"], icon="💬")
    for item in r["ranking"]:
        cartao(prev[item["id"]], item["posicao"], real=reais.get(item["id"]), mostrar_video=mostrar_video)
else:
    c = canais().set_index("channel_id")
    with st.sidebar:
        st.markdown("### Contexto")
        canal = st.selectbox("Canal", c.index, format_func=lambda i: f"{c.titulo[i]} ({c.setor[i]})")
        dia = st.date_input("Data planejada", datetime.now().date(), format="DD/MM/YYYY")
        hora = st.time_input("Hora (Brasília)", time(18, 0))
        analisar = st.segmented_control(
            "Análise do criativo", ["Completa", "Rápida"], default="Completa", width="stretch",
            format_func=lambda a: {"Completa": ":material/movie: Completa", "Rápida": ":material/speed: Rápida"}[a],
            help="Completa: baixa o vídeo, transcreve e chama o VLM (~1 min por vídeo novo). Rápida: só texto e "
                 "imagem de capa, com menos confiança.",
        ) != "Rápida"
    st.divider()
    st.markdown("### Criativos")
    n = st.segmented_control("Quantos criativos comparar", [1, 2, 3, 4, 5], default=2) or 1
    entradas = []
    for i in range(int(n)):
        a, b = st.columns([3, 2])
        criativo = a.text_input(f"Criativo {i + 1}", key=f"c{i}",
                                placeholder="URL do YouTube, caminho de vídeo/imagem ou o texto do criativo")
        legenda = b.text_input("Legenda planejada", key=f"l{i}", placeholder="vazio = a publicada, se for URL")
        if criativo:
            entradas.append({"id": f"criativo_{i + 1}", "creative": criativo, "caption": legenda,
                             "planned_datetime": datetime.combine(dia, hora).isoformat(), "channel_id": canal,
                             "sector": c.setor[canal], "analisar_video": analisar})
    if st.button("Prever", type="primary", disabled=not entradas, width="stretch"):
        from galaxies_tool.agente import no_explicar_ranking, no_ordenar
        from galaxies_tool.contrato import PrevisaoCriativo

        try:
            previsoes = [rodar_criativo(e) for e in entradas]
            ranking, texto = None, None
            if len(previsoes) > 1:
                objs = [PrevisaoCriativo.model_validate(p) for p in previsoes]
                estado = {"previsoes": objs, **no_ordenar({"previsoes": objs})}
                texto = no_explicar_ranking(estado)["explicacao"]
                ranking = [x.model_dump(mode="json") for x in estado["ranking"]]
            st.session_state["resultado"] = (previsoes, ranking, texto, {e["id"]: e["creative"] for e in entradas})
        except Exception as e:  # noqa: BLE001 — erro de entrada ou de rede aparece na tela
            st.error(f"{type(e).__name__}: {e}")
    if res := st.session_state.get("resultado"):
        previsoes, ranking, texto, criativos = res
        prev = {p["id"]: p for p in previsoes}
        st.divider()
        if ranking:
            st.markdown("### Ranking")
            st.plotly_chart(grafico_ranking(previsoes), width="stretch", config={"displayModeBar": False})
            resumo_ranking(ranking)
            st.info(texto, icon="💬")
            ordem = [(x["id"], x["posicao"]) for x in ranking]
        else:
            ordem = [(previsoes[0]["id"], None)]
        for i, pos in ordem:
            cartao(prev[i], pos, criativo=criativos.get(i), mostrar_video=mostrar_video)
