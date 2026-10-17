"""Tools 3 e 4 do agente: previsão (LightGBM + intervalo de 80%) e explicabilidade (SHAP).

O número sai sempre do modelo treinado; o LLM só redige a explicação a partir do que está aqui.

SHAP: valores de TreeSHAP exatos calculados pelo próprio LightGBM (`pred_contrib=True`), na escala do alvo
(log da razão curtidas / mediana do canal). Um SHAP de +0,10 significa ~10,5% mais curtidas (exp(0,10)).
Os 16 componentes de cada embedding são somados num único driver, porque isolados não têm significado.
"""

import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from galaxies_tool import modelo
from galaxies_tool.artefato import preparar
from galaxies_tool.artefato import prever as prever_artefato
from galaxies_tool.metricas import faixa

PASTA = Path("models")
N_DRIVERS = 6
DIAS = ("segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo")

DESCRICOES = {
    "log_mediana_likes_ult20": "Patamar de curtidas recente do canal",
    "std_log_likes_ult20": "Instabilidade das curtidas recentes do canal",
    "tendencia": "Tendência recente do canal (últimos 5 vs. últimos 20 Shorts)",
    "n_shorts_anteriores": "Tamanho do histórico do canal",
    "shorts_ult_30d": "Shorts publicados pelo canal nos 30 dias anteriores",
    "dias_desde_ultimo_post": "Dias desde o último Short do canal",
    "tipo_canal": "Tipo de canal",
    "setor": "Setor",
    "hora": "Hora de publicação",
    "dia_semana": "Dia da semana",
    "fim_de_semana": "Publicação no fim de semana",
    "mes": "Mês de publicação",
    "data_sazonal": "Data sazonal (Black Friday, Dia das Mães, Natal...)",
    "duracao_s": "Duração do vídeo",
    "categoria_id": "Categoria do YouTube",
    "n_tags": "Número de tags",
    "titulo_n_chars": "Tamanho da legenda",
    "titulo_n_palavras": "Palavras na legenda",
    "n_hashtags": "Número de hashtags",
    "tem_hashtag_shorts": "Hashtag #shorts",
    "n_emojis": "Emojis na legenda",
    "titulo_pergunta": "Legenda com pergunta",
    "titulo_exclamacao": "Legenda com exclamação",
    "prop_maiusculas": "Proporção de maiúsculas na legenda",
    "titulo_tem_numero": "Número na legenda",
    "n_mencoes": "Menções (@) na descrição",
    "tem_link": "Link na descrição",
    "tem_cta": "Chamada para ação (comenta, salva, link na bio...)",
    "fmt_grwm": "Formato arrume-se comigo (GRWM)",
    "fmt_tutorial": "Formato tutorial/dica",
    "fmt_haul": "Formato haul/recebidos",
    "fmt_review": "Formato resenha/review",
    "fmt_provador": "Formato provador/look do dia",
    "fmt_antes_depois": "Formato antes e depois",
    "fmt_trend": "Formato trend/desafio",
    "thumb_brilho": "Brilho da imagem",
    "thumb_contraste": "Contraste da imagem",
    "thumb_saturacao": "Saturação da imagem",
    "thumb_colorfulness": "Intensidade de cor da imagem",
    "thumb_n_rostos": "Rostos na imagem",
    "thumb_tem_rosto": "Rosto na imagem",
    "thumb_area_rosto": "Tamanho do rosto na imagem (close-up)",
    "thumb_tem_texto": "Texto escrito na imagem",
    "thumb_n_palavras_ocr": "Palavras escritas na imagem",
    "emb_texto": "Tema e estilo da legenda (embedding de texto)",
    "emb_thumb": "Estilo visual da imagem (embedding SigLIP)",
}
BLOCOS = {
    "A (canal e contexto)": set(modelo.BLOCO_A),
    "B (texto)": set(modelo.BLOCO_B_MANUAL) | {"emb_texto"},
    "C (visual e áudio)": set(modelo.BLOCO_C_THUMB) | {"emb_thumb"},
}


def caminho_padrao() -> Path:
    if env := os.environ.get("GALAXIES_MODELO"):
        return Path(env)
    final = PASTA / "lgbm_final.joblib"
    return final if final.exists() else PASTA / "lgbm_dev.joblib"


@lru_cache(maxsize=4)
def carregar_artefato(caminho: str | None = None) -> dict:
    return joblib.load(caminho or caminho_padrao())


def grupo(coluna: str) -> str:
    """Componentes de PCA viram o embedding de origem (emb_texto_pca_3 → emb_texto)."""
    return coluna.split("_pca_")[0] if "_pca_" in coluna else coluna


def bloco(g: str) -> str:
    return next((b for b, cols in BLOCOS.items() if g in cols), "D (VLM)")


def formatar(g: str, valor) -> str:
    if g.startswith("emb_"):
        return "—"
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "ausente"
    if isinstance(valor, (bool, np.bool_)):
        return "sim" if valor else "não"
    if g == "dia_semana":
        return DIAS[int(valor)]
    if g == "hora":
        return f"{int(valor)}h"
    if g == "log_mediana_likes_ult20":
        return f"{np.expm1(valor):,.0f} curtidas (mediana)".replace(",", ".")
    if g == "duracao_s":
        return f"{valor:.0f} s"
    if isinstance(valor, float):
        return f"{valor:.2f}"
    return str(valor)


def prever_linha(linha: dict, art: dict | None = None) -> dict:
    """Previsão, intervalo, faixa, SHAP agregado e sinal de confiança para um criativo (uma linha)."""
    art = art or carregar_artefato()
    df = pd.DataFrame([linha])
    gerados = {n for p in art["pcas"].values() for n in p["nomes"]}
    brutos = [c for p in art["pcas"].values() for c in p["colunas"]]
    for c in [*art["colunas"], *brutos]:  # atributo ausente (ex.: criativo só texto, sem imagem) = NaN
        if c not in df.columns and c not in gerados:
            df[c] = np.nan
    for c in df.columns:  # None (ex.: duração de criativo só texto) vira NaN numérico
        if df[c].dtype == object and c not in art["categorias"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    x = preparar(df, art)
    p = prever_artefato(art, x).iloc[0]
    log_mediana = float(linha["log_mediana_likes_ult20"])

    contrib = art["ponto"].predict(x, pred_contrib=True)[0]
    shap = pd.Series(contrib[:-1], index=art["colunas"]).groupby(grupo).sum()
    valores = {g: (linha.get(g) if not g.startswith("emb_") else None) for g in shap.index}
    drivers = [
        {
            "atributo": g,
            "descricao": DESCRICOES.get(g, g),
            "valor": formatar(g, valores[g]),
            "efeito": "aumenta" if v > 0 else "reduz",
            "impacto_pct": round(float(np.expm1(v) * 100), 1),
            "bloco": bloco(g),
        }
        for g, v in shap.reindex(shap.abs().sort_values(ascending=False).index).items()
    ]
    por_bloco = shap.groupby(bloco).sum()

    largura = p.alvo_alto - p.alvo_baixo
    lim = art["limiares"]
    motivos = []
    n_hist = linha.get("n_historico_total", linha.get("n_shorts_anteriores", 0))
    if n_hist < lim["min_historico"]:
        motivos.append(
            f"canal com só {n_hist} Shorts maduros no histórico (mínimo confiável: {lim['min_historico']})"
        )
    if linha.get("std_log_likes_ult20", 0) > lim["std_instavel"]:
        motivos.append("curtidas recentes do canal muito instáveis (entre os 10% mais instáveis da base)")
    if largura > lim["largura_larga"]:
        motivos.append("intervalo de previsão largo (entre os 10% mais largos)")
    return {
        "alvo_previsto": float(p.alvo_previsto),
        "alvo_baixo": float(p.alvo_baixo),
        "alvo_alto": float(p.alvo_alto),
        "curtidas_previstas": round(np.expm1(p.alvo_previsto + log_mediana)),
        "curtidas_baixo": round(np.expm1(p.alvo_baixo + log_mediana)),
        "curtidas_alto": round(np.expm1(p.alvo_alto + log_mediana)),
        "mediana_canal": float(np.expm1(log_mediana)),
        "razao_prevista": round(float(np.exp(p.alvo_previsto)), 2),
        "faixa": str(faixa(np.array([p.alvo_previsto]))[0]),
        "drivers": drivers,
        "contribuicao_por_bloco": {b: round(float(np.expm1(v) * 100), 1) for b, v in por_bloco.items()},
        "motivos_baixa_confianca": motivos,
        "modelo_versao": art["versao"],
    }
