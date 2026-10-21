"""Tools 3 e 4 do agente: previsão (LightGBM + intervalo de 80% calibrado) e explicabilidade (SHAP).

O número sai sempre do modelo treinado; o LLM só redige a explicação a partir do que está aqui.

SHAP: o ponto é a média de LightGBM e CatBoost, e o SHAP é a média dos TreeSHAP exatos de cada um
(`pred_contrib=True` e `ShapValues`), aditivo como os dois, na escala do alvo
(log da razão curtidas / mediana do canal). Um SHAP de +0,10 significa ~10,5% mais curtidas (exp(0,10)).
Os componentes de cada embedding são somados num único driver, porque isolados não têm significado.
"""

import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from galaxies_tool import modelo, multimodal, rag
from galaxies_tool.artefato import contribuicoes, distancia_ood, motivos, preparar
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
    "fmt_grwm": "Formato arrume-se comigo (GRWM) declarado no texto",
    "fmt_tutorial": "Formato tutorial/dica declarado no texto",
    "fmt_haul": "Formato haul/recebidos declarado no texto",
    "fmt_review": "Formato resenha/review declarado no texto",
    "fmt_provador": "Formato provador/look do dia declarado no texto",
    "fmt_antes_depois": "Formato antes e depois declarado no texto",
    "fmt_trend": "Formato trend/desafio declarado no texto",
    "thumb_brilho": "Brilho da imagem de capa",
    "thumb_contraste": "Contraste da imagem de capa",
    "thumb_saturacao": "Saturação da imagem de capa",
    "thumb_colorfulness": "Intensidade de cor da imagem de capa",
    "thumb_n_rostos": "Rostos na imagem de capa",
    "thumb_tem_rosto": "Rosto na imagem de capa",
    "thumb_area_rosto": "Tamanho do rosto na capa (close-up)",
    "thumb_tem_texto": "Texto escrito na imagem de capa",
    "thumb_n_palavras_ocr": "Palavras escritas na imagem de capa",
    "emb_texto": "Tema e estilo da legenda",
    "emb_thumb": "Estilo visual da imagem de capa",
    "emb_gancho": "Visual dos primeiros 3 segundos (gancho)",
    "cortes_por_segundo": "Ritmo de cortes de cena",
    "movimento_medio": "Movimento da imagem",
    "tem_voz": "Voz no áudio (fala ou canto)",
    "voz_por_segundo": "Densidade de fala",
    "energia_rms": "Energia (volume) do áudio",
    "vlm_tem_rosto": "Rosto no vídeo",
    "vlm_close_up_produto": "Close-up de produto",
    "vlm_before_after_ou_transformacao": "Antes/depois ou transformação",
    "vlm_try_on_ou_demonstracao": "Prova de roupa ou demonstração de uso",
    "vlm_swatch": "Swatch (amostra de cor/textura)",
    "vlm_aplicacao_na_pele": "Aplicação de produto na pele/cabelo/unhas",
    "vlm_texto_na_tela": "Texto na tela durante o vídeo",
    "vlm_tem_musica_de_fundo": "Música de fundo",
    "vlm_forca_hook_0a10": "Força do gancho nos 3 primeiros segundos (nota 0–10)",
    "vlm_densidade_texto_0a10": "Quantidade de texto na tela (nota 0–10)",
    "vlm_ritmo_cortes_0a10": "Ritmo de edição percebido (nota 0–10)",
    "vlm_estetica_geral_0a10": "Qualidade estética e de produção (nota 0–10)",
    "vlm_clareza_beneficio_produto_0a10": "Clareza do benefício do produto (nota 0–10)",
    "vlm_corpo_inteiro_vs_detalhe": "Look de corpo inteiro ou detalhe",
    "vlm_audio_original_ou_musica": "Tipo de áudio predominante",
    "vlm_tom_narracao": "Tom da narração",
    "vlm_formato": "Formato real do vídeo",
    **rag.DESCRICOES,
}
BLOCOS = {
    "A (canal e contexto)": set(modelo.BLOCO_A),
    "B (texto)": set(modelo.BLOCO_B_MANUAL) | {"emb_texto"},
    "C (imagem de capa)": set(modelo.BLOCO_C_THUMB) | {"emb_thumb"},
    "C (vídeo e áudio)": set(modelo.BLOCO_C_VIDEO) | {"emb_gancho"},
    "D (VLM)": set(modelo.BLOCO_D),
    "R (comparáveis)": set(rag.COLUNAS),
}
BINARIAS = set(multimodal.COLUNAS_VLM_BINARIAS) | {"tem_voz"}


V1, V2 = PASTA / "lgbm_final.joblib", PASTA / "lgbm_v2.joblib"


def caminho_padrao() -> Path:
    """Modelo servido: a v2 (escolhida pela regra pré-registrada no teste prospectivo de 21/10/2026; ver
    docs/EXPERIMENTO.md). Ela precisa dos embeddings da base para as features de comparáveis; sem eles (não
    vão para o git), cai para a v1, a avaliada no teste temporal. GALAXIES_MODELO escolhe outro artefato."""
    if env := os.environ.get("GALAXIES_MODELO"):
        return Path(env)
    if V2.exists() and modelo.EMB_TEXTO.exists() and modelo.EMB_THUMB.exists():
        return V2
    return V1


def aviso_modelo() -> str | None:
    """Frase para a resposta quando a tool caiu para a v1 por falta dos embeddings da base (None se não caiu)."""
    if os.environ.get("GALAXIES_MODELO") or not V2.exists() or caminho_padrao() == V2:
        return None
    return ("modelo v2 indisponível nesta máquina (faltam data/features/emb_texto.npy e emb_thumb.npy, "
            "não versionados): a previsão usa a v1 (lgbm_final), a avaliada no teste temporal")


@lru_cache(maxsize=4)
def carregar_artefato(caminho: str | None = None) -> dict:
    return joblib.load(caminho or caminho_padrao())


def artefato_da_entrada(modelo_cliente: str | None) -> dict:
    """Modelo principal ou, com `modelo_cliente`, o especialista de models/clientes/<nome>.joblib."""
    if not modelo_cliente:
        return carregar_artefato(str(caminho_padrao()))
    from galaxies_tool.treino_cliente import caminho

    arq = caminho(modelo_cliente)
    if not arq.exists():
        raise ValueError(f"modelo '{modelo_cliente}' não encontrado: treine com treinar_modelo_cliente")
    return carregar_artefato(str(arq))


PARTES = {  # parte do criativo (como em extracao._faltando) → colunas que dependem dela
    "imagem": tuple(modelo.BLOCO_C_THUMB) + ("emb_thumb_pca",),
    "vídeo": tuple(modelo.BLOCO_C_VIDEO) + ("emb_gancho_pca",),
    "rubrica do VLM": ("vlm_",),
}


def partes_usadas(art: dict) -> set[str]:
    """Partes do criativo que o modelo usa (o principal usa todas; um modelo de cliente, só algumas)."""
    return {p for p, prefixos in PARTES.items() if any(c.startswith(prefixos) for c in art["colunas"])}


def grupo(coluna: str) -> str:
    """Componentes de PCA viram o embedding de origem (emb_texto_pca_3 → emb_texto)."""
    return coluna.split("_pca_")[0] if "_pca_" in coluna else coluna


def bloco(g: str) -> str:
    return next(b for b, cols in BLOCOS.items() if g in cols)


def formatar(g: str, valor) -> str:
    if g.startswith("emb_"):
        return "—"
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "ausente"
    if isinstance(valor, (bool, np.bool_)) or g in BINARIAS:
        return "sim" if valor else "não"
    if g == "dia_semana":
        return DIAS[int(valor)]
    if g == "hora":
        return f"{int(valor)}h"
    if g == "log_mediana_likes_ult20":
        return f"{np.expm1(valor):,.0f} curtidas (mediana)".replace(",", ".")
    if g == "duracao_s":
        return f"{valor:.0f} s"
    if g in ("rag_canal_alvo", "rag_canal_recente", "rag_setor_alvo"):  # log da razão → vezes a mediana
        return f"{np.exp(valor):.2f}× a mediana do canal".replace(".", ",")
    if g.endswith("_0a10"):
        return f"{valor:.0f}"
    if isinstance(valor, float):
        return f"{valor:.2f}".replace(".", ",")
    return str(valor).replace("_", " ")


def prever_linha(linha: dict, art: dict | None = None, extracao_incompleta: list[str] | None = None) -> dict:
    """Previsão, intervalo, faixa, SHAP agregado e sinal de confiança para um criativo (uma linha).

    `extracao_incompleta`: partes do criativo que não puderam ser analisadas (vira um motivo do flag)."""
    art = art or carregar_artefato()
    gerados = {n for p in art["pcas"].values() for n in p["nomes"]}
    brutos = [c for p in art["pcas"].values() for c in p["colunas"]]
    # atributo ausente (ex.: criativo só texto, sem imagem) = NaN; todas as colunas de uma vez
    faltam = [c for c in dict.fromkeys([*art["colunas"], *brutos]) if c not in linha and c not in gerados]
    df = pd.DataFrame([{**linha, **dict.fromkeys(faltam, np.nan)}])
    for c in df.columns:  # None (ex.: duração de criativo só texto) vira NaN numérico
        if df[c].dtype == object and c not in art["categorias"] and c != "published_at":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    x = preparar(df, art)
    p = prever_artefato(art, x).iloc[0]
    log_mediana = float(linha["log_mediana_likes_ult20"])

    contrib = contribuicoes(art, x)[0]
    # aditividade do TreeSHAP: valor base + soma das contribuições = previsão (aula 5.6)
    assert abs(contrib.sum() - p.alvo_previsto) < 1e-6, "SHAP não fecha com a previsão"
    shap = pd.Series(contrib[:-1], index=art["colunas"]).groupby(grupo).sum()
    drivers = [
        {
            "atributo": g,
            "descricao": DESCRICOES.get(g, g),
            "valor": formatar(g, None if g.startswith("emb_") else linha.get(g)),
            "efeito": "aumenta" if v > 0 else "reduz",
            "impacto_pct": round(float(np.expm1(v) * 100), 1),
            "bloco": bloco(g),
        }
        for g, v in shap.reindex(shap.abs().sort_values(ascending=False).index).items()
        if abs(v) > 0
    ]
    por_bloco = shap.groupby(bloco).sum()
    largura = float(p.alvo_alto - p.alvo_baixo)
    razoes = motivos(art, linha, largura, float(distancia_ood(art, x)[0]), extracao_incompleta or [])
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
        "motivos_baixa_confianca": razoes,
        "modelo_versao": art["versao"],
    }
