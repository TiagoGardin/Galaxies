"""Tool 1 do agente: extrai os atributos de um criativo novo com as MESMAS funções usadas no treino.

O criativo pode ser:
- URL do YouTube: usa a thumbnail do vídeo (a mesma que o treino usou) e, se a legenda não for informada,
  título, descrição, tags e duração do vídeo (via base coletada ou YouTube Data API). Nunca usa curtidas,
  views ou comentários do vídeo (regra 1 de docs/FEATURES.md);
- vídeo local: o quadro em 1 s faz o papel da thumbnail, e a duração vem do arquivo;
- imagem local: a própria imagem é a thumbnail;
- só texto: sem atributos visuais (o modelo trata como ausentes, e a confiança cai).

Saída: um dicionário de atributos com os mesmos nomes de coluna do treino (embeddings brutos inclusos;
o PCA é aplicado depois, pelo artefato do modelo).

Use `extrair_isolado` em qualquer processo que também use o LightGBM: no macOS, o PyTorch (SigLIP e
embeddings de texto) e o LightGBM trazem cópias diferentes do OpenMP e derrubam o processo se convivem
(segmentation fault). A extração roda então num processo próprio, que fica vivo com os modelos carregados.
Por isso este módulo nunca importa o LightGBM.
"""

import atexit
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
from PIL import Image

from galaxies_tool import thumbs
from galaxies_tool.features import features_texto

BASE = Path("data/processed/base_v1.parquet")
RX_YOUTUBE = re.compile(r"(?:youtube\.com/(?:shorts/|watch\?v=|embed/|live/)|youtu\.be/)([\w-]{11})")
EXT_IMAGEM = {".jpg", ".jpeg", ".png", ".webp"}
EXT_VIDEO = {".mp4", ".mov", ".m4v", ".webm", ".mkv"}
INSTANTE_QUADRO_S = 1.0
QUALIDADE_JPEG = 90  # as thumbnails do treino foram salvas assim


def tipo_do_criativo(creative: str) -> tuple[str, str | None]:
    """("youtube", video_id), ("imagem", caminho), ("video", caminho) ou ("texto", None)."""
    if m := RX_YOUTUBE.search(creative):
        return "youtube", m.group(1)
    caminho = Path(creative).expanduser()
    if len(creative) < 1024 and caminho.suffix.lower() in EXT_IMAGEM | EXT_VIDEO:
        if not caminho.exists():
            raise FileNotFoundError(f"criativo não encontrado: {caminho}")
        return ("imagem" if caminho.suffix.lower() in EXT_IMAGEM else "video"), str(caminho)
    return "texto", None


# ---------------------------------------------------------------- quadro (thumbnail)


def como_no_treino(im: Image.Image, recortar: bool) -> Image.Image:
    """Mesmo tratamento das thumbnails do treino: quadro vertical central, 288x512 e JPEG de qualidade 90."""
    im = im.convert("RGB")
    if recortar or im.width > im.height:
        im = thumbs.recortar_quadro(im)
    buf = io.BytesIO()
    im.resize(thumbs.TAMANHO).save(buf, format="JPEG", quality=QUALIDADE_JPEG)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def quadro_do_youtube(video_id: str) -> Image.Image | None:
    em_cache = thumbs.PASTA / f"{video_id}.jpg"
    if em_cache.exists():
        return Image.open(em_cache).convert("RGB")
    for nome in ("maxresdefault", "hqdefault"):
        r = requests.get(f"https://i.ytimg.com/vi/{video_id}/{nome}.jpg", timeout=30)
        if r.ok and len(r.content) > 2000:  # o YouTube devolve um placeholder minúsculo quando não há maxres
            return como_no_treino(Image.open(io.BytesIO(r.content)), recortar=True)
    return None


def quadro_e_duracao_do_video(caminho: str) -> tuple[Image.Image | None, float | None]:
    cap = cv2.VideoCapture(caminho)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(round(INSTANTE_QUADRO_S * fps), max(n - 1, 0)))
    ok, bgr = cap.read()
    cap.release()
    quadro = (
        como_no_treino(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), recortar=False) if ok else None
    )
    return quadro, (n / fps if n else None)


# ---------------------------------------------------------------- modelos (carregados uma vez)


@lru_cache(maxsize=1)
def _detector():
    return thumbs.detector_de_rosto()


@lru_cache(maxsize=1)
def _leitor_ocr():
    import easyocr

    return easyocr.Reader(["pt", "en"], gpu=False, verbose=False)


@lru_cache(maxsize=1)
def _siglip():
    import torch
    from transformers import AutoModel, AutoProcessor

    disp = "mps" if torch.backends.mps.is_available() else "cpu"
    return (
        AutoModel.from_pretrained(thumbs.MODELO_SIGLIP).to(disp).eval(),
        AutoProcessor.from_pretrained(thumbs.MODELO_SIGLIP),
        disp,
    )


@lru_cache(maxsize=1)
def _modelo_texto():
    from sentence_transformers import SentenceTransformer

    from galaxies_tool.embeddings_texto import MODELO  # importa o PyTorch: só dentro do subprocesso

    return SentenceTransformer(MODELO, device="cpu")


@lru_cache(maxsize=1)
def _base() -> pd.DataFrame:
    return pd.read_parquet(
        BASE, columns=["video_id", "title", "description", "tags", "duration_s", "category_id"]
    ).set_index("video_id")


# ---------------------------------------------------------------- atributos


def atributos_imagem(quadro: Image.Image) -> dict:
    """Cor, rosto e OCR (bloco C) e o embedding SigLIP 2 bruto (embimg_*), como em thumbs.py."""
    import torch

    rgb = np.asarray(quadro)
    # o treino passou o caminho do arquivo: com array ou bytes o EasyOCR decodifica de outro jeito e lê
    # outras palavras. Um PNG temporário (sem perda) reproduz o treino.
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        quadro.save(tmp.name)
        palavras = [t for _, t, conf in _leitor_ocr().readtext(tmp.name, detail=1) if conf >= 0.25]
    modelo, proc, disp = _siglip()
    with torch.no_grad():
        e = modelo.get_image_features(**proc(images=[quadro], return_tensors="pt").to(disp))
        e = getattr(e, "pooler_output", e)
        emb = torch.nn.functional.normalize(e, dim=-1).cpu().numpy()[0].astype(np.float16).astype(np.float32)
    return {
        **thumbs.estatisticas_cor(rgb),
        **thumbs.rostos(_detector(), rgb),
        "thumb_tem_texto": bool(palavras),
        "thumb_n_palavras_ocr": sum(len(t.split()) for t in palavras),
        **{f"embimg_{i}": float(v) for i, v in enumerate(emb)},
    }


def atributos_texto(titulo: str, descricao: str, tags: list[str]) -> dict:
    """Flags e contagens do bloco B e o embedding de texto bruto (embtxt_*), como no treino."""
    emb = _modelo_texto().encode([f"{titulo}\n{descricao}"], normalize_embeddings=True)[0]
    emb = emb.astype(np.float16).astype(np.float32)
    return {
        **features_texto(titulo, descricao, tags),
        "n_tags": len(tags),
        **{f"embtxt_{i}": float(v) for i, v in enumerate(emb)},
    }


def metadados_youtube(video_id: str) -> dict | None:
    """Título, descrição, tags, duração e categoria — só o que existe antes da publicação."""
    base = _base()
    if video_id in base.index:
        r = base.loc[video_id]
        return {
            "titulo": r.title,
            "descricao": r.description,
            "tags": list(r.tags) if r.tags is not None else [],
            "duracao_s": float(r.duration_s),
            "categoria_id": r.category_id,
        }
    try:
        from galaxies_tool.youtube_api import YouTube, duracao_segundos

        itens = YouTube().videos([video_id])
    except Exception:  # noqa: BLE001 — sem chave ou sem rede: segue só com o que o usuário informou
        return None
    if not itens:
        return None
    s, cd = itens[0]["snippet"], itens[0]["contentDetails"]
    return {
        "titulo": s.get("title", ""),
        "descricao": s.get("description", ""),
        "tags": s.get("tags", []),
        "duracao_s": float(duracao_segundos(cd.get("duration", ""))),
        "categoria_id": s.get("categoryId"),
    }


def extrair(creative: str, caption: str = "", description: str = "", tags: list[str] | None = None) -> dict:
    """Atributos do criativo com os nomes de coluna do treino, mais `_tipo` e `_avisos` (para inspeção)."""
    tipo, ref = tipo_do_criativo(creative)
    tags = list(tags or [])
    titulo, descricao, duracao, categoria, quadro, avisos = caption, description, None, None, None, []
    if tipo == "youtube":
        meta = metadados_youtube(ref)
        if meta:
            if not caption:  # legenda não informada: usa a publicada (o YouTube Short já existe)
                titulo, descricao, tags = meta["titulo"], meta["descricao"], tags or meta["tags"]
            duracao, categoria = meta["duracao_s"], meta["categoria_id"]
        quadro = quadro_do_youtube(ref)
    elif tipo == "imagem":
        quadro = como_no_treino(Image.open(ref), recortar=False)
    elif tipo == "video":
        quadro, duracao = quadro_e_duracao_do_video(ref)
    else:
        titulo = titulo or creative  # criativo só em texto: o próprio texto é a legenda
    if quadro is None:
        avisos.append("sem imagem do criativo: atributos visuais ausentes")
    if not titulo:
        avisos.append("sem legenda: atributos de texto calculados sobre texto vazio")
    atr = {"duracao_s": duracao, "categoria_id": categoria, **atributos_texto(titulo, descricao, tags)}
    if quadro is not None:
        atr.update(atributos_imagem(quadro))
    atr["_tipo"], atr["_ref"], atr["_titulo"], atr["_avisos"] = tipo, ref, titulo, avisos
    return atr


# ---------------------------------------------------------------- processo isolado


def _para_json(v):
    if isinstance(v, np.generic):
        return v.item()
    raise TypeError(type(v))


def servidor():
    """Lê pedidos JSON (um por linha) no stdin e responde os atributos em JSON, um por linha.
    Tudo o que as bibliotecas imprimirem vai para o stderr, para não misturar com as respostas."""
    resposta = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    for linha in sys.stdin:
        try:
            r = {"ok": extrair(**json.loads(linha))}
        except Exception as e:  # noqa: BLE001 — o erro volta para quem chamou
            r = {"erro": f"{type(e).__name__}: {e}"}
        resposta.write(json.dumps(r, default=_para_json) + "\n")
        resposta.flush()


_processo: subprocess.Popen | None = None


def extrair_isolado(
    creative: str, caption: str = "", description: str = "", tags: list[str] | None = None
) -> dict:
    """`extrair` num subprocesso que fica vivo entre chamadas (ver docstring do módulo).
    A 1ª chamada carrega os modelos (~30 s); as seguintes levam poucos segundos."""
    global _processo
    if _processo is None or _processo.poll() is not None:
        _processo = subprocess.Popen(
            [sys.executable, "-m", "galaxies_tool.extracao"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        atexit.register(_processo.terminate)
    pedido = {"creative": creative, "caption": caption, "description": description, "tags": list(tags or [])}
    _processo.stdin.write(json.dumps(pedido) + "\n")
    _processo.stdin.flush()
    while True:  # ignora o que alguma biblioteca tenha impresso antes do redirecionamento para o stderr
        linha = _processo.stdout.readline()
        if not linha:
            raise RuntimeError("o processo de extração terminou inesperadamente (veja o stderr)")
        if linha.startswith(('{"ok"', '{"erro"')):
            break
    r = json.loads(linha)
    if "erro" in r:
        raise RuntimeError(f"falha na extração: {r['erro']}")
    return r["ok"]


if __name__ == "__main__":
    servidor()
