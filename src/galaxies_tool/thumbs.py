"""Bloco C, parte da thumbnail (base inteira, sem baixar vídeo).

Uso:
  uv run python -m galaxies_tool.thumbs baixar      # thumbnails recortadas → data/media/thumbs/<id>.jpg
  uv run python -m galaxies_tool.thumbs estatisticas # cor, brilho, rosto → data/features/thumbs_cv.parquet
  uv run python -m galaxies_tool.thumbs ocr          # texto na thumbnail → data/features/thumbs_ocr.parquet
  uv run python -m galaxies_tool.thumbs siglip       # embeddings SigLIP 2 → data/features/emb_thumb.npy
  uv run python -m galaxies_tool.thumbs siglip_gancho  # SigLIP 2 dos quadros do gancho (video.py) → emb_gancho.npy

A thumbnail de um Short vem como 16:9 (1280x720) com o quadro vertical no centro e as laterais preenchidas
por uma cópia ampliada e escurecida. Todas as medidas usam só o quadro central 9:16 (`recortar_quadro`).
Cada etapa tem cache e pode ser retomada.
"""

import io
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
from PIL import Image

FEATURES = Path("data/features/features_ab.parquet")
BASE = Path("data/processed/base_v1.parquet")
PASTA = Path("data/media/thumbs")
SAIDA_CV = Path("data/features/thumbs_cv.parquet")
SAIDA_OCR = Path("data/features/thumbs_ocr.parquet")
SAIDA_EMB = Path("data/features/emb_thumb.npy")
SAIDA_EMB_GANCHO = Path("data/features/emb_gancho.npy")
MODELO_ROSTO = Path("data/media/modelos/blaze_face_short_range.tflite")
URL_MODELO_ROSTO = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
                    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite")
MODELO_SIGLIP = "google/siglip2-base-patch16-224"
TAMANHO = (288, 512)  # largura x altura do quadro salvo


def ids_alvo() -> list[str]:
    return pd.read_parquet(FEATURES, columns=["video_id"]).video_id.tolist()


def recortar_quadro(im: Image.Image) -> Image.Image:
    """Recorta o quadro vertical 9:16 central. Thumbnails 4:3 (sd/hq) têm o 16:9 com faixas pretas
    em cima e embaixo; primeiro se isola a área 16:9 e depois o quadro central."""
    w, h = im.size
    altura_util = w * 9 / 16 if h / w > 0.6 else h
    topo = (h - altura_util) / 2
    largura_quadro = altura_util * 9 / 16
    esquerda = (w - largura_quadro) / 2
    x0, y0 = round(esquerda), round(topo)  # borda arredondada + tamanho arredondado: largura exata
    return im.crop((x0, y0, x0 + round(largura_quadro), y0 + round(altura_util)))


def _baixar_um(item: tuple[str, str]) -> str:
    vid, url = item
    destino = PASTA / f"{vid}.jpg"
    if destino.exists():
        return "cache"
    for _ in range(3):
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            quadro = recortar_quadro(Image.open(io.BytesIO(r.content)).convert("RGB")).resize(TAMANHO)
            quadro.save(destino, quality=90)
            return "ok"
        except Exception:  # noqa: BLE001 — rede ou imagem corrompida: espera e tenta de novo
            time.sleep(2)
    return "falha"


def baixar():
    PASTA.mkdir(parents=True, exist_ok=True)
    urls = pd.read_parquet(BASE, columns=["video_id", "thumbnail_url"]).set_index("video_id").thumbnail_url
    itens = [(v, urls[v]) for v in ids_alvo()]
    with ThreadPoolExecutor(8) as ex:
        res = pd.Series(list(ex.map(_baixar_um, itens)))
    print(res.value_counts().to_string())


# ---------------------------------------------------------------- estatísticas de imagem e rosto


def estatisticas_cor(rgb: np.ndarray) -> dict:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    cinza = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    r, g, b = (rgb[..., i].astype(float) for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    colorfulness = np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())  # Hasler-Süsstrunk
    return {
        "thumb_brilho": hsv[..., 2].mean() / 255,
        "thumb_contraste": cinza.std() / 255,
        "thumb_saturacao": hsv[..., 1].mean() / 255,
        "thumb_colorfulness": colorfulness / 255,
    }


def detector_de_rosto():
    from mediapipe.tasks.python import BaseOptions, vision
    if not MODELO_ROSTO.exists():
        MODELO_ROSTO.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL_MODELO_ROSTO, MODELO_ROSTO)
    opcoes = vision.FaceDetectorOptions(base_options=BaseOptions(model_asset_path=str(MODELO_ROSTO)),
                                        min_detection_confidence=0.5)
    return vision.FaceDetector.create_from_options(opcoes)


def rostos(detector, rgb: np.ndarray) -> dict:
    import mediapipe as mp
    det = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    areas = [d.bounding_box.width * d.bounding_box.height for d in det.detections]
    return {"thumb_n_rostos": len(areas), "thumb_tem_rosto": bool(areas),
            "thumb_area_rosto": max(areas) / (rgb.shape[0] * rgb.shape[1]) if areas else 0.0}


def estatisticas():
    detector = detector_de_rosto()
    linhas = []
    for i, vid in enumerate(ids_alvo(), 1):
        arq = PASTA / f"{vid}.jpg"
        if not arq.exists():
            continue
        rgb = np.asarray(Image.open(arq).convert("RGB"))
        linhas.append({"video_id": vid, **estatisticas_cor(rgb), **rostos(detector, rgb)})
        if i % 5000 == 0:
            print(f"{i} thumbnails", flush=True)
    pd.DataFrame(linhas).to_parquet(SAIDA_CV, index=False)
    print(f"{len(linhas)} linhas → {SAIDA_CV}")


# ---------------------------------------------------------------- OCR


def ocr():
    import easyocr
    leitor = easyocr.Reader(["pt", "en"], gpu=False, verbose=False)
    feitos = pd.read_parquet(SAIDA_OCR) if SAIDA_OCR.exists() else pd.DataFrame(columns=["video_id"])
    ja = set(feitos.video_id)
    novas = []
    pendentes = [v for v in ids_alvo() if v not in ja and (PASTA / f"{v}.jpg").exists()]
    print(f"{len(pendentes)} thumbnails pendentes", flush=True)
    for i, vid in enumerate(pendentes, 1):
        textos = leitor.readtext(str(PASTA / f"{vid}.jpg"), detail=1)
        palavras = [t for _, t, conf in textos if conf >= 0.25]  # texto estilizado vem com confiança baixa
        novas.append({"video_id": vid, "thumb_tem_texto": bool(palavras),
                      "thumb_n_palavras_ocr": sum(len(t.split()) for t in palavras)})
        if i % 500 == 0 or i == len(pendentes):  # salva aos poucos para poder retomar
            feitos = pd.concat([feitos, pd.DataFrame(novas)], ignore_index=True)
            feitos.to_parquet(SAIDA_OCR, index=False)
            novas = []
            print(f"{i}/{len(pendentes)}", flush=True)


# ---------------------------------------------------------------- SigLIP 2


def carregar_siglip():
    import torch
    from transformers import AutoModel, AutoProcessor
    disp = "mps" if torch.backends.mps.is_available() else "cpu"
    return AutoModel.from_pretrained(MODELO_SIGLIP).to(disp).eval(), AutoProcessor.from_pretrained(MODELO_SIGLIP), disp


def embeddings_siglip(siglip, imagens: list[Image.Image]) -> np.ndarray:
    """Embeddings SigLIP 2 normalizados (norma 1), em float16 como no cache do treino."""
    import torch
    modelo, proc, disp = siglip
    with torch.no_grad():
        e = modelo.get_image_features(**proc(images=imagens, return_tensors="pt").to(disp))
        e = getattr(e, "pooler_output", e)
        return torch.nn.functional.normalize(e, dim=-1).cpu().numpy().astype(np.float16)


def media_normalizada(embs: np.ndarray) -> np.ndarray:
    """Média dos embeddings dos quadros do gancho, renormalizada (mesma escala do embedding de um quadro)."""
    m = embs.astype(np.float32).mean(axis=0)
    return (m / np.linalg.norm(m)).astype(np.float16)


def siglip():
    modelo = carregar_siglip()
    ids = [v for v in ids_alvo() if (PASTA / f"{v}.jpg").exists()]
    embs = []
    for i in range(0, len(ids), 64):
        embs.append(embeddings_siglip(modelo, [Image.open(PASTA / f"{v}.jpg").convert("RGB")
                                               for v in ids[i:i + 64]]))
        if (i // 64) % 50 == 0:
            print(f"{i}/{len(ids)}", flush=True)
    np.save(SAIDA_EMB, np.concatenate(embs))
    SAIDA_EMB.with_name("emb_thumb_ids.txt").write_text("\n".join(ids))
    print(f"{len(ids)} embeddings → {SAIDA_EMB}")


def siglip_gancho():
    """Embedding do gancho: média dos quadros de 0,5 s, 1,5 s e 2,5 s salvos por video.py (bloco C)."""
    from galaxies_tool.video import PASTA_GANCHO, quadros_do_gancho
    modelo = carregar_siglip()
    ids = sorted({a.name.rsplit("_", 1)[0] for a in PASTA_GANCHO.glob("*.jpg")})
    embs = []
    for i, vid in enumerate(ids):
        embs.append(media_normalizada(embeddings_siglip(modelo, quadros_do_gancho(vid))))
        if i % 1000 == 0:
            print(f"{i}/{len(ids)}", flush=True)
    np.save(SAIDA_EMB_GANCHO, np.stack(embs))
    SAIDA_EMB_GANCHO.with_name("emb_gancho_ids.txt").write_text("\n".join(ids))
    print(f"{len(ids)} embeddings → {SAIDA_EMB_GANCHO}")


if __name__ == "__main__":
    {"baixar": baixar, "estatisticas": estatisticas, "ocr": ocr, "siglip": siglip,
     "siglip_gancho": siglip_gancho}[sys.argv[1]]()
