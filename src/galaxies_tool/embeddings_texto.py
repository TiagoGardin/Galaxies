"""Embeddings de texto (bloco B) com paraphrase-multilingual-mpnet-base-v2.

Uso: uv run python -m galaxies_tool.embeddings_texto
     uv run python -m galaxies_tool.embeddings_texto <textos.json> <saida.npy>   (lista de textos qualquer; usado
     pelo treino dinâmico, num processo sem o LightGBM)
Saída: data/features/emb_texto.npy (float16, uma linha por vídeo) e emb_texto_ids.txt (video_id na mesma ordem).

Os embeddings brutos não dependem do split; a redução por PCA é ajustada só no treino, na etapa de modelagem.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

FEATURES = Path("data/features/features_ab.parquet")
BASE = Path("data/processed/base_v1.parquet")
SAIDA = Path("data/features/emb_texto.npy")
MODELO = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


def main():
    ids = pd.read_parquet(FEATURES, columns=["video_id"]).video_id
    base = pd.read_parquet(BASE, columns=["video_id", "title", "description"]).set_index("video_id").loc[ids]
    textos = (base.title + "\n" + base.description).tolist()
    disp = "mps" if torch.backends.mps.is_available() else "cpu"
    modelo = SentenceTransformer(MODELO, device=disp)
    emb = modelo.encode(textos, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    np.save(SAIDA, emb.astype(np.float16))
    SAIDA.with_name("emb_texto_ids.txt").write_text("\n".join(ids))
    print(f"{emb.shape} salvos em {SAIDA} (dispositivo: {disp}, máx. de tokens: {modelo.max_seq_length})")


def de_arquivo(entrada: str, saida: str):
    """Embeddings de uma lista de textos (JSON) → .npy em float16, o mesmo arredondamento do treino."""
    textos = json.loads(Path(entrada).read_text())
    emb = SentenceTransformer(MODELO, device="cpu").encode(textos, batch_size=64, normalize_embeddings=True)
    np.save(saida, emb.astype(np.float16))


if __name__ == "__main__":
    if len(sys.argv) == 3:
        de_arquivo(sys.argv[1], sys.argv[2])
    else:
        main()
