"""Embeddings de texto (bloco B) com paraphrase-multilingual-mpnet-base-v2.

Uso: uv run python -m galaxies_tool.embeddings_texto
Saída: data/features/emb_texto.npy (float16, uma linha por vídeo) e emb_texto_ids.txt (video_id na mesma ordem).

Os embeddings brutos não dependem do split; a redução por PCA é ajustada só no treino, na etapa de modelagem.
"""

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


if __name__ == "__main__":
    main()
