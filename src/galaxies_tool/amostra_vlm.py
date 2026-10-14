"""Monta a fila de vídeos do VLM (bloco D), em ordem de prioridade, com semente fixa.

Uso: uv run python -m galaxies_tool.amostra_vlm
Saída: data/features/amostra_vlm.txt

A cota gratuita do Gemini não cobre a base inteira (ver docs/EXPERIMENTO.md). Ordem:
1. 2.500 vídeos sorteados do treino (para treinar os modelos com bloco D);
2. 1.000 sorteados da validação (para avaliar durante o desenvolvimento);
3. todo o teste temporal, em ordem aleatória — se a cota acabar antes, a parte coberta continua
   sendo uma amostra aleatória do teste.
"""

from pathlib import Path

import pandas as pd

FEATURES = Path("data/features/features_ab.parquet")
SAIDA = Path("data/features/amostra_vlm.txt")
SEMENTE = 42
TAMANHOS = {"treino": 2500, "validacao": 1000, "teste": None}  # None = todos


def montar(df: pd.DataFrame) -> list[str]:
    fila = []
    for split, n in TAMANHOS.items():
        ids = df.loc[df.split == split, "video_id"].sort_values()
        fila += ids.sample(n=n if n else len(ids), random_state=SEMENTE).tolist()
    return fila


def main():
    fila = montar(pd.read_parquet(FEATURES, columns=["video_id", "split"]))
    assert len(fila) == len(set(fila))
    SAIDA.write_text("\n".join(fila))
    print(f"{len(fila)} vídeos na fila → {SAIDA}")


if __name__ == "__main__":
    main()
