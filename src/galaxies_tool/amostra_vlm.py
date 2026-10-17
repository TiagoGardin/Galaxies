"""Monta a fila de vídeos do VLM (bloco D), em ordem de prioridade, com semente fixa.

Uso: uv run python -m galaxies_tool.amostra_vlm
Saída: data/features/amostra_vlm.txt

A cota gratuita do Gemini não cobre a base inteira (ver docs/EXPERIMENTO.md). Ordem:
1. 2.500 vídeos sorteados do treino (para treinar os modelos com bloco D);
2. 1.000 sorteados da validação (para avaliar durante o desenvolvimento);
3. todo o teste temporal, em ordem aleatória — se a cota acabar antes, a parte coberta continua
   sendo uma amostra aleatória do teste.

Ajuste de 15/10/2026 (antes de qualquer contato com o teste): com ~500 vídeos/dia, a fila acima chegaria
ao prazo sem nenhum vídeo de teste com VLM. A fila do VLM (data/features/fila_vlm.txt) passa a ser a
mesma amostra reordenada: os 1.419 primeiros do treino (os já processados), 400 da validação, todo o
teste e só então o resto. A ordem aleatória dentro de cada split é a mesma. A parte de vídeo do bloco C
continua usando amostra_vlm.txt.
"""

from pathlib import Path

import pandas as pd

FEATURES = Path("data/features/features_ab.parquet")
SAIDA = Path("data/features/amostra_vlm.txt")
SAIDA_VLM = Path("data/features/fila_vlm.txt")
SEMENTE = 42
TAMANHOS = {"treino": 2500, "validacao": 1000, "teste": None}  # None = todos
PRIORIDADE_VLM = {"treino": 1419, "validacao": 400, "teste": None}


def montar(df: pd.DataFrame) -> list[str]:
    fila = []
    for split, n in TAMANHOS.items():
        ids = df.loc[df.split == split, "video_id"].sort_values()
        fila += ids.sample(n=n if n else len(ids), random_state=SEMENTE).tolist()
    return fila


def reordenar(fila: list[str], split: dict[str, str]) -> list[str]:
    """Os primeiros n de cada split, na ordem de PRIORIDADE_VLM, e depois o restante na ordem original."""
    primeiro = []
    for s, n in PRIORIDADE_VLM.items():
        ids = [v for v in fila if split[v] == s]
        primeiro += ids[:n] if n else ids
    vistos = set(primeiro)
    return primeiro + [v for v in fila if v not in vistos]


def main():
    df = pd.read_parquet(FEATURES, columns=["video_id", "split"])
    fila = montar(df)
    assert len(fila) == len(set(fila))
    SAIDA.write_text("\n".join(fila))
    fila_vlm = reordenar(fila, dict(zip(df.video_id, df.split)))
    assert sorted(fila_vlm) == sorted(fila)
    SAIDA_VLM.write_text("\n".join(fila_vlm))
    print(f"{len(fila)} vídeos na fila → {SAIDA} (vídeo) e {SAIDA_VLM} (VLM)")


if __name__ == "__main__":
    main()
