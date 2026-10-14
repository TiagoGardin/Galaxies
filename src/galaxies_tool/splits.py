"""Define os splits do experimento a partir da base processada (protocolo em docs/EXPERIMENTO.md).

Uso: uv run python -m galaxies_tool.splits
Saída: data/processed/splits_v1.json

- Canais de demonstração ficam fora de tudo.
- 20% dos demais canais (estratificado por setor x tipo, semente fixa) formam o teste em canais novos.
- Os vídeos elegíveis dos canais restantes são cortados no tempo em ~70/15/15 (cortes em dia inteiro, UTC).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("data/processed/base_v1.parquet")
SAIDA = Path("data/processed/splits_v1.json")
FRACAO_CANAIS_NOVOS = 0.20
QUANTIS_CORTE = (0.70, 0.85)
SEMENTE = 42


def sortear_canais_novos(canais: pd.DataFrame, semente: int = SEMENTE) -> list[str]:
    rng = np.random.default_rng(semente)
    escolhidos = []
    for _, g in canais.sort_values("channel_id").groupby(["sector", "channel_type"]):
        n = round(FRACAO_CANAIS_NOVOS * len(g))
        escolhidos += list(rng.choice(g.channel_id.to_numpy(), size=n, replace=False))
    return sorted(escolhidos)


def atribuir_split(df: pd.DataFrame, splits: dict) -> np.ndarray:
    """Rótulo de cada vídeo: treino, validacao, teste, teste_canais_novos, fora (canal novo antes
    do corte de teste) ou demo. Usada aqui e na montagem das features, para não haver divergência."""
    c1 = pd.Timestamp(splits["corte_treino_validacao"], tz="UTC")
    c2 = pd.Timestamp(splits["corte_validacao_teste"], tz="UTC")
    pub, eh_novo = df.published_at, df.channel_id.isin(splits["canais_novos"])
    rotulo = np.select([pub < c1, pub < c2], ["treino", "validacao"], "teste")
    rotulo = np.where(eh_novo, np.where(rotulo == "teste", "teste_canais_novos", "fora"), rotulo)
    if "is_demo_channel" in df:
        rotulo = np.where(df.is_demo_channel, "demo", rotulo)
    return rotulo


def definir_splits(base: pd.DataFrame) -> dict:
    eleg = base[base.elegivel_treino & ~base.is_demo_channel]
    canais = eleg.groupby("channel_id")[["sector", "channel_type"]].first().reset_index()
    novos = set(sortear_canais_novos(canais))

    vistos = eleg[~eleg.channel_id.isin(novos)]
    cortes = [vistos.published_at.quantile(q).normalize() for q in QUANTIS_CORTE]
    eleg = eleg.assign(split=atribuir_split(eleg, {"corte_treino_validacao": str(cortes[0].date()),
                                                   "corte_validacao_teste": str(cortes[1].date()),
                                                   "canais_novos": sorted(novos)}))

    contagem = eleg.split.value_counts().to_dict()
    treino = eleg[eleg.split == "treino"]
    meses_treino = (treino.published_at.max() - treino.published_at.min()).days / 30.44
    return {
        "semente": SEMENTE,
        "corte_treino_validacao": str(cortes[0].date()),
        "corte_validacao_teste": str(cortes[1].date()),
        "meses_cobertos_pelo_treino": round(meses_treino, 1),
        "usar_feature_mes": bool(meses_treino >= 12),
        "canais_demo": sorted(base[base.is_demo_channel].channel_id.unique()),
        "canais_novos": sorted(novos),
        "canais_novos_por_estrato": {f"{s}|{t}": int(n) for (s, t), n in
                                     canais[canais.channel_id.isin(novos)].groupby(
                                         ["sector", "channel_type"]).size().items()},
        "videos_por_split": {k: int(v) for k, v in contagem.items()},
        "videos_por_split_e_tipo": {f"{s}|{t}": int(n) for (s, t), n in
                                    eleg.groupby(["split", "channel_type"]).size().items()},
    }


def main():
    base = pd.read_parquet(BASE)
    s = definir_splits(base)
    SAIDA.write_text(json.dumps(s, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in s.items() if k not in ("canais_novos", "canais_demo")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
