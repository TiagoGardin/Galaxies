"""Validação das flags de formato (bloco B) e do `formato` do VLM (bloco D) contra rótulos feitos à mão.

Uso:
  uv run python -m galaxies_tool.rotulos sortear   # cria data/rotulos_formato.csv para preencher
  uv run python -m galaxies_tool.rotulos avaliar   # precisão e recall contra os rótulos preenchidos

Protocolo (docs/EXPERIMENTO.md, "Validação das flags de formato"): 100 vídeos sorteados (semente 42) do
treino, entre os que têm rubrica do VLM, para comparar as duas fontes nos mesmos vídeos. Quem rotula assiste
ao vídeo pelo link SEM olhar as flags nem a rubrica e preenche `formato_real` com o formato principal:
tutorial, grwm, haul, review, trend, humor, provador, antes_depois ou outro.
"""

import sys
from pathlib import Path

import pandas as pd

from galaxies_tool import modelo

ARQUIVO = Path("data/rotulos_formato.csv")
N = 100
FORMATOS = ["tutorial", "grwm", "haul", "review", "trend", "humor", "provador", "antes_depois", "outro"]
FLAGS_TEXTO = ["grwm", "tutorial", "haul", "review", "provador", "antes_depois", "trend"]


def sortear():
    if ARQUIVO.exists():
        raise SystemExit(f"{ARQUIVO} já existe: o sorteio é feito uma vez só")
    df = modelo.carregar()
    pool = df[(df.split == "treino") & df.tem_vlm].sort_values("video_id")
    amostra = pool.sample(N, random_state=modelo.SEMENTE)
    base = pd.read_parquet("data/processed/base_v1.parquet", columns=["video_id", "url"]).set_index(
        "video_id"
    )
    pd.DataFrame(
        {
            "video_id": amostra.video_id,
            "url": base.loc[amostra.video_id, "url"].to_numpy(),
            "formato_real": "",
            "observacao": "",
        }
    ).to_csv(ARQUIVO, index=False)
    print(f"{N} vídeos → {ARQUIVO}. Formatos válidos: {', '.join(FORMATOS)}")


def avaliar():
    rot = pd.read_csv(ARQUIVO, dtype=str).fillna("")
    rot = rot[rot.formato_real != ""]
    invalidos = set(rot.formato_real) - set(FORMATOS)
    if invalidos:
        raise SystemExit(f"formatos inválidos: {sorted(invalidos)}")
    df = modelo.carregar().set_index("video_id").loc[rot.video_id]
    real = rot.set_index("video_id").formato_real
    linhas = []
    for f in FLAGS_TEXTO:
        flag = df[f"fmt_{f}"].astype(bool)
        e_real = real == f
        linhas.append(
            {
                "fonte": "texto (bloco B)",
                "formato": f,
                "n_real": int(e_real.sum()),
                "n_previsto": int(flag.sum()),
                "precisao": (flag & e_real).sum() / flag.sum() if flag.sum() else float("nan"),
                "recall": (flag & e_real).sum() / e_real.sum() if e_real.sum() else float("nan"),
            }
        )
    vlm = df.vlm_formato.astype(str)
    for f in sorted(set(vlm) | set(real)):
        p, e_real = vlm == f, real == f
        linhas.append(
            {
                "fonte": "VLM (bloco D)",
                "formato": f,
                "n_real": int(e_real.sum()),
                "n_previsto": int(p.sum()),
                "precisao": (p & e_real).sum() / p.sum() if p.sum() else float("nan"),
                "recall": (p & e_real).sum() / e_real.sum() if e_real.sum() else float("nan"),
            }
        )
    t = pd.DataFrame(linhas)
    modelo.RESULTADOS.mkdir(exist_ok=True)
    t.to_csv(modelo.RESULTADOS / "rotulos_formato.csv", index=False)
    print(
        f"{len(rot)} vídeos rotulados. Acerto do VLM no formato principal: {(vlm == real).mean():.1%} "
        f"(o mais comum sozinho: {real.value_counts(normalize=True).iloc[0]:.1%})\n"
    )
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    {"sortear": sortear, "avaliar": avaliar}[sys.argv[1]]()
