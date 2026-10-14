"""Features dos blocos A (canal, tempo, metadados) e B (texto, sem embedding), conforme docs/FEATURES.md.

Uso: uv run python -m galaxies_tool.features
Entradas: data/processed/base_v1.parquet, data/raw/coleta=*/videos.parquet, data/processed/splits_v1.json
Saída: data/features/features_ab.parquet (uma linha por vídeo-alvo elegível, com alvo e split)

Regras anti-vazamento aplicadas aqui:
- histórico de curtidas: só Shorts do canal publicados >= 30 dias antes do alvo (regra 2);
- atividade de postagem: só Shorts publicados antes do alvo (datas, não curtidas);
- horário local em America/Sao_Paulo (regra 7);
- nenhuma feature usa curtidas, views ou comentários do próprio vídeo (regra 1).
"""

import json
import re
import unicodedata
from datetime import date, timedelta
from pathlib import Path

import emoji
import numpy as np
import pandas as pd

from galaxies_tool.processar import pasta_coleta
from galaxies_tool.regras import IDADE_MINIMA_DIAS, eh_short
from galaxies_tool.splits import atribuir_split

BASE = Path("data/processed/base_v1.parquet")
SPLITS = Path("data/processed/splits_v1.json")
SAIDA = Path("data/features/features_ab.parquet")
FUSO = "America/Sao_Paulo"
JANELA_HISTORICO = 20
JANELA_TENDENCIA = 5
TETO_N_HISTORICO = 25  # a coleta só garante a contagem exata até 25 (DATA_CARD)

# ---------------------------------------------------------------- histórico e alvo


def historico_do_alvo(base: pd.DataFrame, alvos: pd.DataFrame) -> pd.DataFrame:
    """Estatísticas das curtidas dos últimos (até) 20 Shorts publicados >= 30 dias antes de cada alvo."""
    defasagem = np.timedelta64(IDADE_MINIMA_DIAS, "D")
    linhas = []
    for cid, alv in alvos.groupby("channel_id"):
        h = base[base.channel_id == cid].sort_values("published_at")
        datas, likes = h.published_at.values, h.likes.to_numpy(dtype=float)
        fim = np.searchsorted(datas, alv.published_at.values - defasagem, side="right")
        for idx, f in zip(alv.index, fim, strict=True):
            janela = likes[max(0, f - JANELA_HISTORICO):f]
            if len(janela) == 0:
                linhas.append((idx, np.nan, np.nan, np.nan, 0))
                continue
            mediana = np.median(janela)
            log_janela = np.log1p(janela)
            std = float(np.std(log_janela, ddof=1)) if len(janela) >= 2 else np.nan
            tendencia = np.log1p(np.median(janela[-JANELA_TENDENCIA:])) - np.log1p(mediana)
            linhas.append((idx, mediana, std, tendencia, int(f)))
    out = pd.DataFrame(linhas, columns=["idx", "mediana_likes_ult20", "std_log_likes_ult20", "tendencia",
                                        "n_hist"]).set_index("idx")
    out["log_mediana_likes_ult20"] = np.log1p(out.mediana_likes_ult20)
    return out


def atividade_de_postagem(shorts_raw: pd.DataFrame, alvos: pd.DataFrame) -> pd.DataFrame:
    """Frequência recente e intervalo desde o último Short, usando só Shorts anteriores ao alvo."""
    janela = np.timedelta64(30, "D")
    linhas = []
    for cid, alv in alvos.groupby("channel_id"):
        datas = np.sort(shorts_raw.loc[shorts_raw.channel_id == cid, "published_at"].values)
        t = alv.published_at.values
        antes = np.searchsorted(datas, t, side="left")          # Shorts estritamente anteriores
        desde = np.searchsorted(datas, t - janela, side="left")  # publicados em [t-30d, t)
        ultimo = np.where(antes > 0, datas[np.maximum(antes - 1, 0)], np.datetime64("NaT", "ns"))
        dias = (t - ultimo) / np.timedelta64(1, "D")
        linhas += list(zip(alv.index, antes - desde, dias, strict=True))
    return pd.DataFrame(linhas, columns=["idx", "shorts_ult_30d", "dias_desde_ultimo_post"]).set_index("idx")


# ---------------------------------------------------------------- tempo


def _nesimo_dia_semana(ano: int, mes: int, dia_semana: int, n: int) -> date:
    d = date(ano, mes, 1)
    d += timedelta(days=(dia_semana - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def janelas_sazonais(ano: int) -> list[tuple[date, date]]:
    """Janelas comerciais de moda e beleza no Brasil (inclusivas)."""
    black_friday = _nesimo_dia_semana(ano, 11, 3, 4) + timedelta(days=1)  # dia seguinte à 4ª quinta de novembro
    maes = _nesimo_dia_semana(ano, 5, 6, 2)                                # 2º domingo de maio
    return [
        (black_friday - timedelta(days=7), black_friday + timedelta(days=3)),  # semana da BF até a Cyber Monday
        (maes - timedelta(days=14), maes),
        (date(ano, 6, 1), date(ano, 6, 12)),                                   # Dia dos Namorados
        (date(ano, 12, 1), date(ano, 12, 24)),                                 # Natal
    ]


def eh_data_sazonal(d: date) -> bool:
    return any(ini <= d <= fim for ini, fim in janelas_sazonais(d.year))


def features_tempo(published_at: pd.Series) -> pd.DataFrame:
    local = published_at.dt.tz_convert(FUSO)
    return pd.DataFrame({
        "hora": local.dt.hour,
        "dia_semana": local.dt.dayofweek,
        "fim_de_semana": local.dt.dayofweek >= 5,
        "mes": local.dt.month,
        "data_sazonal": [eh_data_sazonal(d) for d in local.dt.date],
    }, index=published_at.index)


# ---------------------------------------------------------------- texto

TERMOS_FORMATO = {
    "fmt_grwm": r"\bgrwm\b|get ready with me|arrume[- ]se comigo|se arrume comigo",
    "fmt_tutorial": r"\btutorial|passo a passo|como fazer|\baprenda\b|\bdica",
    "fmt_haul": r"\bhaul\b|comprinhas|\brecebidos\b|unboxing|\babrindo\b",
    "fmt_review": r"\bresenha|\breview|\btestei\b|\btestando\b|vale a pena|primeiras impressoes",
    "fmt_provador": r"\bprovador|\bprovando\b|look do dia|lookbook|try ?on|\bootd\b",
    "fmt_antes_depois": r"antes e depois|transformacao|\bresultado\b",
    "fmt_trend": r"\btrend|\bdesafio\b|challenge",
}
CTA = r"\bcomenta\b|\bcomente\b|\bsalva\b|\bsalve\b|\bcompartilha\b|\bcompartilhe\b|link na bio|\bcupom\b|eu quero"
_REGEX_FORMATO = {k: re.compile(v) for k, v in TERMOS_FORMATO.items()}
_REGEX_CTA = re.compile(CTA)


def normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return sem_acento.lower()


def features_texto(titulo: str, descricao: str, tags) -> dict:
    tags = list(tags) if tags is not None else []
    completo = normalizar(" ".join([titulo, descricao, " ".join(tags)]))
    letras = [c for c in titulo if c.isalpha()]
    f = {
        "titulo_n_chars": len(titulo),
        "titulo_n_palavras": len(titulo.split()),
        "n_hashtags": len(re.findall(r"#\w+", f"{titulo} {descricao}")),
        "tem_hashtag_shorts": "#shorts" in completo,
        "n_emojis": emoji.emoji_count(titulo),
        "titulo_pergunta": "?" in titulo,
        "titulo_exclamacao": "!" in titulo,
        "prop_maiusculas": sum(c.isupper() for c in letras) / len(letras) if letras else 0.0,
        "titulo_tem_numero": bool(re.search(r"\d", titulo)),
        "n_mencoes": len(re.findall(r"@\w+", descricao)),
        "tem_link": bool(re.search(r"https?://", descricao)),
        "tem_cta": bool(_REGEX_CTA.search(completo)),
    }
    f.update({k: bool(rx.search(completo)) for k, rx in _REGEX_FORMATO.items()})
    return f


# ---------------------------------------------------------------- montagem


def shorts_da_coleta() -> pd.DataFrame:
    raw = pd.read_parquet(pasta_coleta(None) / "videos.parquet", columns=[
        "channel_id", "published_at", "duration_s", "live_broadcast_content"])
    ok = np.array([eh_short(d, p) for d, p in zip(raw.duration_s, raw.published_at, strict=True)])
    raw = raw[ok & raw.live_broadcast_content.isin(["none", ""])]
    return raw.assign(published_at=pd.to_datetime(raw.published_at, utc=True).dt.as_unit("ns"))


def montar(base: pd.DataFrame, shorts_raw: pd.DataFrame, splits: dict) -> pd.DataFrame:
    alvos = base[base.elegivel_treino].copy()
    hist = historico_do_alvo(base, alvos)
    ativ = atividade_de_postagem(shorts_raw, alvos)
    texto = pd.DataFrame([features_texto(t, d, g) for t, d, g in
                          zip(alvos.title, alvos.description, alvos.tags, strict=True)], index=alvos.index)

    df = pd.concat([alvos[["video_id", "channel_id", "published_at", "likes", "sector", "channel_type",
                           "is_demo_channel", "n_historico", "duration_s", "category_id", "tags"]],
                    hist, ativ, features_tempo(alvos.published_at), texto], axis=1)
    assert (df.n_hist == df.n_historico).all(), "contagem de histórico divergente da base"
    df["n_shorts_anteriores"] = df.n_historico.clip(upper=TETO_N_HISTORICO)
    df["n_tags"] = df.tags.map(lambda t: 0 if t is None else len(t))
    df["alvo"] = np.log1p(df.likes) - df.log_mediana_likes_ult20
    df["split"] = atribuir_split(df, splits)
    df = df.rename(columns={"sector": "setor", "channel_type": "tipo_canal", "duration_s": "duracao_s",
                            "category_id": "categoria_id"})
    df = df.drop(columns=["tags", "n_hist", "n_historico", "mediana_likes_ult20"])
    for col in ("setor", "tipo_canal", "categoria_id"):
        df[col] = df[col].astype("category")
    return df.reset_index(drop=True)


def main():
    base = pd.read_parquet(BASE)
    df = montar(base, shorts_da_coleta(), json.loads(SPLITS.read_text()))
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(SAIDA, index=False)
    print(f"{len(df)} vídeos, {df.shape[1]} colunas → {SAIDA}")
    print(df.split.value_counts().to_string())


if __name__ == "__main__":
    main()
