"""Bloco D: rubrica respondida por VLM (Gemini Flash-Lite) a partir da URL pública do Short.

Uso: uv run python -m galaxies_tool.vlm [--limite N] [--modelo NOME]
Entrada: data/features/fila_vlm.txt (video_id por linha, em ordem de prioridade; ver amostra_vlm.py)
Saída: data/features/vlm/<video_id>.json (cache: cada vídeo é processado uma única vez). Outro modelo
grava em data/features/vlm_<modelo>/, para que rubricas de modelos diferentes nunca se misturem.

Para não gastar a cota do Gemini com vídeos que o YouTube não serve mais, cada execução tira da fila os
vídeos removidos ou privados (pela YouTube Data API, ~1 unidade a cada 50 vídeos) e os marca com
<video_id>.falha, assim como os vídeos que o Gemini recusa (400/403). Os .falha do pipeline de vídeo NÃO
servem para isso: lá eles são vídeos "made for kids", que o yt-dlp não baixa mas o Gemini analisa.

Regra 6 de docs/FEATURES.md: o modelo recebe SÓ o vídeo — sem título, canal ou métricas.
`potencial_viral_0a10` é coletado mas nunca entra no modelo (só na comparação LLM zero-shot × modelo).
"""

import argparse
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field

MODELO = "gemini-3.1-flash-lite"  # o 3.5 vivia sobrecarregado (503); ver docs/FEATURES.md
TEMPO_LIMITE_MS = 180_000
AMOSTRA = Path("data/features/fila_vlm.txt")
PASTA = Path("data/features/vlm")

PROMPT = """Você é um analista de criativos de moda e beleza. Assista ao vídeo e preencha a rubrica.
Avalie apenas o que aparece e se ouve no vídeo. Não use conhecimento sobre quem é o criador,
sobre a marca ou sobre a popularidade do vídeo. Notas de 0 a 10: 0 = ausente/péssimo, 10 = excelente."""


class CorpoInteiro(str, Enum):
    corpo_inteiro = "corpo_inteiro"
    detalhe = "detalhe"
    nao_se_aplica = "nao_se_aplica"


class Audio(str, Enum):
    fala = "fala"
    musica = "musica"
    ambos = "ambos"
    ambiente = "ambiente"


class TomNarracao(str, Enum):
    ausente = "ausente"
    calmo = "calmo"
    animado = "animado"
    instrucional = "instrucional"


class Formato(str, Enum):
    tutorial = "tutorial"
    grwm = "grwm"
    haul = "haul"
    review = "review"
    trend = "trend"
    humor = "humor"
    outro = "outro"


def nota(descricao: str):
    return Field(ge=0, le=10, description=descricao)


class Rubrica(BaseModel):
    forca_hook_0a10: int = nota("Força do gancho nos primeiros 3 segundos")
    tem_rosto: bool = Field(description="Aparece rosto humano")
    close_up_produto: bool = Field(description="Há close-up de um produto")
    before_after_ou_transformacao: bool = Field(description="Mostra antes/depois ou transformação")
    try_on_ou_demonstracao: bool = Field(description="Prova de roupa ou demonstração de uso do produto")
    swatch: bool = Field(description="Mostra swatch (amostra de cor/textura do produto na pele ou superfície)")
    aplicacao_na_pele: bool = Field(description="Aplica produto na pele, rosto, cabelo ou unhas")
    corpo_inteiro_vs_detalhe: CorpoInteiro = Field(description="Em moda, se o look aparece de corpo inteiro ou só em detalhe")
    texto_na_tela: bool = Field(description="Há texto sobreposto na tela")
    densidade_texto_0a10: int = nota("Quanto texto aparece na tela")
    ritmo_cortes_0a10: int = nota("Ritmo de edição/cortes")
    tem_musica_de_fundo: bool = Field(description="Há música de fundo")
    audio_original_ou_musica: Audio = Field(description="Tipo de áudio predominante")
    tom_narracao: TomNarracao = Field(description="Tom da narração, se houver")
    estetica_geral_0a10: int = nota("Qualidade estética e de produção")
    clareza_beneficio_produto_0a10: int = nota("Clareza do benefício do produto mostrado")
    formato: Formato = Field(description="Formato principal do vídeo")
    potencial_viral_0a10: int = nota("Potencial de engajamento estimado")


def pasta_do_modelo(modelo: str) -> Path:
    return PASTA if modelo == MODELO else PASTA.with_name(f"vlm_{modelo}")


def disponivel(status: dict | None) -> bool:
    """O YouTube ainda serve o vídeo (não foi removido, rejeitado nem tornado privado)."""
    return (status is not None and status.get("uploadStatus") == "processed"
            and status.get("privacyStatus") in ("public", "unlisted"))


def tirar_indisponiveis(fila: list[str], pasta: Path) -> list[str]:
    """Marca com .falha e tira da fila os vídeos que o YouTube não serve mais."""
    removidos = {}
    try:
        from galaxies_tool.youtube_api import YouTube
        status = YouTube().status_videos(fila)
        removidos = {v: f"YouTube Data API: {status.get(v) or 'removido'}" for v in fila
                     if not disponivel(status.get(v))}
    except Exception as e:  # noqa: BLE001 — sem a checagem o VLM ainda funciona, só gasta um pouco mais de cota
        print(f"Checagem de disponibilidade falhou ({e}); seguindo sem ela", flush=True)
    for v, motivo in removidos.items():
        (pasta / f"{v}.falha").write_text(motivo)
    if removidos:
        print(f"{len(removidos)} vídeos indisponíveis tirados da fila", flush=True)
    return [v for v in fila if v not in removidos]


def analisar(cliente: genai.Client, video_id: str, modelo: str = MODELO) -> dict:
    url = f"https://www.youtube.com/shorts/{video_id}"
    resp = cliente.models.generate_content(
        model=modelo,
        contents=types.Content(parts=[types.Part(file_data=types.FileData(file_uri=url)),
                                      types.Part(text=PROMPT)]),
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=Rubrica,
                                           temperature=0.0),
    )
    rubrica = Rubrica.model_validate_json(resp.text)
    uso = resp.usage_metadata
    return {"video_id": video_id, "modelo": modelo, "rubrica": rubrica.model_dump(mode="json"),
            "tokens_entrada": uso.prompt_token_count, "tokens_saida": uso.candidates_token_count}


def processar_um(cliente: genai.Client, vid: str, modelo: str = MODELO) -> str:
    """Erros temporários (429 de limite por minuto, 503 de alta demanda) esperam e tentam de novo sem
    limite de vezes; resposta inválida é reenviada uma vez e depois o vídeo fica sem rubrica."""
    espera, invalidas = 10, 0
    while True:
        try:
            dados = analisar(cliente, vid, modelo)
            (pasta_do_modelo(modelo) / f"{vid}.json").write_text(json.dumps(dados, ensure_ascii=False))
            return "ok"
        except errors.APIError as e:
            texto = str(e).lower()
            if e.code == 429 and ("perday" in texto or "per day" in texto):
                limite = re.search(r"limit: (\d+)", str(e))
                print(f"Cota diária de {modelo}: {limite.group(1) if limite else '?'} pedidos", flush=True)
                return "cota_diaria"
            if e.code in (429, 500, 503, 504):
                time.sleep(espera)
                espera = min(espera * 2, 600)
                continue
            if e.code == 400 or e.code == 403:  # vídeo indisponível/privado/bloqueado: não tentar de novo
                (pasta_do_modelo(modelo) / f"{vid}.falha").write_text(f"Gemini {e.code}: {str(e)[:500]}")
                return f"erro_{e.code}"
            invalidas += 1
        except (httpx.TimeoutException, httpx.NetworkError):  # rede caiu/travou: temporário
            time.sleep(espera)
            espera = min(espera * 2, 600)
            continue
        except Exception:  # noqa: BLE001 — JSON fora do esquema
            invalidas += 1
        if invalidas >= 2:
            return "invalido"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=None)
    ap.add_argument("--paralelo", type=int, default=3)
    ap.add_argument("--modelo", default=MODELO)
    args = ap.parse_args()
    load_dotenv(".env")
    # tempo limite por requisição: após suspensão do computador, conexões mortas travariam para sempre
    cliente = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                           http_options=types.HttpOptions(timeout=TEMPO_LIMITE_MS))
    pasta = pasta_do_modelo(args.modelo)
    pasta.mkdir(parents=True, exist_ok=True)
    fila = [v for v in AMOSTRA.read_text().split()
            if not (pasta / f"{v}.json").exists() and not (pasta / f"{v}.falha").exists()]
    if args.limite:
        fila = fila[:args.limite]
    fila = tirar_indisponiveis(fila, pasta)
    print(f"{args.modelo}: {len(fila)} vídeos na fila → {pasta}", flush=True)
    contagem = Counter()
    with ThreadPoolExecutor(args.paralelo) as ex:
        futuros = {ex.submit(processar_um, cliente, v, args.modelo): v for v in fila}
        for i, fut in enumerate(as_completed(futuros), 1):
            res = fut.result()
            contagem[res] += 1
            if res not in ("ok", "cota_diaria"):
                print(f"{futuros[fut]}: {res}", flush=True)
            if res == "cota_diaria":
                print("Cota diária do Gemini atingida. Cancelando o restante; retome amanhã.", flush=True)
                for f in futuros:
                    f.cancel()
                break
            if i % 50 == 0:
                print(f"{i}/{len(fila)}: {dict(contagem)}", flush=True)
    print(f"Fim: {dict(contagem)}", flush=True)


if __name__ == "__main__":
    main()
