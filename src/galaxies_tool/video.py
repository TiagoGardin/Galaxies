"""Bloco C, parte do vídeo (frames e áudio), numa amostra — mesma fila de prioridade do VLM.

Uso: uv run python -m galaxies_tool.video [--limite N]
Entrada: data/features/amostra_vlm.txt
Saída: data/features/video/<video_id>.json (cache) e frames do gancho em data/media/gancho/<id>_<k>.jpg

Para cada vídeo: baixa em 360p (yt-dlp), mede cortes de cena (PySceneDetect), movimento (optical flow),
salva 3 frames do gancho (0,5 s, 1,5 s e 2,5 s), extrai o áudio, transcreve (faster-whisper) e mede a
energia; depois APAGA o vídeo e o áudio (o disco não comporta a base inteira). Downloads em série, com
pausa entre eles, para não ser bloqueado pelo YouTube.
"""

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

AMOSTRA = Path("data/features/amostra_vlm.txt")
PASTA = Path("data/features/video")
PASTA_GANCHO = Path("data/media/gancho")
INSTANTES_GANCHO = (0.5, 1.5, 2.5)
FPS_MOVIMENTO = 2
PAUSA_ENTRE_DOWNLOADS = 8.0  # com 3 s o YouTube pediu verificação de robô após ~1.100 downloads
# só estes erros do yt-dlp viram .falha (o vídeo é pulado para sempre); qualquer outro é retomado depois.
# Na prática, "not available" são vídeos públicos marcados como "made for kids": sem login, o YouTube
# não entrega o vídeo (só o storyboard), com nenhum cliente do yt-dlp (testado em 15/10/2026).
ERROS_PERMANENTES = ("this video is not available", "video unavailable", "private video", "has been removed")
ERROS_BLOQUEIO = ("not a bot",)  # o YouTube bloqueou este IP: parar tudo, não é culpa do vídeo
_whisper = None


def baixar(video_id: str, destino: Path) -> Path:
    saida = destino / f"{video_id}.mp4"
    subprocess.run(["yt-dlp", "-q", "--no-warnings", "-f", "bv*+ba/b", "-S", "res:360,ext:mp4:m4a",
                    "--merge-output-format", "mp4", "-o", str(saida), f"https://www.youtube.com/shorts/{video_id}"],
                   check=True, timeout=300, capture_output=True, text=True)
    return saida


def cortes_por_segundo(arquivo: Path, duracao: float) -> float:
    from scenedetect import ContentDetector, detect
    cenas = detect(str(arquivo), ContentDetector())
    return max(len(cenas) - 1, 0) / duracao if duracao > 0 else 0.0


def movimento_e_gancho(arquivo: Path, video_id: str) -> tuple[float, int]:
    """Magnitude média do optical flow (Farneback) a 2 quadros/s em escala de cinza reduzida,
    e grava os quadros do gancho. Retorna (movimento, nº de quadros do gancho salvos)."""
    cap = cv2.VideoCapture(str(arquivo))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    passo = max(round(fps / FPS_MOVIMENTO), 1)
    alvos_gancho = {round(t * fps): k for k, t in enumerate(INSTANTES_GANCHO)}
    anterior, magnitudes, salvos, i = None, [], 0, 0
    while True:
        ok, quadro = cap.read()
        if not ok:
            break
        if i in alvos_gancho:
            cv2.imwrite(str(PASTA_GANCHO / f"{video_id}_{alvos_gancho[i]}.jpg"), quadro)
            salvos += 1
        if i % passo == 0:
            cinza = cv2.cvtColor(cv2.resize(quadro, (180, 320)), cv2.COLOR_BGR2GRAY)
            if anterior is not None:
                fluxo = cv2.calcOpticalFlowFarneback(anterior, cinza, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                magnitudes.append(float(np.linalg.norm(fluxo, axis=2).mean()))
            anterior = cinza
        i += 1
    cap.release()
    return (float(np.mean(magnitudes)) if magnitudes else 0.0), salvos


def audio(arquivo: Path, pasta: Path) -> dict:
    """Voz (fala ou canto — o Whisper transcreve os dois; regra 9 de docs/FEATURES.md) e energia RMS."""
    global _whisper
    wav = pasta / "audio.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(arquivo), "-ac", "1", "-ar", "16000", str(wav)],
                   check=True, timeout=300)
    import soundfile as sf
    sinal, taxa = sf.read(str(wav))
    duracao = len(sinal) / taxa if len(sinal) else 0.0
    if _whisper is None:
        from faster_whisper import WhisperModel
        _whisper = WhisperModel("small", device="cpu", compute_type="int8")
    # passa o sinal já decodificado (16 kHz mono): evita a decodificação via `av`, incompatível aqui
    segmentos, _ = _whisper.transcribe(np.asarray(sinal, dtype=np.float32), vad_filter=True, beam_size=1)
    palavras = sum(len(s.text.split()) for s in segmentos)
    return {"tem_voz": palavras >= 3, "voz_por_segundo": palavras / duracao if duracao else 0.0,
            "energia_rms": float(np.sqrt(np.mean(np.square(sinal)))) if len(sinal) else 0.0,
            "duracao_audio_s": duracao}


def medir(arquivo: Path, chave: str, pasta: Path) -> dict:
    """Medidas de um vídeo já em disco; os quadros do gancho vão para PASTA_GANCHO/<chave>_<k>.jpg.
    Usada no treino (vídeo baixado) e pelo agente (vídeo baixado ou arquivo local do cliente)."""
    PASTA_GANCHO.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(arquivo))
    duracao = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30)
    largura, altura = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    movimento, n_gancho = movimento_e_gancho(arquivo, chave)
    return {"video_id": chave, "duracao_video_s": duracao, "largura": largura, "altura": altura,
            "vertical": altura > largura, "cortes_por_segundo": cortes_por_segundo(arquivo, duracao),
            "movimento_medio": movimento, "n_quadros_gancho": n_gancho, **audio(arquivo, pasta)}


def processar(video_id: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:  # vídeo e áudio somem ao sair do bloco
        pasta = Path(tmp)
        return medir(baixar(video_id, pasta), video_id, pasta)


def quadros_do_gancho(chave: str) -> list:
    from PIL import Image
    return [Image.open(PASTA_GANCHO / f"{chave}_{k}.jpg").convert("RGB") for k in range(len(INSTANTES_GANCHO))
            if (PASTA_GANCHO / f"{chave}_{k}.jpg").exists()]

def detalhe_do_erro(e: Exception) -> str:
    return (getattr(e, "stderr", None) or str(e)).strip()  # mensagem do yt-dlp/ffmpeg, se houver


def tipo_de_erro(detalhe: str) -> str:
    """Classifica o erro: "permanente" (made for kids, removido ou privado), "bloqueio" (verificação de robô) ou
    "temporario" (403, rede, qualquer outro)."""
    texto = detalhe.lower()
    if any(m in texto for m in ERROS_BLOQUEIO):
        return "bloqueio"
    if any(m in texto for m in ERROS_PERMANENTES):
        return "permanente"
    return "temporario"


def processar_com_tentativas(video_id: str, tentativas: int = 3) -> dict:
    """403 e falhas de rede do YouTube costumam ser temporárias: tenta de novo com pausa crescente."""
    for t in range(tentativas):
        try:
            return processar(video_id)
        except Exception as e:
            if t == tentativas - 1 or tipo_de_erro(detalhe_do_erro(e)) != "temporario":
                raise
            time.sleep(15 * (t + 1))
    raise AssertionError("inalcançável")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limite", type=int, default=None)
    args = ap.parse_args()
    PASTA.mkdir(parents=True, exist_ok=True)
    PASTA_GANCHO.mkdir(parents=True, exist_ok=True)
    fila = [v for v in AMOSTRA.read_text().split()
            if not (PASTA / f"{v}.json").exists() and not (PASTA / f"{v}.falha").exists()]
    if args.limite:
        fila = fila[:args.limite]
    print(f"{len(fila)} vídeos na fila", flush=True)
    ok = falhas = seguidas = 0
    for i, vid in enumerate(fila, 1):
        try:
            (PASTA / f"{vid}.json").write_text(json.dumps(processar_com_tentativas(vid)))
            ok, seguidas = ok + 1, 0
        except Exception as e:  # noqa: BLE001 — vídeo removido, privado, bloqueio do YouTube etc.
            falhas, seguidas = falhas + 1, seguidas + 1
            detalhe, tipo = detalhe_do_erro(e), tipo_de_erro(detalhe_do_erro(e))
            print(f"{vid} ({tipo}): {detalhe[-200:]}", flush=True)
            if tipo == "bloqueio":
                print("Verificação de robô do YouTube. Parando; retome mais tarde.", flush=True)
                return
            if tipo == "permanente":
                (PASTA / f"{vid}.falha").write_text(f"{type(e).__name__}: {detalhe[-500:]}")
            if seguidas >= 10:  # 10 falhas seguidas indicam bloqueio, não vídeos ruins
                for v in fila[i - 10:i]:
                    (PASTA / f"{v}.falha").unlink(missing_ok=True)
                print("10 falhas seguidas: provável bloqueio do YouTube. Parando; retome mais tarde.", flush=True)
                return
        if i % 25 == 0:
            print(f"{i}/{len(fila)}: {ok} ok, {falhas} falhas", flush=True)
        time.sleep(PAUSA_ENTRE_DOWNLOADS)
    print(f"Fim: {ok} ok, {falhas} falhas", flush=True)


if __name__ == "__main__":
    main()
