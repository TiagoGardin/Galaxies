"""Cliente fino da YouTube Data API v3 com contagem de cota."""

import os
import re
import time
from datetime import timedelta

import httplib2
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Custo em unidades de cada endpoint usado
CUSTO = {"channels": 1, "playlistItems": 1, "videos": 1, "search": 100}


class YouTube:
    def __init__(self, env_path: str = ".env"):
        load_dotenv(env_path)
        self.api = build("youtube", "v3", developerKey=os.environ["YOUTUBE_API_KEY"], cache_discovery=False)
        self.cota_usada = 0

    def _executar(self, endpoint: str, request, tentativas: int = 8):
        for i in range(tentativas):
            try:
                resp = request.execute()
                self.cota_usada += CUSTO[endpoint]
                return resp
            except HttpError as e:
                self.cota_usada += CUSTO[endpoint]  # requisições com erro também consomem cota
                erro = str(e)
                if "quotaExceeded" in erro:
                    raise RuntimeError(f"Cota diária esgotada (usado nesta execução: {self.cota_usada})") from e
                temporario = e.resp.status >= 500 or "rateLimitExceeded" in erro
                if not temporario:
                    raise
                time.sleep(min(2 ** (i + 1), 60))
            except (TimeoutError, ConnectionError, OSError, httplib2.HttpLib2Error):  # falha de rede/DNS
                time.sleep(min(2 ** (i + 1), 60))
        raise RuntimeError(f"Falha repetida em {endpoint}")

    def canal_por_handle(self, handle: str) -> dict | None:
        r = self._executar("channels", self.api.channels().list(
            part="snippet,statistics,contentDetails", forHandle=handle))
        return (r.get("items") or [None])[0]

    def canais_por_ids(self, ids: list[str]) -> list[dict]:
        itens = []
        for i in range(0, len(ids), 50):
            r = self._executar("channels", self.api.channels().list(
                part="snippet,statistics,contentDetails", id=",".join(ids[i:i + 50]), maxResults=50))
            itens += r.get("items", [])
        return itens

    def uploads(self, playlist_id: str, max_paginas: int | None = None, parar_antes: str | None = None):
        """Itera itens da playlist de uploads (mais recentes primeiro).

        `parar_antes`: data ISO; para de paginar quando a página termina em vídeo anterior a ela.
        """
        token, paginas = None, 0
        while True:
            try:
                r = self._executar("playlistItems", self.api.playlistItems().list(
                    part="contentDetails", playlistId=playlist_id, maxResults=50, pageToken=token))
            except HttpError as e:
                if e.resp.status == 404:  # playlist de uploads vazia/inexistente
                    return
                raise
            itens = r.get("items", [])
            yield from itens
            paginas += 1
            token = r.get("nextPageToken")
            ultimo = itens[-1]["contentDetails"].get("videoPublishedAt", "") if itens else ""
            if not token or (max_paginas and paginas >= max_paginas) or (parar_antes and ultimo and ultimo < parar_antes):
                return

    def videos(self, ids: list[str]) -> list[dict]:
        itens = []
        for i in range(0, len(ids), 50):
            r = self._executar("videos", self.api.videos().list(
                part="snippet,contentDetails,statistics", id=",".join(ids[i:i + 50]), maxResults=50))
            itens += r.get("items", [])
        return itens

    def status_videos(self, ids: list[str]) -> dict[str, dict]:
        """`status` de cada vídeo ainda servido pelo YouTube; vídeos removidos não aparecem na resposta."""
        status = {}
        for i in range(0, len(ids), 50):
            r = self._executar("videos", self.api.videos().list(
                part="status", id=",".join(ids[i:i + 50]), maxResults=50))
            status.update({it["id"]: it["status"] for it in r.get("items", [])})
        return status

    def buscar_shorts(self, q: str, publicado_depois: str, paginas: int = 1) -> list[dict]:
        """search.list por vídeos curtos no Brasil/pt. Caro (100 un./página): usar só para descoberta."""
        itens, token = [], None
        for _ in range(paginas):
            r = self._executar("search", self.api.search().list(
                part="snippet", q=q, type="video", videoDuration="short", regionCode="BR",
                relevanceLanguage="pt", publishedAfter=publicado_depois, maxResults=50, pageToken=token))
            itens += r.get("items", [])
            token = r.get("nextPageToken")
            if not token:
                break
        return itens


_ISO = re.compile(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")


def duracao_segundos(iso: str) -> int:
    """Converte duração ISO 8601 (ex.: PT1M5S) em segundos."""
    m = _ISO.fullmatch(iso or "")
    if not m:
        return -1
    d, h, mi, s = (int(x or 0) for x in m.groups())
    return int(timedelta(days=d, hours=h, minutes=mi, seconds=s).total_seconds())
