"""Publica as ocorrências confirmadas na API central do MVP Alaga Alerta.

Ativada só quando MVP_URL e MVP_API_KEY estão no ambiente. Sem elas, a API de
visão funciona exatamente como antes (mapa próprio).

Fluxo: ocorrência confirmada (lat/lon) -> bairro do MVP mais próximo
(GET /bairros) -> POST /deteccoes com X-API-Key. Uma falha aqui nunca derruba
a análise da foto: o erro é registrado e a resposta ao cidadão segue normal.
"""
from __future__ import annotations

import logging
import math
import os
import time

import httpx

log = logging.getLogger("alaga_vision.mvp")

RAIO_MAX_KM = 15.0  # longe disso de qualquer bairro monitorado, não publica
CACHE_BAIRROS_S = 600


class PublicadorMVP:
    def __init__(self, url: str, chave: str, cliente: httpx.Client | None = None):
        self.url = url.rstrip("/")
        self.chave = chave
        self.cliente = cliente or httpx.Client(timeout=5)
        self._bairros: list[dict] = []
        self._lidos_em = 0.0

    @classmethod
    def do_ambiente(cls) -> "PublicadorMVP | None":
        url, chave = os.getenv("MVP_URL", "").strip(), os.getenv("MVP_API_KEY", "").strip()
        return cls(url, chave) if url and chave else None

    def _carregar_bairros(self) -> list[dict]:
        if not self._bairros or time.monotonic() - self._lidos_em > CACHE_BAIRROS_S:
            r = self.cliente.get(f"{self.url}/bairros")
            r.raise_for_status()
            self._bairros, self._lidos_em = r.json(), time.monotonic()
        return self._bairros

    @staticmethod
    def _distancia_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        p1, p2 = math.radians(lat1), math.radians(lat2)
        dp, dl = p2 - p1, math.radians(lon2 - lon1)
        a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 6371 * 2 * math.asin(math.sqrt(a))

    def bairro_mais_proximo(self, lat: float, lon: float) -> dict | None:
        candidatos = [(self._distancia_km(lat, lon, b["lat"], b["lon"]), b) for b in self._carregar_bairros()]
        if not candidatos:
            return None
        dist, bairro = min(candidatos, key=lambda c: c[0])
        return bairro if dist <= RAIO_MAX_KM else None

    def publicar(self, lat: float, lon: float, deteccoes: list[dict]) -> dict:
        """Retorna um resumo para incluir na resposta da API de visão. Nunca levanta exceção."""
        try:
            bairro = self.bairro_mais_proximo(lat, lon)
            if bairro is None:
                return {"publicada": False, "motivo": "fora da área dos bairros monitorados"}
            corpo = {
                "bairro_id": bairro["id"],
                "imagem_anonimizada": True,  # rostos e placas já foram borrados antes da detecção
                "deteccoes": [{"classe": d["classe"], "confianca": float(d["confianca"])} for d in deteccoes],
            }
            r = self.cliente.post(f"{self.url}/deteccoes", json=corpo, headers={"X-API-Key": self.chave})
            r.raise_for_status()
            log.info("ocorrência publicada no MVP: bairro=%s", bairro["id"])
            return {"publicada": True, "bairro_id": bairro["id"], "bairro": bairro.get("nome")}
        except httpx.HTTPStatusError as exc:
            # Só o código HTTP: nunca a chave nem o corpo.
            log.warning("MVP recusou a ocorrência: HTTP %s", exc.response.status_code)
            return {"publicada": False, "motivo": f"MVP respondeu HTTP {exc.response.status_code}"}
        except Exception as exc:  # MVP fora do ar, timeout, resposta inválida
            log.warning("MVP indisponível: %s", type(exc).__name__)
            return {"publicada": False, "motivo": "MVP indisponível"}
