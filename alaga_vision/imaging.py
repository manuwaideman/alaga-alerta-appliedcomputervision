"""Leitura segura de imagens enviadas, GPS do EXIF e desenho das detecções."""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .detector import Deteccao

CORES_BGR = {
    "rua_alagada": (235, 140, 30),  # azul
    "bueiro_entupido": (0, 170, 255),  # laranja
    "veiculo_submerso": (60, 40, 230),  # vermelho
}


class ImagemInvalidaError(ValueError):
    pass


def _gps_para_graus(valor, ref) -> float:
    graus, minutos, segundos = (float(v) for v in valor)
    dec = graus + minutos / 60 + segundos / 3600
    return -dec if ref in ("S", "W") else dec


def extrair_gps(pil: Image.Image) -> tuple[float, float] | None:
    """Lê latitude/longitude do EXIF (fotos de celular costumam ter). Retorna None se não houver."""
    try:
        gps = pil.getexif().get_ifd(0x8825)  # GPSInfo
        if not gps or 2 not in gps or 4 not in gps:
            return None
        lat = _gps_para_graus(gps[2], gps.get(1, "N"))
        lon = _gps_para_graus(gps[4], gps.get(3, "E"))
        if -90 <= lat <= 90 and -180 <= lon <= 180 and (lat, lon) != (0.0, 0.0):
            return lat, lon
    except Exception:
        pass
    return None


def carregar_imagem(dados: bytes, max_side: int) -> tuple[np.ndarray, tuple[float, float] | None]:
    """bytes -> (imagem BGR já orientada e redimensionada, gps do EXIF).

    Reencodar a imagem depois descarta TODO o EXIF (GPS, modelo do aparelho, etc.),
    então nada disso é armazenado; o GPS só é usado para posicionar a ocorrência.
    """
    try:
        pil = Image.open(io.BytesIO(dados))
        pil.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImagemInvalidaError("Arquivo não é uma imagem válida (use JPG, PNG ou WEBP).") from exc
    gps = extrair_gps(pil)
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    if max(pil.size) > max_side:
        pil.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR), gps


def desenhar(img: np.ndarray, deteccoes: list[Deteccao]) -> np.ndarray:
    out = img.copy()
    espessura = max(2, round(max(img.shape[:2]) / 400))
    fonte = max(0.5, max(img.shape[:2]) / 1400)
    for d in deteccoes:
        x1, y1, x2, y2 = d.bbox
        cor = CORES_BGR.get(d.classe, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), cor, espessura)
        texto = f"{d.rotulo} {d.confianca:.0%}"
        (tw, th), base = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, fonte, max(1, espessura - 1))
        ty = max(th + base, y1)
        cv2.rectangle(out, (x1, ty - th - base), (x1 + tw + 6, ty), cor, -1)
        cv2.putText(out, texto, (x1 + 3, ty - base // 2), cv2.FONT_HERSHEY_SIMPLEX, fonte, (255, 255, 255), max(1, espessura - 1), cv2.LINE_AA)
    return out


def salvar_jpg(img: np.ndarray, caminho, qualidade: int = 85) -> None:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, qualidade])
    if not ok:
        raise RuntimeError("Falha ao codificar JPEG")
    caminho.write_bytes(buf.tobytes())
