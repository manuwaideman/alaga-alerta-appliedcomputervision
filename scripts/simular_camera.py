"""Simula uma câmera de rua enviando quadros para a API (ex.: CFTV da prefeitura).

    python scripts/simular_camera.py --fonte video.mp4 --lat -23.5614 --lon -46.6559 --camera cam-paulista-01
    python scripts/simular_camera.py --fonte pasta_de_fotos/ --intervalo 5

Lê um vídeo (1 quadro a cada --intervalo segundos de vídeo) ou uma pasta de imagens
e faz POST /api/deteccoes com origem=camera. Útil para a demonstração em vídeo.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import httpx

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def quadros(fonte: Path, intervalo: float):
    if fonte.is_dir():
        for p in sorted(fonte.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                yield p.name, p.read_bytes()
        return
    cap = cv2.VideoCapture(str(fonte))
    if not cap.isOpened():
        sys.exit(f"Não foi possível abrir {fonte}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    passo = max(1, int(fps * intervalo))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % passo == 0:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok:
                yield f"quadro_{i:06d}.jpg", buf.tobytes()
        i += 1
    cap.release()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fonte", required=True, type=Path, help="arquivo de vídeo ou pasta de imagens")
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--lat", type=float, default=-23.5614)
    ap.add_argument("--lon", type=float, default=-46.6559)
    ap.add_argument("--camera", default="cam-demo-01")
    ap.add_argument("--intervalo", type=float, default=2.0, help="segundos entre quadros enviados")
    ap.add_argument("--pausa", type=float, default=1.0, help="pausa real entre envios (s)")
    args = ap.parse_args()

    with httpx.Client(timeout=60) as cli:
        for nome, dados in quadros(args.fonte, args.intervalo):
            r = cli.post(
                f"{args.api}/api/deteccoes",
                files={"imagem": (nome, dados, "image/jpeg")},
                data={"latitude": args.lat, "longitude": args.lon, "origem": "camera", "camera_id": args.camera},
            )
            if r.status_code != 200:
                print(f"{nome}: erro {r.status_code} {r.text[:200]}")
            else:
                b = r.json()
                classes = ", ".join(f"{d['rotulo']} {d['confianca']:.0%}" for d in b["deteccoes"]) or "nada"
                marca = "CONFIRMADA" if b["confirmada"] else "descartada"
                print(f"{nome}: {marca:<10} risco={b['risco']['nivel']:<6} [{classes}]")
            time.sleep(args.pausa)


if __name__ == "__main__":
    main()
