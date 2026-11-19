"""Download dos modelos auxiliares de anonimização (rostos e placas)."""
from __future__ import annotations

import urllib.request
from pathlib import Path

# YuNet (OpenCV Zoo, licença MIT): detector de rostos leve em ONNX.
YUNET = (
    "face_detection_yunet_2023mar.onnx",
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
)
# YOLOv11n fine-tuned para placas (morsetechlab, AGPL-3.0, mesma licença do Ultralytics).
# Usamos a versão ONNX de propósito: .pt é pickle e pode executar código ao ser carregado;
# ONNX é só o grafo de operações, então um modelo de terceiros não roda nada além da inferência.
PLACAS = (
    "license-plate-finetune-v1n.onnx",
    "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1n.onnx",
)


def baixar(url: str, destino: Path) -> Path:
    destino = Path(destino)
    if destino.exists() and destino.stat().st_size > 0:
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(destino.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "alaga-alerta-setup"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
        while chunk := resp.read(1 << 16):
            f.write(chunk)
    tmp.replace(destino)
    return destino


def garantir(face_path: Path, plate_path: Path) -> list[str]:
    """Baixa o que estiver faltando. Retorna a lista de erros (vazia = tudo ok)."""
    erros = []
    for (_, url), destino in ((YUNET, face_path), (PLACAS, plate_path)):
        try:
            baixar(url, destino)
        except Exception as exc:
            erros.append(f"{Path(destino).name}: {exc}")
    return erros
