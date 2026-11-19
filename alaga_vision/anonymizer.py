"""Anonimização (LGPD): borra rostos e placas ANTES de qualquer armazenamento.

Rostos: YuNet (OpenCV Zoo, ONNX), rodado em duas escalas para pegar rostos pequenos.
Placas: YOLOv11n fine-tuned para placas veiculares (ONNX, via onnxruntime).

Política "fail-closed": se os modelos não estiverem disponíveis, o Anonimizador não
é criado e a API recusa novas fotos (503) em vez de guardar imagem sem anonimizar.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

Box = tuple[int, int, int, int]


class AnonimizacaoIndisponivelError(RuntimeError):
    pass


@dataclass
class RelatorioAnonimizacao:
    rostos: int = 0
    placas: int = 0
    metodo_rostos: str = ""
    metodo_placas: str = ""
    regioes: list[Box] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "rostos": self.rostos,
            "placas": self.placas,
            "metodo_rostos": self.metodo_rostos,
            "metodo_placas": self.metodo_placas,
        }


def borrar_regiao(img: np.ndarray, box: Box, padding: float = 0.15) -> None:
    """Pixeliza + desfoca a região in-place. Pixelizar antes do blur torna a reversão inviável."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    pw, ph = int((x2 - x1) * padding), int((y2 - y1) * padding)
    x1, y1 = max(0, x1 - pw), max(0, y1 - ph)
    x2, y2 = min(w, x2 + pw), min(h, y2 + ph)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = img[y1:y2, x1:x2]
    rw, rh = roi.shape[1], roi.shape[0]
    blocos = max(1, min(rw, rh) // 8)
    small = cv2.resize(roi, (max(1, rw // blocos), max(1, rh // blocos)), interpolation=cv2.INTER_LINEAR)
    pix = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    k = max(3, (min(rw, rh) // 4) | 1)  # kernel ímpar proporcional à região
    img[y1:y2, x1:x2] = cv2.GaussianBlur(pix, (k, k), 0)


class Anonimizador:
    metodo_rostos = "yunet"
    metodo_placas = "yolo-placas"

    def __init__(self, face_model: Path, plate_model: Path, face_score: float = 0.6, plate_conf: float = 0.25):
        faltando = [str(p) for p in (face_model, plate_model) if not Path(p).exists()]
        if faltando:
            raise AnonimizacaoIndisponivelError(
                "Modelos de anonimização ausentes: " + ", ".join(faltando)
                + ". Rode `python scripts/download_models.py`."
            )
        from ultralytics import YOLO

        self._yunet = cv2.FaceDetectorYN.create(str(face_model), "", (320, 320), face_score, 0.3, 5000)
        self._placas = YOLO(str(plate_model), task="detect")  # .onnx -> roda via onnxruntime
        self.plate_conf = plate_conf
        self._lock = threading.Lock()

    @classmethod
    def carregar(cls, face_model: Path, plate_model: Path, baixar: bool = True) -> "Anonimizador":
        if baixar:
            from .modelos import garantir

            erros = garantir(face_model, plate_model)
            if erros:
                print("[aviso] falha ao baixar modelos de anonimização: " + "; ".join(erros))
        return cls(face_model, plate_model)

    def detectar_rostos(self, img: np.ndarray) -> list[Box]:
        """YuNet na escala original + 2x: rostos de pedestres em fotos de rua costumam ser pequenos."""
        h, w = img.shape[:2]
        caixas: list[Box] = []
        escalas = [1.0] + ([2.0] if max(h, w) <= 1280 else [])
        with self._lock:
            for s in escalas:
                im = img if s == 1.0 else cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
                self._yunet.setInputSize((im.shape[1], im.shape[0]))
                _, faces = self._yunet.detect(im)
                if faces is not None:
                    caixas += [
                        (int(x / s), int(y / s), int((x + fw) / s), int((y + fh) / s)) for x, y, fw, fh in faces[:, :4]
                    ]
        return _nms(caixas)

    def detectar_placas(self, img: np.ndarray) -> list[Box]:
        with self._lock:
            r = self._placas.predict(img, conf=self.plate_conf, verbose=False)[0]
        return [tuple(int(v) for v in b) for b in r.boxes.xyxy.tolist()]

    def anonimizar(self, img: np.ndarray) -> tuple[np.ndarray, RelatorioAnonimizacao]:
        rostos = self.detectar_rostos(img)
        placas = self.detectar_placas(img)
        out = img.copy()
        for box in rostos + placas:
            borrar_regiao(out, box)
        return out, RelatorioAnonimizacao(
            rostos=len(rostos),
            placas=len(placas),
            metodo_rostos=self.metodo_rostos,
            metodo_placas=self.metodo_placas,
            regioes=rostos + placas,
        )


def _nms(caixas: list[Box], iou_max: float = 0.4) -> list[Box]:
    """Remove caixas duplicadas vindas das duas escalas."""
    if len(caixas) < 2:
        return caixas
    b = np.array(caixas, dtype=np.float32)
    rects = [(float(x1), float(y1), float(x2 - x1), float(y2 - y1)) for x1, y1, x2, y2 in b]
    keep = cv2.dnn.NMSBoxes(rects, [1.0] * len(rects), 0.0, iou_max)
    return [caixas[i] for i in np.array(keep).flatten()]
