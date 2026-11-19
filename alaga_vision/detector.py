"""Detector YOLO das três classes do Alaga Alerta."""
from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import CLASSES, ROTULOS


@dataclass(frozen=True)
class Deteccao:
    classe: str
    confianca: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 em pixels da imagem processada
    area_rel: float  # área da caixa / área da imagem (0..1)

    @property
    def rotulo(self) -> str:
        return ROTULOS.get(self.classe, self.classe)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bbox"] = list(self.bbox)
        d["rotulo"] = self.rotulo
        d["confianca"] = round(self.confianca, 4)
        d["area_rel"] = round(self.area_rel, 4)
        return d


class Detector(Protocol):
    """Qualquer objeto com `detect(img_bgr) -> list[Deteccao]` serve (facilita testes)."""

    classes: list[str]
    descricao: str

    def detect(self, img_bgr: np.ndarray) -> list[Deteccao]: ...


class ModeloAusenteError(RuntimeError):
    pass


class YoloDetector:
    def __init__(self, weights: Path, conf: float = 0.25, imgsz: int = 0, device: str | None = None):
        if not Path(weights).exists():
            raise ModeloAusenteError(
                f"Pesos não encontrados em {weights}. Treine com scripts/train.py (ou no Colab) "
                "e copie o best.pt para models/alaga_yolo.pt, ou aponte ALAGA_MODEL para o arquivo."
            )
        from ultralytics import YOLO  # import tardio: a API sobe rápido e os testes não dependem do torch

        self.model = YOLO(str(weights))
        self.conf = conf
        # 0 = mesmo tamanho do treino (o checkpoint guarda os argumentos de treino)
        self.imgsz = imgsz or int(self.model.overrides.get("imgsz") or 640)
        imgsz = self.imgsz
        self.device = device
        names = self.model.names  # {idx: nome}
        self.classes = [names[i] for i in sorted(names)]
        self.descricao = f"YOLO ({Path(weights).name})"
        desconhecidas = set(self.classes) - set(CLASSES)
        if desconhecidas:
            print(f"[aviso] o modelo tem classes fora do padrão do projeto: {sorted(desconhecidas)}")
        # O predict do Ultralytics não é garantidamente thread-safe; a API roda em threadpool.
        self._lock = threading.Lock()
        # Aquecimento: a 1ª inferência é lenta (alocação/fusão de camadas); paga isso na subida.
        self.model.predict(np.zeros((imgsz, imgsz, 3), np.uint8), imgsz=imgsz, device=device, verbose=False)

    def detect(self, img_bgr: np.ndarray) -> list[Deteccao]:
        h, w = img_bgr.shape[:2]
        with self._lock:
            result = self.model.predict(
                img_bgr, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False
            )[0]
        boxes = result.boxes
        out: list[Deteccao] = []
        for (x1, y1, x2, y2), conf, cls in zip(
            boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
        ):
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            area = max(0, x2 - x1) * max(0, y2 - y1) / float(w * h)
            out.append(Deteccao(self.classes[int(cls)], float(conf), (x1, y1, x2, y2), area))
        out.sort(key=lambda d: d.confianca, reverse=True)
        return out
