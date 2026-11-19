"""Configuração central do serviço de visão do Alaga Alerta.

Tudo pode ser sobrescrito por variáveis de ambiente (prefixo ALAGA_), para que o
mesmo código rode localmente, no Colab e em container sem alterações.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Ordem = índice da classe no modelo YOLO. Não reordene depois de treinar.
CLASSES: list[str] = ["rua_alagada", "bueiro_entupido", "veiculo_submerso"]

ROTULOS: dict[str, str] = {
    "rua_alagada": "Rua alagada",
    "bueiro_entupido": "Bueiro entupido",
    "veiculo_submerso": "Veículo submerso",
}


def _path(env: str, default: Path) -> Path:
    value = os.getenv(env)
    return Path(value).expanduser().resolve() if value else default


def _float(env: str, default: float) -> float:
    value = os.getenv(env)
    return float(value) if value else default


def _int(env: str, default: int) -> int:
    value = os.getenv(env)
    return int(value) if value else default


@dataclass(frozen=True)
class Settings:
    # Pesos do modelo treinado (saída de scripts/train.py).
    model_path: Path = field(default_factory=lambda: _path("ALAGA_MODEL", ROOT / "models" / "alaga_yolo.pt"))
    # Modelos de anonimização (baixados por scripts/download_models.py).
    face_model_path: Path = field(
        default_factory=lambda: _path("ALAGA_FACE_MODEL", ROOT / "models" / "face_detection_yunet_2023mar.onnx")
    )
    plate_model_path: Path = field(
        default_factory=lambda: _path("ALAGA_PLATE_MODEL", ROOT / "models" / "license-plate-finetune-v1n.onnx")
    )
    # Onde ficam o banco SQLite e as imagens anonimizadas.
    data_dir: Path = field(default_factory=lambda: _path("ALAGA_DATA_DIR", ROOT / "data" / "app"))
    web_dir: Path = field(default_factory=lambda: _path("ALAGA_WEB_DIR", ROOT / "web"))

    # Confiança mínima para o YOLO reportar uma caixa.
    conf_threshold: float = field(default_factory=lambda: _float("ALAGA_CONF", 0.25))
    # Confiança mínima para a foto virar ocorrência confirmada no mapa.
    confirm_threshold: float = field(default_factory=lambda: _float("ALAGA_CONFIRM_CONF", 0.40))
    # Tamanho de entrada do YOLO; 0 = o mesmo usado no treino (lido do checkpoint).
    imgsz: int = field(default_factory=lambda: _int("ALAGA_IMGSZ", 0))
    # Lado maior máximo da imagem guardada/processada (reduz custo e armazenamento).
    max_side: int = field(default_factory=lambda: _int("ALAGA_MAX_SIDE", 1920))
    max_upload_mb: int = field(default_factory=lambda: _int("ALAGA_MAX_UPLOAD_MB", 15))
    # Baixa automaticamente os modelos de anonimização na primeira execução.
    auto_download: bool = field(default_factory=lambda: os.getenv("ALAGA_AUTO_DOWNLOAD", "1") != "0")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "alaga.db"

    @property
    def media_dir(self) -> Path:
        return self.data_dir / "media"
