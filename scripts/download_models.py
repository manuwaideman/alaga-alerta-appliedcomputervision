"""Baixa os modelos de anonimização (rostos e placas) para models/.

    python scripts/download_models.py

A API também tenta baixá-los sozinha na primeira execução; este script é útil
para preparar o ambiente offline (ex.: antes de gravar o vídeo).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alaga_vision.config import Settings  # noqa: E402
from alaga_vision.modelos import garantir  # noqa: E402


def main() -> int:
    s = Settings()
    erros = garantir(s.face_model_path, s.plate_model_path)
    for p in (s.face_model_path, s.plate_model_path):
        status = f"{p.stat().st_size / 1e6:.1f} MB" if p.exists() else "AUSENTE"
        print(f"{p.name:<40} {status}")
    for e in erros:
        print(f"[erro] {e}")
    return 1 if erros else 0


if __name__ == "__main__":
    sys.exit(main())
