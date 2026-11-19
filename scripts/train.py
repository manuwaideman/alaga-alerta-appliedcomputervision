"""Treina o detector YOLO do Alaga Alerta (fine-tuning a partir de pesos COCO).

    python scripts/train.py                          # padrão: YOLO26s, 100 épocas, 640px
    python scripts/train.py --modelo yolo26n.pt      # versão nano (CPU / borda)

Recomendado rodar no Colab com GPU (notebooks/treino_colab.ipynb). Ao final,
o melhor checkpoint é copiado para models/alaga_yolo.pt, que a API carrega.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "alaga" / "data.yaml"))
    ap.add_argument("--modelo", default="yolo26s.pt", help="pesos base (yolo26n.pt, yolo26s.pt, yolo11s.pt...)")
    ap.add_argument("--epocas", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="-1 = automático pela memória da GPU")
    ap.add_argument("--paciencia", type=int, default=25, help="early stopping sem melhora no mAP de validação")
    ap.add_argument("--device", default=None, help="0 (GPU), cpu, mps... padrão: automático")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--nome", default="alaga")
    ap.add_argument("--projeto", default=str(ROOT / "runs" / "detect"),
                    help="pasta dos treinos (fora do OneDrive/Dropbox evita arquivo travado ao salvar pesos)")
    ap.add_argument("--fracao", type=float, default=1.0, help="fração do treino a usar (teste rápido)")
    ap.add_argument("--saida", default=str(ROOT / "models" / "alaga_yolo.pt"))
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.modelo)
    model.train(
        data=args.data,
        epochs=args.epocas,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.paciencia,
        device=args.device,
        workers=args.workers,
        fraction=args.fracao,
        project=args.projeto,
        name=args.nome,
        seed=42,
        deterministic=True,
        plots=True,
        # Aumentos de dados: padrões do Ultralytics (mosaic, flip horizontal, HSV, escala)
        # + leve rotação/perspectiva, já que fotos de cidadão vêm tortas e de vários ângulos.
        degrees=5.0,
        perspective=0.0005,
    )

    best = Path(model.trainer.best)
    destino = Path(args.saida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, destino)
    info = {
        "pesos": str(destino),
        "run_dir": str(Path(model.trainer.save_dir)),
        "modelo_base": args.modelo,
        "epocas_max": args.epocas,
        "imgsz": args.imgsz,
        "data": args.data,
        "treinado_em": datetime.now(timezone.utc).isoformat(),
    }
    destino.with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\nMelhor modelo copiado para {destino}")
    print(f"Agora rode: python scripts/evaluate.py --pesos {destino}")


if __name__ == "__main__":
    main()
