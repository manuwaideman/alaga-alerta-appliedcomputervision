"""Avalia o modelo no split de TESTE e gera as métricas para o PDF e o README.

    python scripts/evaluate.py
    python scripts/evaluate.py --split val --exemplos 12

Saídas:
  docs/metricas.json, docs/metricas.md      mAP50, mAP50-95, precisão, recall, F1 (geral e por classe)
  docs/figuras/                              matriz de confusão, curvas PR/F1, curvas de treino
  docs/figuras/exemplos/                     imagens de teste com as caixas previstas
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGURAS_VAL = [
    "confusion_matrix_normalized.png",
    "confusion_matrix.png",
    "BoxPR_curve.png",
    "BoxF1_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "PR_curve.png",
    "F1_curve.png",
]
FIGURAS_TREINO = ["results.png", "labels.jpg"]


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def escolher_exemplos(imgs: list[Path], n_classes: int, n: int) -> list[Path]:
    """Exemplos variados: imagens de cada classe (pelo rótulo verdadeiro) em rodízio, mais uma negativa.
    Prefere fotos com pelo menos 360 px de largura (as de 224 px ficam ilegíveis no PDF)."""
    from PIL import Image

    def rotulos(img: Path) -> set[int]:
        txt = Path(str(img.parent).replace("images", "labels")) / f"{img.stem}.txt"
        linhas = txt.read_text(encoding="utf-8").split("\n") if txt.exists() else []
        return {int(float(l.split()[0])) for l in linhas if l.strip()}

    def largura(img: Path) -> int:
        with Image.open(img) as im:
            return im.width

    grandes = [i for i in imgs if largura(i) >= 360] or imgs
    por_classe = [[i for i in grandes if c in rotulos(i)] for c in range(n_classes)]
    negativas = [i for i in grandes if not rotulos(i)]
    filas = [lista[:: max(1, len(lista) // max(1, n // (n_classes + 1)))] for lista in por_classe if lista]
    escolhidas: list[Path] = []
    while len(escolhidas) < n - (1 if negativas else 0) and any(filas):
        for fila in filas:
            if fila and len(escolhidas) < n - (1 if negativas else 0):
                img = fila.pop(0)
                if img not in escolhidas:
                    escolhidas.append(img)
    if negativas:
        escolhidas.append(negativas[len(negativas) // 2])
    return escolhidas


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # console do Windows (cp1252)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pesos", default=str(ROOT / "models" / "alaga_yolo.pt"))
    ap.add_argument("--data", default=str(ROOT / "data" / "alaga" / "data.yaml"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--imgsz", type=int, default=0, help="0 = o mesmo do treino (lido do checkpoint)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--exemplos", type=int, default=8, help="quantas imagens de exemplo salvar")
    ap.add_argument("--conf-matriz", type=float, default=0.40,
                    help="limiar da matriz de confusão e dos exemplos (= confirmação da API)")
    ap.add_argument("--docs", default=str(ROOT / "docs"))
    args = ap.parse_args()

    from ultralytics import YOLO

    docs = Path(args.docs)
    figs = docs / "figuras"
    figs.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.pesos)
    args.imgsz = args.imgsz or int(model.overrides.get("imgsz") or 640)
    m = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        plots=True,
        project=str(ROOT / "runs" / "val"),
        name=args.split,
        exist_ok=True,
    )
    box = m.box
    geral = {
        "mAP50": float(box.map50),
        "mAP50-95": float(box.map),
        "precisao": float(box.mp),
        "recall": float(box.mr),
    }
    geral["f1"] = f1(geral["precisao"], geral["recall"])
    por_classe = {}
    for i, c in enumerate(m.ap_class_index):
        p, r, ap50, ap5095 = (float(v) for v in box.class_result(i))
        por_classe[m.names[int(c)]] = {"precisao": p, "recall": r, "f1": f1(p, r), "mAP50": ap50, "mAP50-95": ap5095}

    resultado = {
        "split": args.split,
        "pesos": str(args.pesos),
        "geral": geral,
        "por_classe": por_classe,
        "velocidade_ms_por_imagem": {k: round(float(v), 2) for k, v in m.speed.items()},
    }
    (docs / "metricas.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")

    linhas = [
        f"# Métricas do detector (split: {args.split})",
        "",
        "| Classe | Precisão | Recall | F1 | mAP@50 | mAP@50-95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for i in sorted(m.names):
        nome = m.names[i]
        v = por_classe.get(nome)
        if v is None:  # classe sem nenhuma instância neste split
            linhas.append(f"| {nome} | — | — | — | — | — |")
            continue
        linhas.append(f"| {nome} | {v['precisao']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['mAP50']:.3f} | {v['mAP50-95']:.3f} |")
    g = geral
    linhas.append(f"| **todas** | **{g['precisao']:.3f}** | **{g['recall']:.3f}** | **{g['f1']:.3f}** | **{g['mAP50']:.3f}** | **{g['mAP50-95']:.3f}** |")
    vel = resultado["velocidade_ms_por_imagem"]
    linhas += ["", f"Tempo de inferência: {vel.get('inference', 0):.1f} ms/imagem (pré {vel.get('preprocess', 0):.1f} ms, pós {vel.get('postprocess', 0):.1f} ms)."]
    md = "\n".join(linhas) + "\n"
    (docs / "metricas.md").write_text(md, encoding="utf-8")

    # Figuras de validação e de treino
    copiadas = []
    for nome in FIGURAS_VAL:
        src = Path(m.save_dir) / nome
        if src.exists():
            shutil.copy2(src, figs / nome)
            copiadas.append(nome)
    # A matriz de confusão do Ultralytics usa o `conf` da validação (0,001, necessário para o mAP); com o YOLO26
    # sem NMS isso conta centenas de caixas de baixíssima confiança como falso positivo. Para o relatório, a
    # matriz é refeita no limiar em que a API confirma uma ocorrência.
    mc = model.val(data=args.data, split=args.split, imgsz=args.imgsz, device=args.device, conf=args.conf_matriz,
                   plots=True, project=str(ROOT / "runs" / "val"), name=f"{args.split}_conf", exist_ok=True,
                   verbose=False)
    for nome in ("confusion_matrix_normalized.png", "confusion_matrix.png"):
        src = Path(mc.save_dir) / nome
        if src.exists():
            shutil.copy2(src, figs / nome)
    matriz = mc.confusion_matrix.matrix.astype(int).tolist()
    resultado["matriz_confusao"] = {"conf": args.conf_matriz, "linhas_previsto_colunas_real": matriz,
                                    "rotulos": [m.names[i] for i in sorted(m.names)] + ["fundo"]}

    # Visão do app: a FOTO vira ocorrência confirmada? (alguma caixa com confiança >= limiar da API)
    import yaml

    cfg = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
    pasta_split = Path(cfg["path"]) / cfg[args.split]
    fotos = sorted(p for p in pasta_split.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    cont = {"positivas": 0, "positivas_confirmadas": 0, "negativas": 0, "negativas_com_alarme": 0}
    for foto in fotos:
        txt = Path(str(foto.parent).replace("images", "labels")) / f"{foto.stem}.txt"
        positiva = txt.exists() and txt.read_text(encoding="utf-8").strip() != ""
        r = model.predict(str(foto), imgsz=args.imgsz, conf=args.conf_matriz, device=args.device, verbose=False)[0]
        confirmou = len(r.boxes) > 0
        if positiva:
            cont["positivas"] += 1
            cont["positivas_confirmadas"] += confirmou
        else:
            cont["negativas"] += 1
            cont["negativas_com_alarme"] += confirmou
    resultado["por_foto"] = {**cont, "conf": args.conf_matriz}
    (docs / "metricas.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    md += (f"\nPor foto (confiança ≥ {args.conf_matriz:.2f}, como na API): {cont['positivas_confirmadas']} de "
           f"{cont['positivas']} fotos com alagamento/bueiro viraram ocorrência; {cont['negativas_com_alarme']} de "
           f"{cont['negativas']} fotos sem nada geraram alarme falso.\n")
    (docs / "metricas.md").write_text(md, encoding="utf-8")

    info = Path(args.pesos).with_suffix(".json")
    if info.exists():
        run_dir = Path(json.loads(info.read_text(encoding="utf-8"))["run_dir"])
        for nome in FIGURAS_TREINO:
            if (run_dir / nome).exists():
                shutil.copy2(run_dir / nome, figs / f"treino_{nome}")
                copiadas.append(f"treino_{nome}")

    # Exemplos qualitativos para o PDF / vídeo
    if args.exemplos:
        import yaml

        cfg = yaml.safe_load(Path(args.data).read_text(encoding="utf-8"))
        pasta = Path(cfg["path"]) / cfg[args.split]
        imgs = sorted(p for p in pasta.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        ex_dir = figs / "exemplos"
        if ex_dir.exists():
            shutil.rmtree(ex_dir)
        ex_dir.mkdir()
        anon = None
        try:  # mesmo fluxo da API: anonimiza (rostos/placas) antes de detectar
            sys.path.insert(0, str(ROOT))
            from alaga_vision.anonymizer import Anonimizador
            from alaga_vision.config import Settings

            cfg_app = Settings()
            anon = Anonimizador.carregar(cfg_app.face_model_path, cfg_app.plate_model_path, baixar=False)
        except Exception as e:  # noqa: BLE001 - sem os modelos de anonimização, segue sem
            print(f"[aviso] exemplos sem anonimização: {e}")
        import cv2

        for img in escolher_exemplos(imgs, len(m.names), args.exemplos):
            foto = cv2.imread(str(img))
            if anon is not None:
                foto, _ = anon.anonimizar(foto)
            r = model.predict(foto, imgsz=args.imgsz, conf=args.conf_matriz, device=args.device, verbose=False)[0]
            r.save(filename=str(ex_dir / f"{img.stem}.jpg"))

    print("\n" + md)
    print(f"Figuras copiadas para {figs}: {', '.join(copiadas) or 'nenhuma'}")


if __name__ == "__main__":
    main()
