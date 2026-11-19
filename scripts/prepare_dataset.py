"""Monta o dataset unificado do Alaga Alerta a partir de várias fontes.

    python scripts/prepare_dataset.py --inspecionar      # só lista classes e mapeamento
    python scripts/prepare_dataset.py                    # gera data/alaga/ + estatísticas

O que o script faz:
  1. baixa cada fonte (Kaggle, sem login; Roboflow Universe, com ROBOFLOW_API_KEY) para data/raw/<nome>/;
  2. converte polígonos (datasets de segmentação) e máscaras de água em caixas;
  3. gera pseudo-rótulos de "veículo submerso": um YOLO pré-treinado no COCO acha os
     veículos e a máscara de água diz se a base de cada um está dentro d'água;
  4. remapeia as classes de cada fonte para as 3 do projeto (regras em training/datasets.yaml);
  5. remove imagens duplicadas (hash MD5);
  6. divide train/val/test POR FONTE e POR IMAGEM DE ORIGEM: as cópias aumentadas
     que o Roboflow gera (`foto_jpg.rf.<hash>.jpg`) ficam todas no mesmo split,
     evitando vazamento de dados entre treino e validação;
  7. grava data/alaga/data.yaml, data/alaga/stats.json e docs/dataset_stats.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MARCADOR = ".gerado_por_prepare_dataset"


@dataclass
class Item:
    imagem: Path
    fonte: str
    grupo: str
    caixas: list[tuple[int, float, float, float, float]] = field(default_factory=list)


# ---------------------------------------------------------------- utilidades
def rel(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def compilar_regras(mapa: dict | None) -> list[tuple[re.Pattern, str | None]]:
    return [(re.compile(padrao, re.IGNORECASE), destino) for padrao, destino in (mapa or {}).items()]


def mapear(nome: str, regras) -> str | None:
    for padrao, destino in regras:
        if padrao.search(nome):
            return destino
    return None


def grupo_de(stem: str) -> str:
    """'IMG_01_jpg.rf.9f8e...' -> 'IMG_01_jpg' (cópias aumentadas do Roboflow compartilham o grupo)."""
    return stem.split(".rf.")[0]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def nomes_do_yaml(pasta: Path) -> list[str]:
    cand = list(pasta.glob("data.yaml")) + list(pasta.glob("*.yaml"))
    if not cand:
        raise FileNotFoundError(f"data.yaml não encontrado em {pasta}")
    names = yaml.safe_load(cand[0].read_text(encoding="utf-8"))["names"]
    return [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)


# ---------------------------------------------------------------- download
def baixar_kaggle(fonte: dict, destino: Path) -> None:
    """Datasets públicos do Kaggle podem ser baixados sem login pela API v1."""
    if (destino / "fonte.json").exists():
        return
    import urllib.request
    import zipfile

    ref = fonte["kaggle"]
    destino.mkdir(parents=True, exist_ok=True)
    zip_path = destino / "_download.zip"
    print(f"  baixando kaggle.com/datasets/{ref} ...")
    req = urllib.request.Request(f"https://www.kaggle.com/api/v1/datasets/download/{ref}",
                                 headers={"User-Agent": "alaga-alerta-dataset/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(zip_path, "wb") as f:
        shutil.copyfileobj(r, f)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(destino)
    zip_path.unlink()
    (destino / "fonte.json").write_text(
        json.dumps({"kaggle": ref, "url": fonte.get("url"), "licenca": fonte.get("licenca")}, indent=2),
        encoding="utf-8",
    )


def baixar_roboflow(fonte: dict, destino: Path) -> bool:
    """Retorna False (fonte pulada) se não houver ROBOFLOW_API_KEY."""
    if (destino / "data.yaml").exists():
        return True
    key = os.getenv("ROBOFLOW_API_KEY")
    if not key:
        print("  ROBOFLOW_API_KEY não definida: fonte pulada "
              "(conta grátis em roboflow.com > Settings > API Keys).")
        return False
    from roboflow import Roboflow

    proj = Roboflow(api_key=key).workspace(fonte["workspace"]).project(fonte["projeto"])
    versao = fonte.get("versao")
    if versao is None:
        versao = max(int(str(v.version).split("/")[-1]) for v in proj.versions())
    print(f"  baixando {fonte['workspace']}/{fonte['projeto']} v{versao} ...")
    proj.version(int(versao)).download("yolov8", location=str(destino), overwrite=True)
    (destino / "fonte.json").write_text(
        json.dumps({**{k: fonte.get(k) for k in ("workspace", "projeto", "url")}, "versao": versao}, indent=2),
        encoding="utf-8",
    )
    return True


# ---------------------------------------------------------------- leitura
def ler_yolo(fonte: dict, pasta: Path, classes: list[str], stats: dict) -> list[Item]:
    nomes = nomes_do_yaml(pasta)
    regras = compilar_regras(fonte.get("mapa"))
    origem_cnt, mapa_usado = Counter(), {}
    itens = []
    for img in sorted(pasta.rglob("*")):
        if img.suffix.lower() not in IMG_EXT or "images" not in img.parts:
            continue
        partes = list(img.parts)
        idx = len(partes) - 1 - partes[::-1].index("images")
        partes[idx] = "labels"
        txt = Path(*partes).with_suffix(".txt")
        item = Item(img, fonte["nome"], f"{fonte['nome']}/{grupo_de(img.stem)}")
        if txt.exists():
            for linha in txt.read_text(encoding="utf-8").splitlines():
                p = linha.split()
                if len(p) < 5:
                    continue
                cls, vals = int(float(p[0])), [float(v) for v in p[1:]]
                nome = nomes[cls] if cls < len(nomes) else str(cls)
                origem_cnt[nome] += 1
                destino = mapear(nome, regras)
                mapa_usado[nome] = destino
                if destino is None:
                    continue
                if len(vals) == 4:
                    xc, yc, w, h = vals
                else:  # polígono de segmentação -> caixa envolvente
                    xs, ys = vals[0::2], vals[1::2]
                    x1, x2, y1, y2 = max(0, min(xs)), min(1, max(xs)), max(0, min(ys)), min(1, max(ys))
                    xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                if w * h < 1e-5:
                    continue
                item.caixas.append((classes.index(destino), xc, yc, w, h))
        itens.append(item)
    stats[fonte["nome"]] = {"classes_origem": dict(origem_cnt), "mapeamento": mapa_usado, "imagens": len(itens),
                            "negativos_disponiveis": sum(1 for it in itens if not it.caixas)}
    return itens


class ContextoCOCO:
    """YOLO pré-treinado no COCO, usado só na preparação do dataset:
    - filtro urbano: a foto tem algum objeto de rua (veículo, semáforo, placa de pare...)?
    - pseudo-rótulo de veículo submerso: veículo cuja base está dentro da máscara de água.
    Resultados ficam em cache (data/raw/<fonte>/_coco_cache.json) para reexecuções rápidas."""

    URBANO = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 9: "traffic light",
              10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench"}
    VEICULOS = {2, 3, 5, 7}

    def __init__(self, pesos: str, conf: float = 0.30):
        self.pesos, self.conf, self._modelo = pesos, conf, None

    def deteccoes(self, img: Path, cache: dict) -> list[list[float]]:
        """[[cls, conf, x1, y1, x2, y2], ...] com coordenadas normalizadas (0-1)."""
        if img.name in cache:
            return cache[img.name]
        if self._modelo is None:
            from ultralytics import YOLO
            pesos = rel(self.pesos)
            pesos.parent.mkdir(parents=True, exist_ok=True)
            self._modelo = YOLO(str(pesos))
        r = self._modelo.predict(str(img), conf=self.conf, classes=list(self.URBANO), imgsz=640, verbose=False)[0]
        H, W = r.orig_shape
        out = [[int(c), round(float(p), 3), x1 / W, y1 / H, x2 / W, y2 / H]
               for c, p, (x1, y1, x2, y2) in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
        cache[img.name] = out
        return out


def veiculos_submersos(dets, agua, limiar: float, conf_min: float = 0.40, area_min: float = 0.002):
    """Veículo "submerso" = a faixa logo abaixo da base da caixa (de 10% acima a 15% abaixo da
    borda inferior, alargada 10% para os lados) é majoritariamente água. Num carro submerso a
    caixa termina na linha d'água e o que está embaixo é água; num carro em chão seco, é asfalto.
    (A máscara não cobre a lataria, por isso medir DENTRO da caixa subestima a água.)
    Critério da classe: água pelo menos na altura das rodas."""
    H, W = agua.shape
    caixas = []
    for c, conf, x1, y1, x2, y2 in dets:
        if c not in ContextoCOCO.VEICULOS or conf < conf_min or (x2 - x1) * (y2 - y1) < area_min:
            continue
        w, h = x2 - x1, y2 - y1
        fy1, fy2 = int((y2 - 0.10 * h) * H), int(min(1.0, y2 + 0.15 * h) * H)
        fx1, fx2 = int(max(0.0, x1 - 0.10 * w) * W), int(min(1.0, x2 + 0.10 * w) * W)
        faixa = agua[fy1:max(fy2, fy1 + 1), fx1:max(fx2, fx1 + 1)]
        if faixa.size and faixa.mean() >= limiar:
            caixas.append(((x1 + x2) / 2, (y1 + y2) / 2, w, h))
    return caixas


def ler_mascaras(fonte: dict, classes: list[str], stats: dict, coco: ContextoCOCO) -> list[Item]:
    """Fonte de segmentação de água (imagem + máscara binária).

    Campos no YAML (globs relativos à raiz do projeto):
      imagens / mascaras    globs das fotos alagadas e das máscaras
      chave_mascara         {regex: troca} aplicado ao nome da máscara p/ achar a foto (ex. label_1 -> image_1)
      limiar                pixel de água = maior canal > limiar (máscaras 0/1, 0/255 ou coloridas)
      negativos             glob de fotos SEM alagamento (entram como fundo, sem caixa)
      filtro_urbano         só usa fotos alagadas com algum objeto de rua (descarta rios, vistas aéreas)
      veiculos_submersos    gera pseudo-rótulos de veiculo_submerso (limiar = fração de água na base)
      max_imagens           amostra no máx. N fotos alagadas (semente fixa)
    """
    import cv2
    import numpy as np

    nome = fonte["nome"]
    imagens = {p.stem: p for p in sorted(ROOT.glob(fonte["imagens"])) if p.suffix.lower() in IMG_EXT}
    if not imagens:
        sys.exit(f"[{nome}] nenhuma imagem em {fonte['imagens']}")
    trocas = [(re.compile(k), v) for k, v in (fonte.get("chave_mascara") or {}).items()]
    mascaras = {}
    for m in ROOT.glob(fonte["mascaras"]):
        chave = m.stem
        for padrao, troca in trocas:
            chave = padrao.sub(troca, chave)
        mascaras[chave] = m

    cls_agua = classes.index(fonte["classe"])
    cls_veic = classes.index("veiculo_submerso")
    area_min = float(fonte.get("area_min", 0.02))
    limiar = int(fonte.get("limiar", 127))
    filtro = bool(fonte.get("filtro_urbano"))
    submersos = fonte.get("veiculos_submersos")
    kernel = np.ones((7, 7), np.uint8)

    cache_path = rel(fonte.get("cache", f"data/raw/{nome}/_coco_cache.json"))
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    n_cache = len(cache)

    candidatas = sorted(k for k in imagens if k in mascaras)
    if fonte.get("max_imagens") and len(candidatas) > fonte["max_imagens"] and not filtro:
        candidatas = sorted(random.Random(42).sample(candidatas, fonte["max_imagens"]))

    itens, cnt = [], Counter()
    for i, stem in enumerate(candidatas, 1):
        img = imagens[stem]
        if i % 250 == 0:
            print(f"    {i}/{len(candidatas)}")
            cache_path.write_text(json.dumps(cache), encoding="utf-8")
        dets = coco.deteccoes(img, cache) if (filtro or submersos) else []
        if filtro and not dets:
            cnt["descartada_sem_contexto_urbano"] += 1
            continue
        m = cv2.imread(str(mascaras[stem]), cv2.IMREAD_UNCHANGED)
        foto = cv2.imread(str(img))
        if m is None or foto is None:
            continue
        if m.ndim == 3:
            m = m.max(axis=2)
        H, W = foto.shape[:2]
        agua = cv2.resize((m > limiar).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        agua_fechada = cv2.morphologyEx(agua, cv2.MORPH_CLOSE, kernel)
        n, _, st, _ = cv2.connectedComponentsWithStats(agua_fechada)
        item = Item(img, nome, f"{nome}/{stem}")
        for j in range(1, n):
            x, y, w, h, area = st[j]
            if area / (H * W) >= area_min:
                item.caixas.append((cls_agua, (x + w / 2) / W, (y + h / 2) / H, w / W, h / H))
        if not item.caixas:  # foto marcada como alagada mas sem mancha relevante: ambígua, fica fora
            cnt["descartada_mascara_pequena"] += 1
            continue
        cnt["mascara"] += len(item.caixas)
        if submersos:
            for xc, yc, w, h in veiculos_submersos(dets, agua, float(submersos)):
                item.caixas.append((cls_veic, xc, yc, w, h))
                cnt["veiculo_pseudo_rotulo"] += 1
        itens.append(item)

    if fonte.get("max_imagens") and filtro and len(itens) > fonte["max_imagens"]:
        itens = sorted(random.Random(42).sample(itens, fonte["max_imagens"]), key=lambda it: it.imagem.name)
        cnt["mascara"] = sum(1 for it in itens for c, *_ in it.caixas if c == cls_agua)
        cnt["veiculo_pseudo_rotulo"] = sum(1 for it in itens for c, *_ in it.caixas if c == cls_veic)

    negativos = [p for p in sorted(ROOT.glob(fonte["negativos"])) if p.suffix.lower() in IMG_EXT] \
        if fonte.get("negativos") else []
    itens += [Item(p, nome, f"{nome}/{p.stem}") for p in negativos]
    if len(cache) != n_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache), encoding="utf-8")

    mapeamento = {"mascara": fonte["classe"]}
    if submersos:
        mapeamento["veiculo_pseudo_rotulo"] = "veiculo_submerso"
    stats[nome] = {
        "classes_origem": {k: v for k, v in cnt.items() if not k.startswith("descartada")},
        "mapeamento": mapeamento,
        "imagens": len(itens),
        "negativos_disponiveis": len(negativos),
        "descartadas": {k: v for k, v in cnt.items() if k.startswith("descartada")},
    }
    return itens


# ---------------------------------------------------------------- divisão
def dividir(itens: list[Item], divisao: dict, semente: int) -> dict[str, list[Item]]:
    """Divide por fonte e por grupo; val/test ficam com 1 imagem por grupo (sem cópias aumentadas)."""
    rnd = random.Random(semente)
    out = {"train": [], "val": [], "test": []}
    por_fonte = defaultdict(lambda: defaultdict(list))
    for it in itens:
        por_fonte[it.fonte][it.grupo].append(it)
    for fonte in sorted(por_fonte):
        grupos = sorted(por_fonte[fonte])
        rnd.shuffle(grupos)
        n = len(grupos)
        n_val = max(1, round(n * divisao["val"])) if n >= 3 else 0
        n_test = max(1, round(n * divisao["test"])) if n >= 3 else 0
        for i, g in enumerate(grupos):
            membros = sorted(por_fonte[fonte][g], key=lambda it: it.imagem.name)
            if i < n_val:
                out["val"].append(membros[0])
            elif i < n_val + n_test:
                out["test"].append(membros[0])
            else:
                out["train"].extend(membros)
    return out


def limitar_fundo(itens: list[Item], max_frac: float, semente: int) -> list[Item]:
    com = [i for i in itens if i.caixas]
    sem = [i for i in itens if not i.caixas]
    random.Random(semente).shuffle(sem)
    return com + sem[: int(len(com) * max_frac)]


# ---------------------------------------------------------------- saída
def escrever(splits: dict[str, list[Item]], saida: Path, classes: list[str],
             repetir: dict[str, int] | None = None) -> None:
    """`repetir`: {fonte: n} grava n cópias de cada imagem de TREINO dessa fonte (classe rara);
    o mosaic/HSV/flip do treino faz cada cópia sair diferente. Val/test nunca são repetidos."""
    if saida.exists():
        if not (saida / MARCADOR).exists():
            sys.exit(f"{saida} existe e não foi gerado por este script; escolha outra --saida.")
        import time
        for tentativa in range(6):  # OneDrive/antivírus travam arquivos por alguns segundos
            try:
                shutil.rmtree(saida)
                break
            except PermissionError:
                if tentativa == 5:
                    raise
                time.sleep(2)
    for split, itens in splits.items():
        (saida / "images" / split).mkdir(parents=True, exist_ok=True)
        (saida / "labels" / split).mkdir(parents=True, exist_ok=True)
        for it in itens:
            n = (repetir or {}).get(it.fonte, 1) if split == "train" and it.caixas else 1
            for k in range(n):
                nome = f"{it.fonte}__{it.imagem.stem}" + (f"__r{k}" if k else "")
                shutil.copy2(it.imagem, saida / "images" / split / f"{nome}{it.imagem.suffix.lower()}")
                linhas = [f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in it.caixas]
                (saida / "labels" / split / f"{nome}.txt").write_text("\n".join(linhas), encoding="utf-8")
    (saida / MARCADOR).touch()
    data_yaml = {
        "path": str(saida.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: c for i, c in enumerate(classes)},
    }
    (saida / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")


def estatisticas(splits, classes, fontes_stats, fontes_cfg) -> dict:
    tabela = {}
    for split, itens in splits.items():
        cnt = Counter(classes[c] for it in itens for c, *_ in it.caixas)
        tabela[split] = {
            "imagens": len(itens),
            "imagens_sem_caixa": sum(1 for it in itens if not it.caixas),
            "instancias": {c: cnt.get(c, 0) for c in classes},
            "por_fonte": dict(Counter(it.fonte for it in itens)),
        }
    return {"classes": classes, "splits": tabela, "fontes": fontes_stats,
            "urls": {f["nome"]: f.get("url") for f in fontes_cfg}}


def markdown(st: dict) -> str:
    classes = st["classes"]
    L = ["# Estatísticas do dataset (gerado por scripts/prepare_dataset.py)", "",
         "| Split | Imagens | Sem caixa (negativas) | " + " | ".join(classes) + " |",
         "|---|---:|---:|" + "---:|" * len(classes)]
    tot_img = tot_neg = 0
    tot = Counter()
    for split in ("train", "val", "test"):
        s = st["splits"][split]
        tot_img += s["imagens"]
        tot_neg += s["imagens_sem_caixa"]
        tot.update(s["instancias"])
        L.append(f"| {split} | {s['imagens']} | {s['imagens_sem_caixa']} | "
                 + " | ".join(str(s["instancias"][c]) for c in classes) + " |")
    L.append(f"| **total** | **{tot_img}** | **{tot_neg}** | " + " | ".join(f"**{tot[c]}**" for c in classes) + " |")
    L += ["", "Valores das colunas de classe = número de caixas (instâncias).", "", "## Fontes", "",
          "| Fonte | Imagens lidas | Classes originais (instâncias) | Mapeamento | URL |", "|---|---:|---|---|---|"]
    for nome, f in st["fontes"].items():
        orig = ", ".join(f"{k} ({v})" for k, v in sorted(f["classes_origem"].items(), key=lambda kv: -kv[1]))
        usadas = [f"{k} → {v}" for k, v in f["mapeamento"].items() if v]
        descartadas = sum(1 for v in f["mapeamento"].values() if not v)
        mapa = ", ".join(usadas) + (f"; {descartadas} classe(s) descartada(s)" if descartadas else "")
        L.append(f"| {nome} | {f['imagens']} | {orig} | {mapa} | {st['urls'].get(nome) or '-'} |")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main
def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # console do Windows (cp1252)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="training/datasets.yaml")
    ap.add_argument("--saida", default="data/alaga")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--docs", default="docs", help="onde gravar dataset_stats.md")
    ap.add_argument("--inspecionar", action="store_true", help="só baixa e mostra classes/mapeamento")
    args = ap.parse_args()

    cfg = yaml.safe_load(rel(args.config).read_text(encoding="utf-8"))
    classes = cfg["classes"]
    fontes = [f for f in cfg["fontes"] if f.get("ativo", True)]
    fontes_stats: dict = {}
    itens: list[Item] = []
    coco = ContextoCOCO(cfg.get("modelo_contexto", "models/yolo26s.pt"))

    usadas = []
    for fonte in fontes:
        print(f"[{fonte['nome']}] ({fonte['tipo']})")
        if fonte.get("kaggle"):
            baixar_kaggle(fonte, rel(args.raw) / fonte["nome"])
        if fonte["tipo"] == "roboflow":
            pasta = rel(args.raw) / fonte["nome"]
            if not baixar_roboflow(fonte, pasta):
                continue
            itens += ler_yolo(fonte, pasta, classes, fontes_stats)
        elif fonte["tipo"] == "yolo":
            pasta = rel(fonte["caminho"])
            if not pasta.exists():
                print(f"  {pasta} não existe: fonte pulada" + (f" ({fonte['gerar']})" if fonte.get("gerar") else ""))
                continue
            itens += ler_yolo(fonte, pasta, classes, fontes_stats)
        elif fonte["tipo"] == "mascaras":
            itens += ler_mascaras(fonte, classes, fontes_stats, coco)
        else:
            sys.exit(f"tipo de fonte desconhecido: {fonte['tipo']}")
        usadas.append(fonte)
        s = fontes_stats[fonte["nome"]]
        print(f"  {s['imagens']} imagens")
        for nome, n in sorted(s["classes_origem"].items(), key=lambda kv: -kv[1]):
            print(f"    {nome:<30} {n:>6} caixas  ->  {s['mapeamento'].get(nome) or 'DESCARTADA'}")
        for motivo, n in (s.get("descartadas") or {}).items():
            print(f"    ({motivo}: {n} imagens)")
    fontes = usadas

    if args.inspecionar:
        print("\nConfira o mapeamento acima; ajuste training/datasets.yaml e rode sem --inspecionar.")
        return

    # duplicatas exatas (mesma foto publicada em mais de um dataset)
    vistos, unicos = set(), []
    for it in itens:
        h = md5(it.imagem)
        if h not in vistos:
            vistos.add(h)
            unicos.append(it)
    print(f"\n{len(itens) - len(unicos)} duplicatas removidas")

    splits = dividir(unicos, cfg["divisao"], cfg.get("semente", 42))
    splits = {k: limitar_fundo(v, cfg.get("max_fundo", 0.15), cfg.get("semente", 42)) for k, v in splits.items()}
    saida = rel(args.saida)
    repetir = {f["nome"]: int(f["repetir_treino"]) for f in fontes if f.get("repetir_treino")}
    escrever(splits, saida, classes, repetir)

    st = estatisticas(splits, classes, fontes_stats, fontes)
    st["repetir_treino"] = repetir
    (saida / "stats.json").write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")
    md = markdown(st)
    if repetir:
        md += ("\nContagens = imagens únicas. No treino, as fotos positivas destas fontes são gravadas mais de uma "
               "vez para compensar a classe rara: " + ", ".join(f"{k} ×{v}" for k, v in repetir.items()) + ".\n")
    docs = rel(args.docs)
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "dataset_stats.md").write_text(md, encoding="utf-8")
    (docs / "dataset_stats.json").write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + md)
    print(f"Dataset pronto em {saida / 'data.yaml'}")


if __name__ == "__main__":
    main()
