"""Fotos de bueiro/boca de lobo do Wikimedia Commons para a classe bueiro_entupido.

Não existe dataset público (sem login) de bueiro entupido. As fotos vêm do Wikimedia
Commons (licenças livres: CC0, CC BY, CC BY-SA, domínio público) e foram revisadas uma a uma:

    entupido  folhas, lixo, terra ou água parada tapando a grade/abertura -> caixa bueiro_entupido
    limpo     bueiro desobstruído -> imagem negativa (ensina o modelo que bueiro limpo não é alerta)

A revisão fica versionada em training/bueiros_commons.csv (arquivo, rótulo, caixas, autor,
licença), então qualquer pessoa reproduz o mesmo conjunto:

    python scripts/coletar_bueiros.py                 # baixa as fotos do CSV -> data/raw/bueiros_commons
    python scripts/coletar_bueiros.py --candidatos    # (curadoria) lista candidatas -> data/raw/bueiros_candidatos.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "https://commons.wikimedia.org/w/api.php"
UA = "AlagaAlertaDataset/1.0 (projeto academico; coleta de fotos de bueiros com licenca livre)"
CSV_PADRAO = ROOT / "training" / "bueiros_commons.csv"
SAIDA_PADRAO = ROOT / "data" / "raw" / "bueiros_commons"

CATEGORIAS = [
    "Storm drains", "Storm drains in Brazil", "Drainage pits", "Storm drains in the United States",
    "Storm drains in Germany", "Storm drains in the United Kingdom", "Storm drains in Japan",
    "Storm drains in Australia", "Storm drains in France", "Storm drains in the Netherlands",
    "Gully gratings", "Drain covers",
]
BUSCAS = [
    "clogged drain", "blocked drain", "storm drain leaves", "storm drain litter", "storm drain trash",
    "drain debris", "drain blocked leaves", "gully grate leaves", "bueiro", "boca de lobo",
    "storm drain flooding", "street drain", "leaves drain grate", "drain clogged",
]
LICENCAS_OK = re.compile(r"^(cc0|cc[ -]by(-sa)?[ -]?\d|public domain|pd)", re.I)


def api(params: dict, tentativas: int = 5) -> dict:
    params = {**params, "format": "json", "maxlag": 5}
    url = API + "?" + urllib.parse.urlencode(params)
    for i in range(tentativas):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                dados = json.loads(r.read().decode("utf-8"))
            if "error" in dados and dados["error"].get("code") == "maxlag":
                raise RuntimeError("maxlag")
            time.sleep(1.0)  # educação com a API pública
            return dados
        except Exception as e:  # noqa: BLE001 - rede/limite de taxa: espera e tenta de novo
            espera = 5 * (i + 1)
            print(f"  aviso: {e}; nova tentativa em {espera}s", file=sys.stderr)
            time.sleep(espera)
    raise RuntimeError(f"falha na API do Commons: {url}")


def info_arquivos(titulos: list[str], largura: int = 1024) -> dict[str, dict]:
    """Metadados (URL redimensionada, autor, licença) de até 50 arquivos por chamada."""
    out = {}
    for i in range(0, len(titulos), 50):
        lote = titulos[i:i + 50]
        d = api({"action": "query", "titles": "|".join(lote), "prop": "imageinfo",
                 "iiprop": "url|extmetadata|size|mime", "iiurlwidth": largura})
        for p in d.get("query", {}).get("pages", {}).values():
            ii = (p.get("imageinfo") or [None])[0]
            if not ii or ii.get("mime") not in ("image/jpeg", "image/png", "image/webp"):
                continue
            meta = ii.get("extmetadata", {})
            limpar = lambda s: re.sub(r"<[^>]+>", "", s or "").strip()  # noqa: E731
            out[p["title"]] = {
                "arquivo": p["title"],
                "thumb": ii.get("thumburl") or ii["url"],
                "pagina": ii.get("descriptionurl", ""),
                "autor": limpar(meta.get("Artist", {}).get("value"))[:120],
                "licenca": limpar(meta.get("LicenseShortName", {}).get("value")),
                "largura": ii.get("width"), "altura": ii.get("height"),
            }
    return out


def candidatos(saida_csv: Path) -> None:
    titulos: dict[str, str] = {}
    for cat in CATEGORIAS:
        cont = None
        while True:
            q = {"action": "query", "list": "categorymembers", "cmtitle": f"Category:{cat}",
                 "cmtype": "file", "cmlimit": 500}
            if cont:
                q["cmcontinue"] = cont
            d = api(q)
            for m in d.get("query", {}).get("categorymembers", []):
                titulos.setdefault(m["title"], f"categoria:{cat}")
            cont = d.get("continue", {}).get("cmcontinue")
            if not cont:
                break
        print(f"  categoria {cat}: {len(titulos)} acumulados")
    for termo in BUSCAS:
        d = api({"action": "query", "list": "search", "srsearch": termo, "srnamespace": 6, "srlimit": 100})
        for m in d.get("query", {}).get("search", []):
            titulos.setdefault(m["title"], f"busca:{termo}")
        print(f"  busca '{termo}': {len(titulos)} acumulados")

    infos = info_arquivos(sorted(titulos))
    linhas = [{**v, "origem": titulos[k]} for k, v in infos.items() if LICENCAS_OK.search(v["licenca"] or "")]
    saida_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(saida_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["arquivo", "origem", "licenca", "autor", "largura", "altura", "thumb", "pagina"])
        w.writeheader()
        w.writerows(linhas)
    print(f"{len(linhas)} candidatas com licença livre (de {len(titulos)}) -> {saida_csv}")


def baixar(csv_path: Path, saida: Path, largura: int) -> None:
    linhas = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    infos = info_arquivos([l["arquivo"] for l in linhas], largura)
    (saida / "images").mkdir(parents=True, exist_ok=True)
    (saida / "labels").mkdir(parents=True, exist_ok=True)
    creditos, n_ok = [], 0
    for i, l in enumerate(linhas):
        info = infos.get(l["arquivo"])
        if not info:
            print(f"  indisponível: {l['arquivo']}")
            continue
        stem = f"commons_{i:04d}"
        destino = saida / "images" / f"{stem}.jpg"
        if not destino.exists():
            req = urllib.request.Request(info["thumb"], headers={"User-Agent": UA})
            for t in range(4):
                try:
                    with urllib.request.urlopen(req, timeout=60) as r:
                        destino.write_bytes(r.read())
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"  aviso: {e}; nova tentativa", file=sys.stderr)
                    time.sleep(5 * (t + 1))
            time.sleep(0.5)
        caixas = [c.split() for c in (l.get("caixas") or "").split(";") if c.strip()]
        linhas_yolo = [f"0 {xc} {yc} {w} {h}" for xc, yc, w, h in caixas] if l["rotulo"] == "entupido" else []
        (saida / "labels" / f"{stem}.txt").write_text("\n".join(linhas_yolo), encoding="utf-8")
        creditos.append({"imagem": destino.name, "arquivo": l["arquivo"], "rotulo": l["rotulo"],
                         "autor": info["autor"], "licenca": info["licenca"], "pagina": info["pagina"]})
        n_ok += 1
    (saida / "data.yaml").write_text("names: [bueiro_entupido]\n", encoding="utf-8")
    with open(saida / "creditos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(creditos[0]))
        w.writeheader()
        w.writerows(creditos)
    ent = sum(1 for c in creditos if c["rotulo"] == "entupido")
    print(f"{n_ok} fotos em {saida} ({ent} entupido, {n_ok - ent} limpo/negativo); créditos em creditos.csv")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(CSV_PADRAO), help="revisão versionada (arquivo, rotulo, caixas)")
    ap.add_argument("--saida", default=str(SAIDA_PADRAO))
    ap.add_argument("--largura", type=int, default=500,
                    help="largura das miniaturas (tamanho padrão do Commons; o treino usa 448 px)")
    ap.add_argument("--candidatos", action="store_true", help="lista fotos candidatas para revisão")
    args = ap.parse_args()
    if args.candidatos:
        candidatos(ROOT / "data" / "raw" / "bueiros_candidatos.csv")
    else:
        baixar(Path(args.csv), Path(args.saida), args.largura)


if __name__ == "__main__":
    main()
