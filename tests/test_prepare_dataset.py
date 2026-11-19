"""Testes da preparação do dataset: máscara -> caixas, filtro urbano, pseudo-rótulo de veículo e repetição."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import prepare_dataset as P  # noqa: E402

CLASSES = ["rua_alagada", "bueiro_entupido", "veiculo_submerso"]


def agua_inferior(h=100, w=100, linha=60):
    """Máscara com água da linha `linha` para baixo."""
    m = np.zeros((h, w), np.uint8)
    m[linha:] = 1
    return m


def test_veiculo_com_base_na_agua_vira_submerso():
    agua = agua_inferior(linha=60)
    # carro (cls 2) cuja caixa termina em y=0.70, dentro da água
    dets = [[2, 0.9, 0.30, 0.45, 0.60, 0.70]]
    assert len(P.veiculos_submersos(dets, agua, limiar=0.4)) == 1


def test_veiculo_em_chao_seco_nao_vira_submerso():
    agua = agua_inferior(linha=80)
    dets = [[2, 0.9, 0.30, 0.30, 0.60, 0.55]]  # base em y=0.55, água só a partir de 0.80
    assert P.veiculos_submersos(dets, agua, limiar=0.4) == []


def test_ignora_baixa_confianca_e_classes_nao_veiculo():
    agua = agua_inferior(linha=50)
    dets = [[2, 0.2, 0.3, 0.45, 0.6, 0.7],   # confiança baixa
            [9, 0.9, 0.3, 0.45, 0.6, 0.7]]   # semáforo
    assert P.veiculos_submersos(dets, agua, limiar=0.4) == []


class CocoFalso:
    """Substitui o YOLO-COCO: devolve detecções fixas por nome de arquivo."""

    def __init__(self, por_arquivo):
        self.por_arquivo = por_arquivo

    def deteccoes(self, img, cache):
        return self.por_arquivo.get(img.name, [])


@pytest.fixture
def fonte_mascaras(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "ROOT", tmp_path)
    (tmp_path / "img").mkdir()
    (tmp_path / "msk").mkdir()
    (tmp_path / "neg").mkdir()
    for nome in ("rua", "rio"):
        cv2.imwrite(str(tmp_path / "img" / f"{nome}.jpg"), np.full((100, 100, 3), 120, np.uint8))
        m = np.zeros((100, 100), np.uint8)
        m[60:, :] = 255
        cv2.imwrite(str(tmp_path / "msk" / f"label_{nome}.png"), m)
    cv2.imwrite(str(tmp_path / "neg" / "seca.jpg"), np.full((100, 100, 3), 90, np.uint8))
    return {
        "nome": "teste", "tipo": "mascaras", "classe": "rua_alagada",
        "imagens": "img/*.jpg", "mascaras": "msk/*.png", "chave_mascara": {"^label_": ""},
        "limiar": 127, "negativos": "neg/*.jpg", "filtro_urbano": True, "veiculos_submersos": 0.4,
        "cache": str(tmp_path / "cache.json"),
    }


def test_ler_mascaras_filtro_urbano_pseudo_rotulo_e_negativos(fonte_mascaras):
    coco = CocoFalso({"rua.jpg": [[2, 0.9, 0.3, 0.45, 0.6, 0.7]]})  # "rio.jpg" sem objeto de rua
    stats = {}
    itens = P.ler_mascaras(fonte_mascaras, CLASSES, stats, coco)
    por_nome = {it.imagem.stem: it for it in itens}
    assert "rio" not in por_nome                              # descartada pelo filtro urbano
    classes = sorted(c for c, *_ in por_nome["rua"].caixas)
    assert classes == [0, 2]                                  # rua_alagada (máscara) + veículo submerso
    xc, yc, w, h = next(b[1:] for b in por_nome["rua"].caixas if b[0] == 0)
    assert yc == pytest.approx(0.8, abs=0.02) and h == pytest.approx(0.4, abs=0.02)
    assert por_nome["seca"].caixas == []                      # negativa entra sem caixa
    assert stats["teste"]["descartadas"]["descartada_sem_contexto_urbano"] == 1
    assert stats["teste"]["negativos_disponiveis"] == 1


def test_escrever_repete_so_positivas_de_treino(tmp_path):
    img = tmp_path / "a.jpg"
    cv2.imwrite(str(img), np.zeros((10, 10, 3), np.uint8))
    pos = P.Item(img, "bueiros", "g1", [(1, 0.5, 0.5, 0.2, 0.2)])
    neg = P.Item(img, "bueiros", "g2", [])
    neg.imagem = tmp_path / "b.jpg"
    cv2.imwrite(str(neg.imagem), np.zeros((10, 10, 3), np.uint8))
    saida = tmp_path / "out"
    P.escrever({"train": [pos, neg], "val": [pos], "test": []}, saida, CLASSES, {"bueiros": 3})
    assert len(list((saida / "images" / "train").iterdir())) == 4   # 3 cópias da positiva + 1 negativa
    assert len(list((saida / "images" / "val").iterdir())) == 1     # validação nunca é repetida
