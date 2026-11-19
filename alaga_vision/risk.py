"""Converte as detecções de uma foto em um score de risco (0-100) para o mapa.

Regra (simples e explicável, documentada no README):
- veiculo_submerso: peso 1.00 -> água já atinge a altura de um carro, risco direto à vida.
- rua_alagada:      peso 0.60 a 0.90 -> cresce com a fração da imagem coberta por água.
- bueiro_entupido:  peso 0.40 -> risco preventivo (ainda não alagou, mas vai acumular).
score = max(peso x confiança) + 5 pontos por classe distinta adicional, limitado a 100.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .detector import Deteccao

PESOS = {"veiculo_submerso": 1.0, "rua_alagada": 0.6, "bueiro_entupido": 0.4}
PESO_RUA_MAX = 0.9
COBERTURA_SATURACAO = 0.4  # 40% da imagem coberta de água já conta como peso máximo
LIMIAR_ALTO = 60
LIMIAR_MEDIO = 30


@dataclass(frozen=True)
class Risco:
    score: int
    nivel: str  # nenhum | baixo | medio | alto
    motivo: str

    def to_dict(self) -> dict:
        return {"score": self.score, "nivel": self.nivel, "motivo": self.motivo}


def cobertura(deteccoes: list[Deteccao], classe: str, shape: tuple[int, int], grade: int = 100) -> float:
    """Fração da imagem coberta pela UNIÃO das caixas de uma classe (caixas sobrepostas não somam 2x)."""
    h, w = shape
    mask = np.zeros((grade, grade), dtype=bool)
    for d in deteccoes:
        if d.classe != classe:
            continue
        x1, y1, x2, y2 = d.bbox
        mask[int(y1 / h * grade) : int(np.ceil(y2 / h * grade)), int(x1 / w * grade) : int(np.ceil(x2 / w * grade))] = True
    return float(mask.mean())


def nivel_de(score: int) -> str:
    if score <= 0:
        return "nenhum"
    if score >= LIMIAR_ALTO:
        return "alto"
    if score >= LIMIAR_MEDIO:
        return "medio"
    return "baixo"


def avaliar_risco(deteccoes: list[Deteccao], shape: tuple[int, int]) -> Risco:
    if not deteccoes:
        return Risco(0, "nenhum", "Nenhum sinal de alagamento detectado.")

    cob_rua = cobertura(deteccoes, "rua_alagada", shape)
    peso_rua = PESOS["rua_alagada"] + (PESO_RUA_MAX - PESOS["rua_alagada"]) * min(1.0, cob_rua / COBERTURA_SATURACAO)

    melhor, origem = 0.0, deteccoes[0]
    for d in deteccoes:
        peso = peso_rua if d.classe == "rua_alagada" else PESOS.get(d.classe, 0.5)
        s = peso * d.confianca
        if s > melhor:
            melhor, origem = s, d

    distintas = len({d.classe for d in deteccoes})
    score = int(round(min(1.0, melhor + 0.05 * (distintas - 1)) * 100))
    motivo = f"{origem.rotulo} ({origem.confianca:.0%} de confiança)"
    if cob_rua > 0:
        motivo += f"; água cobre ~{cob_rua:.0%} da imagem"
    if distintas > 1:
        motivo += f"; {distintas} tipos de evidência na mesma foto"
    return Risco(score, nivel_de(score), motivo + ".")
