import numpy as np
import pytest

from alaga_vision.anonymizer import AnonimizacaoIndisponivelError, Anonimizador, borrar_regiao
from alaga_vision.config import Settings


def xadrez(h=400, w=400, passo=10):
    y, x = np.mgrid[0:h, 0:w]
    v = (((x // passo) + (y // passo)) % 2 * 255).astype(np.uint8)
    return np.dstack([v, v, v])


def test_borrar_regiao_altera_so_a_regiao():
    img = xadrez()
    original = img.copy()
    borrar_regiao(img, (100, 100, 200, 200), padding=0)
    assert not np.array_equal(img[100:200, 100:200], original[100:200, 100:200])
    # detalhe fino (xadrez) desaparece: variância despenca
    assert img[110:190, 110:190].std() < original[110:190, 110:190].std() * 0.5
    # fora da região nada muda
    assert np.array_equal(img[:100], original[:100])
    assert np.array_equal(img[200:], original[200:])


def test_borrar_regiao_na_borda_nao_quebra():
    img = xadrez()
    borrar_regiao(img, (380, 380, 420, 420))  # parcialmente fora da imagem
    borrar_regiao(img, (5, 5, 6, 6))  # minúscula


def test_sem_modelos_falha_fechado(tmp_path):
    with pytest.raises(AnonimizacaoIndisponivelError):
        Anonimizador(tmp_path / "x.onnx", tmp_path / "y.pt")


@pytest.mark.skipif(
    not (Settings().face_model_path.exists() and Settings().plate_model_path.exists()),
    reason="modelos de anonimização não baixados (scripts/download_models.py)",
)
def test_modelos_reais_em_imagem_vazia():
    s = Settings()
    anon = Anonimizador(s.face_model_path, s.plate_model_path)
    img = np.full((480, 640, 3), 127, np.uint8)
    out, rel = anon.anonimizar(img)
    assert (rel.rostos, rel.placas) == (0, 0)
    assert np.array_equal(out, img) and out is not img  # nunca altera a imagem de entrada
