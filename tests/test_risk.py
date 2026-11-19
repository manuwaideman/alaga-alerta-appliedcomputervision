from alaga_vision.detector import Deteccao
from alaga_vision.risk import avaliar_risco, cobertura

SHAPE = (1000, 1000)


def det(classe, conf, bbox=(0, 0, 100, 100)):
    x1, y1, x2, y2 = bbox
    return Deteccao(classe, conf, bbox, (x2 - x1) * (y2 - y1) / 1e6)


def test_sem_deteccao():
    r = avaliar_risco([], SHAPE)
    assert (r.score, r.nivel) == (0, "nenhum")


def test_veiculo_submerso_e_risco_alto():
    r = avaliar_risco([det("veiculo_submerso", 0.9)], SHAPE)
    assert (r.score, r.nivel) == (90, "alto")


def test_bueiro_entupido_e_risco_preventivo():
    r = avaliar_risco([det("bueiro_entupido", 0.5)], SHAPE)
    assert (r.score, r.nivel) == (20, "baixo")


def test_rua_alagada_cresce_com_area_coberta():
    pequena = avaliar_risco([det("rua_alagada", 0.8, (0, 0, 200, 200))], SHAPE)  # 4% da imagem
    grande = avaliar_risco([det("rua_alagada", 0.8, (0, 300, 1000, 1000))], SHAPE)  # 70% da imagem
    assert pequena.nivel == "medio"
    assert grande.score == 72 and grande.nivel == "alto"
    assert grande.score > pequena.score


def test_varias_evidencias_somam_bonus():
    so_rua = avaliar_risco([det("rua_alagada", 0.6)], SHAPE)
    rua_e_bueiro = avaliar_risco([det("rua_alagada", 0.6), det("bueiro_entupido", 0.6)], SHAPE)
    assert rua_e_bueiro.score == so_rua.score + 5


def test_cobertura_nao_conta_sobreposicao_duas_vezes():
    a = det("rua_alagada", 0.9, (0, 0, 500, 1000))
    b = det("rua_alagada", 0.9, (0, 0, 500, 1000))
    assert cobertura([a, b], "rua_alagada", SHAPE) == 0.5
