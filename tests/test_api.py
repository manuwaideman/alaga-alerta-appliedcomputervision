"""Testes ponta a ponta da API com um detector falso (não precisa de pesos YOLO nem GPU)."""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from alaga_vision.anonymizer import RelatorioAnonimizacao
from alaga_vision.api import criar_app
from alaga_vision.config import Settings
from alaga_vision.detector import Deteccao

LAT, LON = -23.5614, -46.6559  # Av. Paulista


class DetectorFalso:
    classes = ["rua_alagada", "bueiro_entupido", "veiculo_submerso"]
    descricao = "falso"

    def __init__(self, deteccoes: list[Deteccao]):
        self.deteccoes = deteccoes
        self.chamadas = 0

    def detect(self, img):
        self.chamadas += 1
        return self.deteccoes


class AnonimizadorFalso:
    metodo_rostos = metodo_placas = "falso"

    def __init__(self):
        self.chamadas = 0

    def anonimizar(self, img):
        self.chamadas += 1
        return img.copy(), RelatorioAnonimizacao(1, 2, "falso", "falso")


def jpeg(gps: tuple[float, float] | None = None, tamanho=(640, 480)) -> bytes:
    img = Image.new("RGB", tamanho, (90, 110, 130))
    buf = io.BytesIO()
    if gps:
        exif = Image.Exif()
        lat, lon = gps

        def dms(v):
            v = abs(v)
            g, m = int(v), int((abs(v) - int(v)) * 60)
            return (float(g), float(m), round((v - g - m / 60) * 3600, 4))

        exif[0x8825] = {1: "S" if lat < 0 else "N", 2: dms(lat), 3: "W" if lon < 0 else "E", 4: dms(lon)}
        img.save(buf, "JPEG", exif=exif)
    else:
        img.save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def settings(tmp_path):
    return Settings(
        model_path=tmp_path / "nao_existe.pt",
        face_model_path=tmp_path / "nao_existe.onnx",
        plate_model_path=tmp_path / "nao_existe_placa.pt",
        data_dir=tmp_path / "app",
        web_dir=tmp_path / "sem_web",
        auto_download=False,
    )


def cliente(settings, deteccoes=None, sem_modelo=False, sem_anonimizacao=False):
    det = None if sem_modelo else DetectorFalso(deteccoes or [])
    anon = None if sem_anonimizacao else AnonimizadorFalso()
    app = criar_app(settings, detector=det, anonimizador=anon)
    return TestClient(app), det


def enviar(client, dados=None, **form):
    form.setdefault("latitude", LAT)
    form.setdefault("longitude", LON)
    form = {k: v for k, v in form.items() if v is not None}
    return client.post("/api/deteccoes", files={"imagem": ("rua.jpg", dados or jpeg(), "image/jpeg")}, data=form)


def test_sem_modelo_api_sobe_e_responde_503(settings):
    client, _ = cliente(settings, sem_modelo=True)
    with client:
        h = client.get("/api/health").json()
        assert h["status"] == "sem_modelo"
        assert "Pesos não encontrados" in h["erro_modelo"]
        assert enviar(client).status_code == 503
        assert client.get("/api/ocorrencias").json()["features"] == []


def test_sem_anonimizacao_recusa_fotos(settings):
    """Fail-closed: sem os modelos de rosto/placa, nenhuma foto é processada nem salva."""
    client, det = cliente(settings, [Deteccao("rua_alagada", 0.9, (0, 0, 10, 10), 0.1)], sem_anonimizacao=True)
    with client:
        h = client.get("/api/health").json()
        assert h["status"] == "sem_anonimizacao"
        assert h["anonimizacao"]["ativa"] is False
        r = enviar(client)
        assert r.status_code == 503
        assert "anonimização" in r.json()["detail"].lower()
        assert det.chamadas == 0
        assert not settings.media_dir.exists() or list(settings.media_dir.iterdir()) == []


def test_deteccao_confirmada_vira_ocorrencia_no_mapa(settings):
    dets = [Deteccao("rua_alagada", 0.91, (0, 200, 640, 480), 0.58), Deteccao("veiculo_submerso", 0.77, (100, 250, 300, 400), 0.098)]
    client, det = cliente(settings, dets)
    with client:
        r = enviar(client, descricao="água no joelho")
        assert r.status_code == 200, r.text
        corpo = r.json()
        assert corpo["confirmada"] is True
        assert det.chamadas == 1
        oc = corpo["ocorrencia"]
        assert oc["classes"] == ["rua_alagada", "veiculo_submerso"]
        assert corpo["risco"]["nivel"] == "alto"
        assert corpo["localizacao"]["fonte"] == "formulario"
        assert corpo["anonimizacao"] == {"rostos": 1, "placas": 2, "metodo_rostos": "falso", "metodo_placas": "falso"}

        # imagens anonimizadas foram gravadas e são servidas
        assert (settings.media_dir / oc["imagem"]).exists()
        assert client.get(f"/media/{oc['imagem_anotada']}").status_code == 200

        geo = client.get("/api/ocorrencias").json()
        assert geo["type"] == "FeatureCollection"
        [f] = geo["features"]
        assert f["geometry"]["coordinates"] == [LON, LAT]  # GeoJSON é [lon, lat]
        assert f["properties"]["descricao"] == "água no joelho"
        assert f["properties"]["imagem_anotada_url"].endswith(oc["imagem_anotada"])

        assert client.get(f"/api/ocorrencias/{oc['id']}").json()["id"] == oc["id"]
        assert client.get("/api/ocorrencias", params={"nivel_minimo": "alto"}).json()["features"]


def test_sem_deteccao_descarta_imagem(settings):
    client, _ = cliente(settings, [])
    with client:
        corpo = enviar(client).json()
        assert corpo["confirmada"] is False
        assert corpo["ocorrencia"] is None
        assert list(settings.media_dir.iterdir()) == []  # nada armazenado (LGPD)
        assert client.get("/api/ocorrencias").json()["features"] == []


def test_deteccao_abaixo_do_limiar_nao_confirma(settings):
    client, _ = cliente(settings, [Deteccao("bueiro_entupido", 0.30, (10, 10, 50, 50), 0.01)])
    with client:
        corpo = enviar(client).json()
        assert corpo["confirmada"] is False
        assert len(corpo["deteccoes"]) == 1  # aparece como candidata, mas não vai ao mapa


def test_sem_localizacao_retorna_422(settings):
    client, _ = cliente(settings, [])
    with client:
        r = enviar(client, latitude=None, longitude=None)
        assert r.status_code == 422
        assert "latitude" in r.json()["detail"]


def test_localizacao_do_exif_e_exif_removido(settings):
    client, _ = cliente(settings, [Deteccao("rua_alagada", 0.8, (0, 0, 640, 480), 1.0)])
    with client:
        r = enviar(client, jpeg(gps=(LAT, LON)), latitude=None, longitude=None)
        assert r.status_code == 200, r.text
        loc = r.json()["localizacao"]
        assert loc["fonte"] == "exif"
        assert loc["latitude"] == pytest.approx(LAT, abs=1e-3)
        assert loc["longitude"] == pytest.approx(LON, abs=1e-3)
        salvo = Image.open(settings.media_dir / r.json()["ocorrencia"]["imagem"])
        assert not salvo.getexif().get_ifd(0x8825)  # GPS não foi persistido


def test_arquivo_invalido(settings):
    client, _ = cliente(settings, [])
    with client:
        r = enviar(client, b"isto nao e uma imagem")
        assert r.status_code == 422


def test_imagem_grande_e_reduzida(settings):
    client, _ = cliente(settings, [Deteccao("rua_alagada", 0.9, (0, 0, 100, 100), 0.1)])
    with client:
        r = enviar(client, jpeg(tamanho=(4000, 3000)))
        oc = r.json()["ocorrencia"]
        assert max(Image.open(settings.media_dir / oc["imagem"]).size) == settings.max_side
