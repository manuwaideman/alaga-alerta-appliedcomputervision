"""Integração com a API central do MVP: ocorrência confirmada vira POST /deteccoes no bairro mais próximo."""
from __future__ import annotations

import json

import httpx
from fastapi.testclient import TestClient

from alaga_vision.api import criar_app
from alaga_vision.detector import Deteccao
from alaga_vision.mvp import PublicadorMVP
from test_api import LAT, LON, AnonimizadorFalso, DetectorFalso, jpeg, settings  # noqa: F401

CHAVE = "chave-secreta-do-mvp"
BAIRROS = [
    {"id": "pinheiros", "nome": "Pinheiros", "lat": -23.5670, "lon": -46.6920},
    {"id": "se-anhangabau", "nome": "Sé / Vale do Anhangabaú", "lat": -23.5470, "lon": -46.6370},
    {"id": "penha", "nome": "Penha", "lat": -23.5240, "lon": -46.5420},
]


class MVPFalso:
    def __init__(self, status=201, fora=False):
        self.status, self.fora, self.recebidos, self.chaves = status, fora, [], []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        if self.fora:
            raise httpx.ConnectError("recusado", request=req)
        if req.method == "GET" and req.url.path == "/bairros":
            return httpx.Response(200, json=BAIRROS)
        if req.method == "POST" and req.url.path == "/deteccoes":
            self.chaves.append(req.headers.get("x-api-key"))
            self.recebidos.append(json.loads(req.content))
            return httpx.Response(self.status, json={"registradas": []})
        return httpx.Response(404)


def publicador(mvp: MVPFalso) -> PublicadorMVP:
    return PublicadorMVP("http://mvp.teste", CHAVE, httpx.Client(transport=httpx.MockTransport(mvp)))


def enviar(client, lat=LAT, lon=LON):
    return client.post(
        "/api/deteccoes",
        files={"imagem": ("rua.jpg", jpeg(), "image/jpeg")},
        data={"latitude": str(lat), "longitude": str(lon)},
    )


def app_com(settings, deteccoes, mvp):  # noqa: F811
    return criar_app(settings, DetectorFalso(deteccoes), AnonimizadorFalso(), publicador(mvp))


def test_ocorrencia_confirmada_vai_para_o_mvp_no_bairro_mais_proximo(settings):  # noqa: F811
    mvp = MVPFalso()
    with TestClient(app_com(settings, [Deteccao("veiculo_submerso", 0.87, [1, 2, 3, 4], 0.1)], mvp)) as c:
        r = enviar(c).json()  # Av. Paulista: mais perto da Sé do que de Pinheiros e da Penha
    assert r["mvp"] == {"publicada": True, "bairro_id": "se-anhangabau", "bairro": "Sé / Vale do Anhangabaú"}
    corpo = mvp.recebidos[0]
    assert corpo["bairro_id"] == "se-anhangabau" and corpo["imagem_anonimizada"] is True
    assert corpo["deteccoes"] == [{"classe": "veiculo_submerso", "confianca": 0.87}]
    assert mvp.chaves == [CHAVE]


def test_foto_sem_alagamento_nao_vai_para_o_mvp(settings):  # noqa: F811
    mvp = MVPFalso()
    with TestClient(app_com(settings, [Deteccao("rua_alagada", 0.2, [1, 2, 3, 4], 0.1)], mvp)) as c:
        r = enviar(c).json()
    assert r["confirmada"] is False and "mvp" not in r and mvp.recebidos == []


def test_longe_dos_bairros_nao_publica(settings):  # noqa: F811
    mvp = MVPFalso()
    with TestClient(app_com(settings, [Deteccao("rua_alagada", 0.9, [1, 2, 3, 4], 0.3)], mvp)) as c:
        r = enviar(c, lat=-22.9, lon=-43.2).json()  # Rio de Janeiro
    assert r["confirmada"] is True and r["mvp"]["publicada"] is False and mvp.recebidos == []


def test_mvp_fora_do_ar_nao_derruba_a_analise(settings):  # noqa: F811
    with TestClient(app_com(settings, [Deteccao("rua_alagada", 0.9, [1, 2, 3, 4], 0.3)], MVPFalso(fora=True))) as c:
        resp = enviar(c)
    assert resp.status_code == 200
    r = resp.json()
    assert r["confirmada"] is True and r["ocorrencia"] is not None
    assert r["mvp"] == {"publicada": False, "motivo": "MVP indisponível"}


def test_chave_recusada_nao_vaza_a_chave(settings):  # noqa: F811
    with TestClient(app_com(settings, [Deteccao("rua_alagada", 0.9, [1, 2, 3, 4], 0.3)], MVPFalso(status=401))) as c:
        r = enviar(c).json()
    assert r["mvp"] == {"publicada": False, "motivo": "MVP respondeu HTTP 401"}
    assert CHAVE not in json.dumps(r)


def test_sem_variaveis_de_ambiente_integracao_desligada(monkeypatch):
    monkeypatch.delenv("MVP_URL", raising=False)
    monkeypatch.delenv("MVP_API_KEY", raising=False)
    assert PublicadorMVP.do_ambiente() is None
