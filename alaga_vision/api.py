"""API de visão do Alaga Alerta.

Fluxo de POST /api/deteccoes:
    foto -> validação -> anonimização (rostos/placas) -> YOLO -> risco -> ocorrência no mapa

Rodar:  uvicorn alaga_vision.api:app --reload
Docs interativas: http://localhost:8000/docs
"""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .anonymizer import AnonimizacaoIndisponivelError, Anonimizador
from .config import CLASSES, ROTULOS, Settings
from .detector import Detector, ModeloAusenteError, YoloDetector
from .imaging import ImagemInvalidaError, carregar_imagem, desenhar, salvar_jpg
from .mvp import PublicadorMVP
from .risk import avaliar_risco
from .storage import Repositorio

TIPOS_ACEITOS = {"image/jpeg", "image/png", "image/webp", "application/octet-stream"}


def criar_app(
    settings: Settings | None = None,
    detector: Detector | None = None,
    anonimizador: Anonimizador | None = None,
    publicador_mvp: PublicadorMVP | None = None,
) -> FastAPI:
    """Fábrica da aplicação. Os testes injetam um detector falso; em produção carrega o YOLO."""
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.media_dir.mkdir(parents=True, exist_ok=True)
        app.state.repo = Repositorio(settings.db_path)
        app.state.anonimizador, app.state.erro_anonimizacao = anonimizador, None
        if anonimizador is None:
            try:
                app.state.anonimizador = Anonimizador.carregar(
                    settings.face_model_path, settings.plate_model_path, baixar=settings.auto_download
                )
            except AnonimizacaoIndisponivelError as exc:
                app.state.erro_anonimizacao = str(exc)
                print(f"[aviso] {exc}")
        # Integração com a API central do MVP (opcional: MVP_URL + MVP_API_KEY).
        app.state.mvp = publicador_mvp or PublicadorMVP.do_ambiente()
        app.state.detector = detector
        app.state.erro_modelo = None
        if app.state.detector is None:
            try:
                app.state.detector = YoloDetector(settings.model_path, settings.conf_threshold, settings.imgsz)
            except ModeloAusenteError as exc:
                # Sobe mesmo assim: mapa e listagem funcionam, só a detecção responde 503.
                app.state.erro_modelo = str(exc)
                print(f"[aviso] {exc}")
        yield

    app = FastAPI(
        title="Alaga Alerta — Visão",
        version="1.0.0",
        description="Detecção de rua alagada, bueiro entupido e veículo submerso em fotos de cidadãos e câmeras.",
        lifespan=lifespan,
    )
    # O MVP pode estar em outro domínio/porta (front do time); liberar CORS.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    app.mount("/media", StaticFiles(directory=settings.media_dir, check_dir=False), name="media")

    def _publica(oc: dict, base: str) -> dict:
        oc = dict(oc)
        oc["imagem_url"] = f"{base}/media/{oc['imagem']}"
        oc["imagem_anotada_url"] = f"{base}/media/{oc['imagem_anotada']}"
        return oc

    @app.get("/api/health", tags=["status"])
    def health(request: Request):
        st = request.app.state
        det, anon = st.detector, st.anonimizador
        return {
            "status": "ok" if det and anon else ("sem_modelo" if not det else "sem_anonimizacao"),
            "modelo": getattr(det, "descricao", None),
            "classes": getattr(det, "classes", CLASSES),
            "erro_modelo": st.erro_modelo,
            "anonimizacao": {
                "ativa": anon is not None,
                "rostos": getattr(anon, "metodo_rostos", None),
                "placas": getattr(anon, "metodo_placas", None),
                "erro": st.erro_anonimizacao,
            },
            "limiar_confirmacao": settings.confirm_threshold,
            "integracao_mvp": getattr(st.mvp, "url", None),
        }

    @app.post("/api/deteccoes", tags=["detecção"])
    async def detectar(
        request: Request,
        imagem: UploadFile = File(..., description="Foto da rua (JPG/PNG/WEBP)"),
        latitude: float | None = Form(None, ge=-90, le=90),
        longitude: float | None = Form(None, ge=-180, le=180),
        origem: Literal["cidadao", "camera"] = Form("cidadao"),
        camera_id: str | None = Form(None, max_length=64),
        descricao: str | None = Form(None, max_length=500),
    ):
        st = request.app.state
        if st.detector is None:
            raise HTTPException(503, st.erro_modelo or "Modelo indisponível.")
        if st.anonimizador is None:  # fail-closed: sem anonimização, nenhuma foto é processada
            raise HTTPException(503, st.erro_anonimizacao or "Anonimização indisponível.")
        if imagem.content_type and imagem.content_type not in TIPOS_ACEITOS:
            raise HTTPException(415, f"Tipo de arquivo não suportado: {imagem.content_type}")

        dados = await imagem.read(settings.max_upload_mb * 1024 * 1024 + 1)
        if len(dados) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"Imagem maior que {settings.max_upload_mb} MB.")

        t0 = time.perf_counter()
        try:
            img, gps = carregar_imagem(dados, settings.max_side)
        except ImagemInvalidaError as exc:
            raise HTTPException(422, str(exc)) from exc
        del dados  # a original nunca é gravada em disco

        if latitude is not None and longitude is not None:
            local, fonte_local = (latitude, longitude), "formulario"
        elif gps:
            local, fonte_local = gps, "exif"
        else:
            raise HTTPException(422, "Informe latitude e longitude (ou envie uma foto com GPS no EXIF).")

        t1 = time.perf_counter()
        anon, rel_anon = st.anonimizador.anonimizar(img)
        del img
        t2 = time.perf_counter()
        deteccoes = st.detector.detect(anon)
        t3 = time.perf_counter()
        risco = avaliar_risco(deteccoes, anon.shape[:2])

        confirmadas = [d for d in deteccoes if d.confianca >= settings.confirm_threshold]
        resposta = {
            "confirmada": bool(confirmadas),
            "deteccoes": [d.to_dict() for d in deteccoes],
            "risco": risco.to_dict(),
            "anonimizacao": rel_anon.to_dict(),
            "localizacao": {"latitude": local[0], "longitude": local[1], "fonte": fonte_local},
            "ocorrencia": None,
            "tempo_ms": {
                "leitura": round((t1 - t0) * 1000),
                "anonimizacao": round((t2 - t1) * 1000),
                "deteccao": round((t3 - t2) * 1000),
            },
        }
        if not confirmadas:
            # Minimização de dados (LGPD): sem evidência, nada é armazenado.
            resposta["mensagem"] = "Nenhum alagamento confirmado nesta foto; a imagem foi descartada."
            return resposta

        oc_id = uuid.uuid4().hex[:12]
        nome, nome_anot = f"{oc_id}.jpg", f"{oc_id}_anotada.jpg"
        salvar_jpg(anon, settings.media_dir / nome)
        salvar_jpg(desenhar(anon, confirmadas), settings.media_dir / nome_anot)
        risco = avaliar_risco(confirmadas, anon.shape[:2])
        oc = {
            "id": oc_id,
            "criada_em": datetime.now(timezone.utc).isoformat(),
            "lat": local[0],
            "lon": local[1],
            "origem": origem,
            "camera_id": camera_id,
            "descricao": descricao,
            "risco_score": risco.score,
            "risco_nivel": risco.nivel,
            "risco_motivo": risco.motivo,
            "classes": sorted({d.classe for d in confirmadas}),
            "deteccoes": [d.to_dict() for d in confirmadas],
            "anonimizacao": rel_anon.to_dict(),
            "imagem": nome,
            "imagem_anotada": nome_anot,
        }
        st.repo.inserir(oc)
        if st.mvp is not None:
            resposta["mvp"] = st.mvp.publicar(local[0], local[1], oc["deteccoes"])
        resposta["risco"] = risco.to_dict()
        resposta["ocorrencia"] = _publica(oc, str(request.base_url).rstrip("/"))
        resposta["mensagem"] = "Ocorrência confirmada e publicada no mapa."
        return resposta

    @app.get("/api/ocorrencias", tags=["mapa"])
    def listar(
        request: Request,
        horas: float = Query(24, gt=0, le=24 * 30, description="Janela de tempo"),
        nivel_minimo: Literal["baixo", "medio", "alto"] | None = None,
    ):
        """Ocorrências confirmadas em GeoJSON (consumível direto por Leaflet/Mapbox/Google Maps)."""
        base = str(request.base_url).rstrip("/")
        features = []
        for oc in request.app.state.repo.listar(horas, nivel_minimo):
            oc = _publica(oc, base)
            features.append(
                {
                    "type": "Feature",
                    "id": oc["id"],
                    "geometry": {"type": "Point", "coordinates": [oc.pop("lon"), oc.pop("lat")]},
                    "properties": oc,
                }
            )
        return {"type": "FeatureCollection", "features": features}

    @app.get("/api/ocorrencias/{oc_id}", tags=["mapa"])
    def obter(oc_id: str, request: Request):
        oc = request.app.state.repo.obter(oc_id)
        if not oc:
            raise HTTPException(404, "Ocorrência não encontrada.")
        return _publica(oc, str(request.base_url).rstrip("/"))

    @app.get("/api/classes", tags=["status"])
    def classes():
        return [{"classe": c, "rotulo": ROTULOS[c]} for c in CLASSES]

    # Front de demonstração do MVP (mapa + envio de foto). Montado por último para não sombrear /api.
    if settings.web_dir.exists():
        app.mount("/", StaticFiles(directory=settings.web_dir, html=True), name="web")
    return app


app = criar_app()
