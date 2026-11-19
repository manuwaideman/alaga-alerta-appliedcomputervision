"""Persistência das ocorrências confirmadas (SQLite, sem dependências extras)."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS ocorrencias (
    id              TEXT PRIMARY KEY,
    criada_em       TEXT NOT NULL,          -- ISO 8601 UTC
    lat             REAL NOT NULL,
    lon             REAL NOT NULL,
    origem          TEXT NOT NULL,          -- cidadao | camera
    camera_id       TEXT,
    descricao       TEXT,
    risco_score     INTEGER NOT NULL,
    risco_nivel     TEXT NOT NULL,
    risco_motivo    TEXT NOT NULL,
    classes         TEXT NOT NULL,          -- JSON: lista de classes detectadas
    deteccoes       TEXT NOT NULL,          -- JSON: lista de detecções
    anonimizacao    TEXT NOT NULL,          -- JSON: relatório de anonimização
    imagem          TEXT NOT NULL,          -- arquivo anonimizado (sem caixas)
    imagem_anotada  TEXT NOT NULL           -- arquivo anonimizado com as caixas
);
CREATE INDEX IF NOT EXISTS idx_ocorrencias_criada_em ON ocorrencias(criada_em);
"""

JSON_COLS = ("classes", "deteccoes", "anonimizacao")


class Repositorio:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _row(row: sqlite3.Row) -> dict:
        d = dict(row)
        for col in JSON_COLS:
            d[col] = json.loads(d[col])
        return d

    def inserir(self, oc: dict) -> None:
        dados = {k: (json.dumps(v, ensure_ascii=False) if k in JSON_COLS else v) for k, v in oc.items()}
        cols = ", ".join(dados)
        marcadores = ", ".join(f":{k}" for k in dados)
        with self._conn() as c:
            c.execute(f"INSERT INTO ocorrencias ({cols}) VALUES ({marcadores})", dados)

    def obter(self, oc_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM ocorrencias WHERE id = ?", (oc_id,)).fetchone()
        return self._row(row) if row else None

    def listar(self, horas: float | None = 24, nivel_minimo: str | None = None, limite: int = 500) -> list[dict]:
        sql, params = "SELECT * FROM ocorrencias WHERE 1=1", []
        if horas:
            corte = (datetime.now(timezone.utc) - timedelta(hours=horas)).isoformat()
            sql += " AND criada_em >= ?"
            params.append(corte)
        if nivel_minimo:
            ordem = ["baixo", "medio", "alto"]
            aceitos = ordem[ordem.index(nivel_minimo):]
            sql += f" AND risco_nivel IN ({','.join('?' * len(aceitos))})"
            params.extend(aceitos)
        sql += " ORDER BY criada_em DESC LIMIT ?"
        params.append(limite)
        with self._conn() as c:
            return [self._row(r) for r in c.execute(sql, params).fetchall()]
