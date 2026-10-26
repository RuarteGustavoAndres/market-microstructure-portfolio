"""Conexión a TimescaleDB local (predikt_ml) — batch insert, sin lockear por fila.

Contenedor Docker 'predikt_timescale', puerto 5434. AISLADO del SaaS
(saas_postgres 5433) — no toca ese proyecto. Self-hosted, sin terceros.

Por qué TimescaleDB y no SQLite: el WebSocket de mercado emite miles de
eventos/segundo; SQLite lockea toda la DB en cada INSERT y se vuelve cuello de
botella. Timescale hypertable + batch insert vía execute_values aguanta
(techo medido en benchmarks: 490k filas/s, nuestro volumen es ~miles/hora).

Patrón (del research de captura de alta frecuencia):
  - Buffer en memoria + flush cada BATCH_SIZE filas o BATCH_SEGUNDOS.
  - orjson para deserializar el stream (3-8x más rápido que json stdlib).
  - Reconexión con backoff; persistir last_ts fuera del proceso.
"""
import os
import time

import orjson
import psycopg2
import psycopg2.extras

DSN = os.environ.get(
    "TS_DSN",
    "postgresql://postgres:***@localhost:5434/microstructure")

BATCH_SIZE = int(os.environ.get("PREDIKT_BATCH", "500"))
BATCH_SEGUNDOS = float(os.environ.get("PREDIKT_BATCH_S", "2.0"))


def conectar():
    """Conexión fresh. autocommit=True: los batch insert no necesitan transacción larga."""
    con = psycopg2.connect(DSN)
    con.autocommit = True
    return con


def _a_tupla(ev):
    """Normaliza un evento del stream a tupla para INSERT. Devuelve None si no aplica."""
    t = ev.get("type")
    ts = ev.get("ts")
    if not ts:
        ts = int(time.time() * 1000)
    # ts viene en ms epoch desde el WS; convertir a timestamptz
    from datetime import datetime, timezone
    ts_dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
    return (ts_dt, t, orjson.dumps(ev).decode())


class BufferFlush:
    """Acumula filas en memoria y hace flush en batch. Evita el lock por fila."""

    def __init__(self, tabla, columnas, batch_size=BATCH_SIZE,
                 batch_segundos=BATCH_SEGUNDOS):
        self.tabla = tabla
        self.columnas = columnas
        self.batch_size = batch_size
        self.batch_segundos = batch_segundos
        self.buf = []
        self.ultimo_flush = time.time()
        self.total = 0

    def agregar(self, tupla):
        self.buf.append(tupla)

    def listo(self):
        return (len(self.buf) >= self.batch_size
                or (self.buf and time.time() - self.ultimo_flush >= self.batch_segundos))

    def flush(self, con):
        if not self.buf:
            return 0
        cols = ",".join(self.columnas)
        psycopg2.extras.execute_values(
            con,
            f"INSERT INTO {self.tabla} ({cols}) VALUES %s",
            self.buf, page_size=1000)
        n = len(self.buf)
        self.total += n
        self.buf.clear()
        self.ultimo_flush = time.time()
        return n


def insertar_snapshot_libro(con, ts_dt, token_id, best_bid, best_ask,
                            bid_depth, ask_depth, n_bids, n_asks):
    with con.cursor() as cur:
        cur.execute(
            "INSERT INTO libro_snapshots (ts,token_id,best_bid,best_ask,"
            "bid_depth,ask_depth,n_bids,n_asks) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            (ts_dt, token_id, best_bid, best_ask, bid_depth, ask_depth,
             n_bids, n_asks))


def contar(tabla):
    con = conectar()
    with con.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {tabla}")
        n = cur.fetchone()[0]
    con.close()
    return n
