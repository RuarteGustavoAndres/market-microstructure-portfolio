"""Simulador puro de llenado de órdenes maker con posición en cola FIFO.

Dados un tape de trades (aggTrades), una serie de snapshots de libro top5 y
un conjunto de órdenes maker ya enviadas, estima cómo se llenó cada orden.

MODELO (supuestos aprobados por Gustavo — NO cambiar sin avisar):

1. Queue-ahead FIFO: al postear, el volumen por delante en nuestra cola =
   tamaño en NUESTRO nivel exacto del último snapshot con t <= posted_t
   (el estado del libro vigente al postear). Para bid: índice i con
   bid_px[i] == price -> bid_sz[i]; si nuestro precio NO está en ese libro,
   queue_ahead = 0 (esa orden solo puede llenar por trade-through).

2. Fill = trade-through con volumen:
   - Trade del agresor contrario EN nuestro precio EXACTO: consume primero
     el queue_ahead restante; el excedente (trade.size - cola restante) llena
     NUESTRA orden hasta su tamaño restante. SIMPLIFICACIÓN DOCUMENTADA: no
     modelamos volumen oculto agregado detrás nuestro con posterioridad al
     posteo — el excedente completo del trade es nuestro, con tope en el
     restante propio (si un trade llega después de que nos postearon, el
     excedente no se reparte con nadie más).
   - Trade que BARRIÓ nuestro nivel (bid: aggressor 'sell' con
     price < order.price; ask: aggressor 'buy' con price > order.price):
     el nivel fue barrido -> se consume TODO el queue_ahead restante y la
     orden llena COMPLETA, sin importar el tamaño del trade. Sin el libro
     exacto al instante del trade no se puede alocar cross-level;
     sweep-through = nivel barrido.
   - Trades del MISMO agresor que nuestro lado (bid con aggressor 'buy')
     no tocan nuestro nivel.

3. Re-quote (modos comparables, parámetro 'requote'):
   - 'cancel' (status quo, default del motor): cancel conservador si el best
     PROPIO mejora (bid: bid_px[0] > price; ask: ask_px[0] < price) en un
     snapshot posterior al posteo. OPTIMISTA: evita fills stale tóxicos.
   - 'persist' (opción a, honesta): sin auto-cancel — la orden stale persiste
     y solo llena por trade-through/sweep-through (regla 2). Restaura fills
     tóxicos. El re-posteo tras fill usa queue_ahead del snapshot (no 0):
     modelar la cola al re-cotar es inherente a este diseño (opción b).
   - 'contra' (opción c, conservadora): cancel solo por movimiento
     DESFAVORABLE del mid (bid: mid < price; ask: mid > price) — gestión de
     riesgo, no evasión de stale. Si el snapshot no tiene ambos lados, no
     cancela. Fills ya acumulados se conservan (igual que 'cancel').
   En 'cancel' y 'contra': en empate de timestamps el snapshot se procesa
   ANTES que el trade (el cancel gana sobre el fill del mismo instante).

4. Momento de posteo: una orden se activa para eventos con t > posted_t
   (estricto): un trade con t == posted_t no la llena y un snapshot con
   t == posted_t no la cancela.

5. Órdenes al mismo precio y mismo lado comparten la cola FIFO entre sí
   (la más vieja llena primero). El queue ahead efectivo de una orden =
   queue_ahead de su snapshot + tamaño restante (sin cancelar) de las
   órdenes propias más viejas en ese nivel.

Salida por orden: dict con side, price, size, posted_t,
queue_ahead_inicial, filled_total, fill_events [(t_fill, size_filled)...],
cancelled_t (None si no se canceló) y status:
'filled' | 'partial' | 'cancelled' | 'unfilled'.

Puro: stdlib-only (bisect, dataclasses), sin I/O, sin numpy, determinista.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass

EPS = 1e-9

# Entradas (tuples, stdlib puro):
#   Snapshot = (t: float, bid_px: list[float], bid_sz: list[float],
#               ask_px: list[float], ask_sz: list[float])  # nivel 0 = best
#   Trade    = (t: float, price: float, size: float, aggressor: str)
#              # aggressor 'buy' levanta el ask, 'sell' golpea el bid
Snapshot = tuple
Trade = tuple


@dataclass
class Order:
    side: str      # 'bid' | 'ask'
    price: float
    size: float
    posted_t: float


class _St:
    """Estado interno mutable de una orden durante la simulación."""
    __slots__ = ("qa_inicial", "qa_rem", "rem", "filled", "events",
                 "cancelled_t", "done")

    def __init__(self, size: float):
        self.qa_inicial = 0.0
        self.qa_rem = 0.0
        self.rem = size
        self.filled = 0.0
        self.events: list[tuple[float, float]] = []
        self.cancelled_t: float | None = None
        self.done = False


def _snap_queue_ahead(order: Order, snaps: list, snap_ts: list[float]) -> float:
    """Regla 1: tamaño en nuestro nivel exacto del último snapshot t<=posted_t."""
    k = bisect.bisect_right(snap_ts, order.posted_t) - 1
    if k < 0:
        return 0.0
    t, bpx, bsz, apx, asz = snaps[k]
    pxs, szs = (bpx, bsz) if order.side == "bid" else (apx, asz)
    for i, p in enumerate(pxs):
        if p == order.price:
            return float(szs[i])
    return 0.0


def _on_snapshot(snap, st: list, orders: list, active: list, requote: str) -> None:
    """Regla 3: cancel según modo ('cancel' | 'persist' | 'contra')."""
    if requote == "persist":
        return
    t, bpx, bsz, apx, asz = snap
    for i in active:
        o = orders[i]
        s = st[i]
        if s.done:
            continue
        if requote == "cancel":
            adverso = ((o.side == "bid" and bpx and bpx[0] > o.price + EPS)
                       or (o.side == "ask" and apx and apx[0] < o.price - EPS))
        else:  # contra
            if not bpx or not apx:
                continue
            mid = (bpx[0] + apx[0]) / 2.0
            adverso = ((o.side == "bid" and mid < o.price - EPS)
                       or (o.side == "ask" and mid > o.price + EPS))
        if adverso:
            s.cancelled_t = t
            s.done = True


def _on_trade(trade, st: list, orders: list, active: list) -> None:
    """Regla 2: consume colas FIFO en el precio exacto y barre niveles cruzados."""
    t, px, sz, agg = trade
    if sz <= 0:
        return
    side_needed = "bid" if agg == "sell" else "ask"

    # Órdenes propias EN el precio del trade (FIFO entre ellas, regla 5)
    at_price = [i for i in active
                if orders[i].side == side_needed
                and orders[i].price == px and not st[i].done]
    at_price.sort(key=lambda i: orders[i].posted_t)

    # 1) quemar colas (queue ahead) en orden FIFO
    vol = sz
    for i in at_price:
        if vol <= EPS:
            break
        take = min(st[i].qa_rem, vol)
        st[i].qa_rem -= take
        vol -= take
    # 2) el excedente llena nuestras órdenes, la más vieja primero
    for i in at_price:
        if vol <= EPS:
            break
        f = min(st[i].rem, vol)
        if f > EPS:
            st[i].rem -= f
            st[i].filled += f
            st[i].events.append((t, f))
            vol -= f
            if st[i].rem <= EPS:
                st[i].done = True

    # 3) barrido: trade cruzó NUESTRO nivel exacto -> nivel barrido, fill completo
    for i in active:
        o = orders[i]
        s = st[i]
        if s.done or o.side != side_needed or o.price == px:
            continue
        swept = (agg == "sell" and px < o.price) or (agg == "buy" and px > o.price)
        if swept:
            f = s.rem
            s.qa_rem = 0.0
            s.rem = 0.0
            s.filled += f
            s.events.append((t, f))
            s.done = True


def simular_cola(orders: list[Order], trades: list[Trade], snaps: list[Snapshot],
                 requote: str = "cancel") -> list[dict]:
    """Simula el llenado de órdenes maker sobre un tape real de trades.

    Ver docstring del módulo para el modelo completo. Devuelve un dict por
    orden (mismo orden de entrada) con fills, cola inicial y estado final.
    """
    if requote not in ("cancel", "persist", "contra"):
        raise ValueError(f"requote inválido: {requote!r}")
    n = len(orders)
    order_by_t = sorted(range(n), key=lambda i: orders[i].posted_t)
    snap_ts = [s[0] for s in snaps]

    st = [_St(orders[i].size) for i in range(n)]
    for i in range(n):
        st[i].qa_inicial = _snap_queue_ahead(orders[i], snaps, snap_ts)

    # grupos por (side, price) en orden de posted_t (regla 5)
    groups: dict[tuple[str, float], list[int]] = {}
    for i in order_by_t:
        o = orders[i]
        groups.setdefault((o.side, o.price), []).append(i)

    active: list[int] = []  # órdenes ya activadas y no terminadas

    def activate(t_ev: float) -> None:
        nonlocal ptr
        # regla 4: activación estricta (t_ev > posted_t)
        while ptr < len(order_by_t) and orders[order_by_t[ptr]].posted_t < t_ev:
            i = order_by_t[ptr]
            ptr += 1
            o = orders[i]
            if st[i].done:
                continue
            # regla 5: cola efectiva = snapshot + restante de órdenes propias
            # más viejas en el mismo nivel (que no estén canceladas)
            extra = 0.0
            for j in groups[(o.side, o.price)]:
                if j == i:
                    break
                sj = st[j]
                if not sj.done and sj.cancelled_t is None:
                    extra += sj.rem
            st[i].qa_rem = st[i].qa_inicial + extra
            active.append(i)

    ptr = 0
    # eventos cronológicos; en empate de t, snapshot antes que trade (regla 3)
    events = sorted(
        [(s[0], 0, k, s) for k, s in enumerate(snaps)]
        + [(tr[0], 1, k, tr) for k, tr in enumerate(trades)],
        key=lambda e: (e[0], e[1], e[2]),
    )
    for t_ev, kind, _, ev in events:
        activate(t_ev)
        if kind == 0:
            _on_snapshot(ev, st, orders, active, requote)
        else:
            _on_trade(ev, st, orders, active)

    out = []
    for i in range(n):
        o = orders[i]
        s = st[i]
        if s.filled >= o.size - EPS:
            status = "filled"
        elif s.cancelled_t is not None:
            status = "cancelled"
        elif s.filled > EPS:
            status = "partial"
        else:
            status = "unfilled"
        out.append({
            "side": o.side,
            "price": o.price,
            "size": o.size,
            "posted_t": o.posted_t,
            "queue_ahead_inicial": s.qa_inicial,
            "filled_total": s.filled,
            "fill_events": list(s.events),
            "status": status,
            "cancelled_t": s.cancelled_t,
        })
    return out
