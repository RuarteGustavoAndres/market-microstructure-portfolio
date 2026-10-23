"""Dataset de fills de un maker simulado con sim_cola sobre la captura REAL.

Simula un maker que cotiza best bid + best ask (size fija, --size, default 20)
sobre libro_top5 + trades_full reales, usando bot/sim_cola.py (modelo FIFO
aprobado, NO re-implementado aquí). Estrategia (supuestos aprobados):
  - En cada snapshot donde el lado correspondiente no tiene orden activa,
    se postea una Orden al best actual de ese lado (bid -> bid_px[0],
    ask -> ask_px[0]) con size fijo.
  - Re-quote (parámetro 'requote' de sim_cola, 3 modos comparables):
    'persist' (default del CLI, honesto) = sin auto-cancel, la orden stale
    persiste y solo llena por trade-through/sweep-through; 'cancel'
    (optimista) = cancel si el best PROPIO mejora; 'contra' (conservador) =
    cancel solo por movimiento DESFAVORABLE del mid. Cuando la orden se
    termina (cancel o fill completa), se vuelve a postear al best del
    snapshot siguiente; al re-postear, el queue_ahead sale del snapshot (no
    0). Una orden partial/unfilled sin cancel sigue activa hasta el fin de
    los datos (no se re-postea).
  - Lados independientes: bid y ask se simulan por separado (no interactúan:
    el bid llena con agresor 'sell', el ask con agresor 'buy').

Features por fila: mismas columnas y mismo orden que ml_features.csv
(ver features_ml.py), calculadas al MOMENTO DEL QUOTE desde los arrays del
snapshot (nivel 0 = best):
  spread_rel, imbalance, depth_total (suma top5), n_total (niveles top5),
  microprice_bias (weighted mid nivel-0 - mid), pos_en_spread, size_rel_depth,
  momentum_60s / momentum_300s (mid actual vs mid del último snapshot
  válido <= t-w; 0.0 si no hay historial), ofi (APROXIMACIÓN: deltas de
  bid_sz[0]/ask_sz[0] entre snapshots consecutivos en ventana de 5s con regla
  Cont-Kukanov-Stoikov, normalizada por depth medio de la ventana — la real
  usa libro_eventos; aquí solo hay snapshots), side_buy, horizonte_s.
  fill_price = price_entry y fill_size = size_filled (se duplican al final
  como columnas extra por conveniencia; ts_epoch/token_id ya están en el
  header base, NO se duplican).

P&L y target (MISMA convención que features_ml.py):
  P&L del fill = side_maker * (mid_futuro - price_entry), con side_maker
  +1 para bid y -1 para ask. mid_futuro = mid del ÚLTIMO snapshot válido
  en o antes de t_fill + horizonte (bisect, sin look-ahead). target = 1 si
  pnl < 0 (adverso) — equivalente exacto a la convención markout de
  features_ml.py: allí target=1 sii el markout del AGRESOR fue favorable,
  o sea sii el maker perdió. Por eso side_buy sigue la semántica de
  features_ml (lado del AGRESOR: 1 = fill de nuestro ask por comprador,
  0 = fill de nuestro bid por vendedor). Horizontes 60s y 300s -> una fila
  por (fill, horizonte).

Selección de tokens: los N (--tokens, default 50) con más filas etiquetadas
en dataset/ml_features.csv. Ventana temporal: min/max ts_epoch del CSV.
Solo lectura de DB (readonly). No toca servicios ni commitea.

Uso:
  ./venv-live/bin/python sim_maker_dataset.py --tokens 50 --json
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from sim_cola import Order, simular_cola  # noqa: E402
import psycopg2  # noqa: E402

BOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BOT_DIR, "dataset")
ML_FEATURES_CSV = os.path.join(DATASET_DIR, "ml_features.csv")
OUT_CSV_DEFAULT = os.path.join(DATASET_DIR, "maker_fills_sim.csv")

HORIZONTES = [60, 300]   # 1m y 5m, igual que features_ml (solo estos dos)
VENTANAS_MOM = [60, 300]
VENTANA_OFI = 5          # ventana OFI en segundos, igual que features_ml
EPS = 1e-9
MIN_SNAPS = 10           # snapshots mínimos por token, si no -> saltar
MIN_TRADES = 5           # trades mínimos por token, si no -> saltar

BASE_COLS = ["mid_futuro", "ts", "ts_epoch", "token_id", "horizonte_s",
             "side_buy", "best_bid", "best_ask", "mid", "spread",
             "spread_rel", "imbalance", "depth_total", "n_total",
             "fill_price", "fill_size", "momentum_60s", "momentum_300s",
             "pos_en_spread", "size_rel_depth", "ofi", "microprice_bias",
             "target"]
EXTRA_COLS = ["side", "price_entry", "size_filled", "status"]


def log(m):
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {m}", file=sys.stderr,
          flush=True)


def cargar_dsn():
    """Lee PREDIKT_TS_DSN de bot/.env.capturador sin imprimir credenciales."""
    path = os.path.join(BOT_DIR, ".env.capturador")
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("PREDIKT_TS_DSN="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("PREDIKT_TS_DSN no encontrado en .env.capturador")


def conectar():
    con = psycopg2.connect(cargar_dsn())
    con.set_session(readonly=True, autocommit=True)
    return con


# ---------------------------------------------------------------- tokens ----

def tokens_top_del_csv(n, path=ML_FEATURES_CSV):
    """Los n token_id con más filas etiquetadas en ml_features.csv."""
    with open(path, newline="") as fh:
        rdr = csv.reader(fh)
        header = next(rdr)
        i_tok = header.index("token_id")
        i_tgt = header.index("target")
    counts = {}
    with open(path, newline="") as fh:
        rdr = csv.reader(fh)
        next(rdr)
        for row in rdr:
            if row[i_tgt] != "":
                counts[row[i_tok]] = counts.get(row[i_tok], 0) + 1
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
    return [t for t, _ in top], counts


def ventana_del_csv(path=ML_FEATURES_CSV):
    """(min_ts_epoch, max_ts_epoch) de la columna ts_epoch del CSV."""
    mn, mx = None, None
    with open(path, newline="") as fh:
        rdr = csv.reader(fh)
        header = next(rdr)
        i = header.index("ts_epoch")
        for row in rdr:
            v = float(row[i])
            mn = v if mn is None or v < mn else mn
            mx = v if mx is None or v > mx else mx
    return mn, mx


# ------------------------------------------------------------- features -----

# Snapshot = (t, bid_px, bid_sz, ask_px, ask_sz) — nivel 0 = best (layout de sim_cola)
# s[1]=bid_px  s[2]=bid_sz  s[3]=ask_px  s[4]=ask_sz


def snap_valido(s):
    return bool(s[1]) and bool(s[3]) and s[1][0] is not None \
        and s[3][0] is not None and s[1][0] > 0 and s[3][0] > 0


def snap_en_o_antes(snaps, ts_list, t):
    """Último snapshot válido con ts <= t (bisect hacia atrás)."""
    i = bisect.bisect_right(ts_list, t) - 1
    while i >= 0:
        if snap_valido(snaps[i]):
            return snaps[i]
        i -= 1
    return None


def mid_de(s):
    if s is None or not snap_valido(s):
        return None
    return (s[1][0] + s[3][0]) / 2


def ofi_aprox(snaps, ts_list, t_q, ventana_s=VENTANA_OFI):
    """OFI APROXIMADA por snapshots (la real usa libro_eventos).

    Cont-Kukanov-Stoikov sobre nivel 0 con deltas entre snapshots
    consecutivos en [t_q - ventana_s, t_q]; normalizada por depth medio.
    """
    t_ini = t_q - ventana_s
    i = bisect.bisect_left(ts_list, t_ini)
    j = bisect.bisect_right(ts_list, t_q)
    ofi = 0.0
    depth_sum = 0.0
    n = 0
    prev = None
    for k in range(i, j):
        s = snaps[k]
        if not snap_valido(s):
            continue
        bb = s[1][0]
        ba = s[3][0]
        bd = sum(x or 0 for x in s[2])
        ad = sum(x or 0 for x in s[4])
        sz_b0 = s[2][0] or 0
        sz_a0 = s[4][0] or 0
        depth_sum += bd + ad
        n += 1
        if prev is not None:
            p_bb, p_ba, p_sz_b0, p_sz_a0 = prev
            if bb > p_bb:
                ofi += sz_b0
            elif bb < p_bb:
                ofi -= p_sz_b0
            else:
                ofi += sz_b0 - p_sz_b0
            if ba < p_ba:
                ofi -= sz_a0
            elif ba > p_ba:
                ofi += p_sz_a0
            else:
                ofi -= sz_a0 - p_sz_a0
        prev = (bb, ba, sz_b0, sz_a0)
    depth_medio = depth_sum / n if n else 0.0
    return ofi / depth_medio if depth_medio > 0 else 0.0


def features_en_quote(s, snaps, ts_list, t_q, size_filled, price_entry,
                      side_maker):
    """Features de features_ml.py recalculadas desde el snapshot del quote."""
    bb, ba = s[1][0], s[3][0]
    mid = (bb + ba) / 2
    spread = ba - bb
    bd = sum(x or 0 for x in s[2])
    ad = sum(x or 0 for x in s[4])
    depth_total = bd + ad
    spread_rel = spread / mid if mid > 0 else 0
    imbalance = (bd - ad) / depth_total if depth_total > 0 else 0.0
    n_total = len(s[1]) + len(s[3])

    mom = {}
    for w in VENTANAS_MOM:
        sw = snap_en_o_antes(snaps, ts_list, t_q - w)
        mid_prev = mid_de(sw)
        mom[f"momentum_{w}s"] = (mid - mid_prev) if mid_prev is not None else 0.0

    # pos_en_spread con la convención de features_ml (lado del agresor):
    # bid del maker -> agresor SELL -> pos = 1 - (p - bb)/spread; ask -> BUY.
    pos = (price_entry - bb) / spread if spread > 0 else 0.5
    if side_maker == "bid":
        pos = 1 - pos
    size_rel_depth = size_filled / depth_total if depth_total > 0 else 0.0

    ofi = ofi_aprox(snaps, ts_list, t_q)

    # micro-price proxy (Stoikov 2018) nivel-0: weighted mid - mid
    qd = (s[2][0] or 0) + (s[4][0] or 0)
    weighted_mid = (bb * (s[4][0] or 0) + ba * (s[2][0] or 0)) / qd if qd > 0 else mid
    microprice_bias = weighted_mid - mid

    return {
        "best_bid": bb, "best_ask": ba, "mid": mid, "spread": spread,
        "spread_rel": spread_rel, "imbalance": imbalance,
        "depth_total": depth_total, "n_total": n_total,
        "momentum_60s": mom["momentum_60s"], "momentum_300s": mom["momentum_300s"],
        "pos_en_spread": pos, "size_rel_depth": size_rel_depth,
        "ofi": ofi, "microprice_bias": microprice_bias,
    }


# -------------------------------------------------------------- simulación --

def sim_lado(side, snaps, ts_list, trades, size, requote="persist"):
    """Postea órdenes de un lado secuencialmente hasta el fin de los datos.

    Devuelve lista de dicts de simular_cola (+ quote_idx). En cada snapshot
    sin orden activa postea al best; en modo persist la orden solo termina
    al llenarse (la ventana no se trunca); en cancel/contra trunca en el
    primer snapshot que dispararía el cancel. Con re-quote implícito: al
    re-postear, queue_ahead sale del snapshot (no 0).
    """
    n = len(snaps)
    out = []
    idx = 0
    activo_hasta = -1.0
    while idx < n:
        t, bpx, bsz, apx, asz = snaps[idx]
        pxs = bpx if side == "bid" else apx
        # solo se cotiza en snapshots con best bid Y best ask válidos (igual
        # criterio de calidad que features_ml.snap_valido): sin ambos lados
        # no se pueden calcular las features del quote
        if not snap_valido(snaps[idx]):
            idx += 1
            continue
        if t <= activo_hasta:
            idx += 1
            continue
        price = pxs[0]

        # ventana de vida hasta el cancel (o fin de datos), según el modo:
        # persist -> j = None siempre (sin truncar); cancel/contra -> primer
        # snapshot posterior que dispararía el cancel en ese modo
        j = None
        if requote != "persist":
            for m in range(idx + 1, n):
                sm = snaps[m]
                if requote == "cancel":
                    p2 = (sm[1][0] if side == "bid" else sm[3][0]) \
                        if (sm[1] if side == "bid" else sm[3]) else None
                    if p2 is None:
                        continue
                    if side == "bid" and p2 > price + EPS:
                        j = m
                        break
                    if side == "ask" and p2 < price - EPS:
                        j = m
                        break
                else:  # contra
                    if not sm[1] or not sm[3]:
                        continue
                    mid = (sm[1][0] + sm[3][0]) / 2.0
                    if side == "bid" and mid < price - EPS:
                        j = m
                        break
                    if side == "ask" and mid > price + EPS:
                        j = m
                        break
        t_fin = ts_list[j] if j is not None else float("inf")
        sub_snaps = snaps[idx:(j + 1) if j is not None else n]
        sub_trades = [tr for tr in trades if t < tr[0] <= t_fin]

        o = Order(side=side, price=price, size=size, posted_t=t)
        r = simular_cola([o], sub_trades, sub_snaps, requote=requote)[0]
        r["quote_idx"] = idx
        out.append(r)

        if r["status"] == "filled":
            term = r["fill_events"][-1][0]
        elif r["cancelled_t"] is not None:
            term = r["cancelled_t"]
        else:
            break  # sigue activa al final de los datos
        activo_hasta = term
        nxt = bisect.bisect_right(ts_list, term)
        idx = nxt if nxt > idx else idx + 1
    return out


def build_por_token(con, token_id, t0, t1, size, requote="persist"):
    """Simula bid+ask del token y devuelve (filas, stats) del token."""
    cur = con.cursor()
    cur.execute(
        "SELECT ts, bid_px, bid_sz, ask_px, ask_sz FROM libro_top5 "
        "WHERE token_id=%s AND ts BETWEEN %s AND %s ORDER BY ts",
        (token_id, t0, t1),
    )
    snaps = []
    for r in cur.fetchall():
        # niveles con NULL (precio o tamaño) se descartan: sim_cola compara
        # precios de nivel contra floats y no tolera None dentro del array
        def _san(pxs, szs):
            op, osz = [], []
            for p, s in zip(pxs or [], szs or []):
                if p is None or s is None:
                    continue
                op.append(float(p))
                osz.append(float(s))
            return op, osz
        bpx, bsz = _san(r[1], r[2])
        apx, asz = _san(r[3], r[4])
        snaps.append((r[0].timestamp(), bpx, bsz, apx, asz))
    cur.execute(
        "SELECT ts, side, price, size FROM trades_full "
        "WHERE token_id=%s AND ts BETWEEN %s AND %s ORDER BY ts",
        (token_id, t0, t1),
    )
    trades = []
    for r in cur.fetchall():
        try:
            px = float(r[2])
            sz = float(r[3])
        except (TypeError, ValueError):
            continue
        if px <= 0 or sz <= 0 or r[1] not in ("BUY", "SELL"):
            continue
        trades.append((r[0].timestamp(), px, sz, r[1].lower()))
    if len(snaps) < MIN_SNAPS or len(trades) < MIN_TRADES:
        return None, {"razon": f"snapshots={len(snaps)} trades={len(trades)} "
                               f"< minimos ({MIN_SNAPS}/{MIN_TRADES})"}

    ts_list = [s[0] for s in snaps]
    stats = {"quotes": 0, "fills": 0, "status": {}}
    rows = []
    for side, side_maker_num in (("bid", 1.0), ("ask", -1.0)):
        orders = sim_lado(side, snaps, ts_list, trades, size, requote)
        for r in orders:
            stats["quotes"] += 1
            st = r["status"]
            stats["status"][st] = stats["status"].get(st, 0) + 1
            if r["filled_total"] <= EPS:
                continue
            stats["fills"] += 1
            t_fill = r["fill_events"][0][0]
            price_entry = r["price"]
            size_filled = r["filled_total"]
            q = snaps[r["quote_idx"]]
            fq = features_en_quote(q, snaps, ts_list, q[0], size_filled,
                                   price_entry, side)
            for h in HORIZONTES:
                sf = snap_en_o_antes(snaps, ts_list, t_fill + h)
                mid_futuro = mid_de(sf)
                if mid_futuro is None:  # no debería pasar: el quote mismo sirve
                    mid_futuro = fq["mid"]
                pnl = side_maker_num * (mid_futuro - price_entry)
                target = 1 if pnl < 0 else 0
                row = dict(fq)
                row.update({
                    "mid_futuro": mid_futuro,
                    "ts": datetime.fromtimestamp(t_fill, timezone.utc)
                          .isoformat(),
                    "ts_epoch": t_fill,
                    "token_id": token_id,
                    "horizonte_s": h,
                    "side_buy": 1 if side == "ask" else 0,
                    "fill_price": price_entry,
                    "fill_size": size_filled,
                    "target": target,
                    "side": side,
                    "price_entry": price_entry,
                    "size_filled": size_filled,
                    "status": r["status"],
                })
                rows.append(row)
    stats["fill_rate"] = stats["fills"] / stats["quotes"] if stats["quotes"] else 0.0
    return rows, stats


# -------------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=50)
    ap.add_argument("--size", type=float, default=20.0)
    ap.add_argument("--out", default=OUT_CSV_DEFAULT)
    ap.add_argument("--requote", choices=["persist", "cancel", "contra"],
                    default="persist",
                    help="modo de cancel/re-quote del sim "
                         "(persist=honesto sin auto-cancel)")
    ap.add_argument("--json", action="store_true",
                    help="resumen JSON en stdout")
    args = ap.parse_args()

    t_ini = time.time()
    errores = []
    log("leyendo tokens top y ventana de ml_features.csv ...")
    tokens, counts = tokens_top_del_csv(args.tokens)
    t0e, t1e = ventana_del_csv(ML_FEATURES_CSV)
    t0 = datetime.fromtimestamp(t0e, timezone.utc)
    t1 = datetime.fromtimestamp(t1e, timezone.utc)
    log(f"{len(tokens)} tokens; ventana {t0.isoformat()} .. {t1.isoformat()}")

    con = conectar()
    all_rows = []
    detalle = []
    saltados = []
    for i, tok in enumerate(tokens):
        ti = time.time()
        try:
            rows, stats = build_por_token(con, tok, t0, t1, args.size,
                                          args.requote)
        except Exception as e:  # noqa: BLE001
            import traceback
            tb = " ; ".join(traceback.format_exc().strip().splitlines()[-3:])
            errores.append(f"{tok[:12]}: {type(e).__name__}: {e} | {tb}")
            saltados.append({"token_id": tok, "razon": f"excepcion {type(e).__name__}"})
            continue
        if rows is None:
            saltados.append({"token_id": tok, **stats})
            log(f"  [{i+1}/{len(tokens)}] saltado {tok[:12]}: {stats['razon']}")
            continue
        all_rows.extend(rows)
        detalle.append({"token_id": tok, **stats,
                        "filas": len(rows), "seg": round(time.time() - ti, 1)})
        log(f"  [{i+1}/{len(tokens)}] {tok[:12]}: {stats['quotes']} quotes, "
            f"{stats['fills']} fills, rate={stats['fill_rate']:.3f}, "
            f"{len(rows)} filas, {time.time()-ti:.0f}s")
    con.close()

    if all_rows:
        cols = BASE_COLS + EXTRA_COLS
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in all_rows:
                w.writerow({c: r.get(c, "") for c in cols})
        log(f"dataset -> {args.out} ({os.path.getsize(args.out)} bytes, "
            f"{len(all_rows)} filas)")

    status_counts = {}
    for d in detalle:
        for k, v in d["status"].items():
            status_counts[k] = status_counts.get(k, 0) + v
    rates = [d["fill_rate"] for d in detalle if d["quotes"] > 0]
    fill_rate_promedio = sum(rates) / len(rates) if rates else 0.0

    resumen = {
        "script_path": os.path.join(BOT_DIR, "sim_maker_dataset.py"),
        "csv_path": args.out,
        "rows_written": len(all_rows),
        "tokens_procesados": len(detalle),
        "tokens_saltados": len(saltados),
        "fill_rate_promedio": round(fill_rate_promedio, 6),
        "status_counts": status_counts,
        "duracion_seg": round(time.time() - t_ini, 1),
        "errores": errores,
        "tokens_detalle": detalle,
        "tokens_saltados_detalle": saltados,
        "ventana": {"min_ts_epoch": t0e, "max_ts_epoch": t1e},
        "args": {"tokens": args.tokens, "size": args.size,
                 "requote": args.requote},
    }
    if args.json:
        print(json.dumps(resumen, indent=1))
    else:
        log(f"resumen: {json.dumps({k: v for k, v in resumen.items() if k != 'tokens_detalle'})}")


if __name__ == "__main__":
    main()
