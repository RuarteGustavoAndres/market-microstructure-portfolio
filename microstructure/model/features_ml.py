"""Dataset de features para modelo de toxicidad — stdlib pura.

Target honesto: para CADA trade real del libro (trades_ml), markout propio
(precio del libro a +horizon vs precio del fill). Markout negativo = el flujo
informado nos llenó el lado perdedor = ADVERSE SELECTION real.

Features (microestructura enriquecida, siguiendo la literatura):
  Estado del libro antes del fill: spread, imbalance, depth, n niveles.
  Momentum en ventanas: mid vs mid hace 1m y 5m.
  Agresividad del fill: posición dentro del spread + tamaño relativo al depth.
  OFI (Order Flow Imbalance) — Cont-Kukanov-Stoikov 2014. LA feature central.
    e_n = +Δbid_size si P_bid sube, -Δbid_size si P_bid baja, -Δask_size si
    P_ask baja, +Δask_size si P_ask sube. Sumado en ventana de VENTANA_OFI s,
    normalizado por depth medio de la ventana.
  Micro-price proxy (Stoikov 2018): weighted_mid - mid, donde
    weighted_mid = (bid*ask_sz + ask*bid_sz)/(bid_sz+ask_sz).

Multi-horizonte (jugada gratis): etiqueta CADA fill a 4 horizontes
(1s/10s/1m/5m) => multiplica muestras sin trades nuevos. Cada fila del CSV
es (trade, horizonte) con su target correspondiente.

Filtros de calidad:
  - tokens con < MIN_TRADES trades
  - mercados penny (mid mediano < min_mid)

Uso:
  ./venv-live/bin/python features_ml.py --min-mid 0.05
"""

import argparse
import bisect
import csv
import os
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(__file__))
import tsdb

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

MIN_TRADES = 15
MIN_MID = 0.05
VENTANAS_MOM = [60, 300]       # momentum 1m y 5m
VENTANA_OFI = 5                # ventana OFI en segundos
HORIZONTES = [1, 10, 60, 300]  # multi-etiquetado: 1s, 10s, 1m, 5m


def log(m):
    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] {m}", flush=True)


def snap_valido(s):
    return s[1] is not None and s[2] is not None and s[1] > 0 and s[2] > 0


def snap_en_o_antes(snaps, ts_list, t):
    """Último snapshot válido con ts <= t (búsqueda binaria hacia atrás)."""
    i = bisect.bisect_right(ts_list, t) - 1
    while i >= 0:
        if snap_valido(snaps[i]):
            return snaps[i]
        i -= 1
    return None


def snap_en_o_despues(snaps, ts_list, t):
    """Primer snapshot válido con ts >= t (búsqueda binaria hacia adelante)."""
    i = bisect.bisect_left(ts_list, t)
    while i < len(snaps):
        if snap_valido(snaps[i]):
            return snaps[i]
        i += 1
    return None


def ofi_ventana(snaps, ts_list, ts_fin, ventana_s):
    """OFI normalizado de la ventana [ts_fin - ventana_s, ts_fin].

    Recorre los eventos del libro en la ventana y suma e_n; normaliza por
    depth medio. Es la medida de flujo de Cont et al. (2014).
    """
    t_ini = ts_fin - timedelta(seconds=ventana_s)
    i = bisect.bisect_left(ts_list, t_ini)
    j = bisect.bisect_right(ts_list, ts_fin)
    ofi = 0.0
    depth_sum = 0.0
    n = 0
    prev = None
    for k in range(i, j):
        s = snaps[k]
        if not snap_valido(s):
            continue
        _, bb, ba, bd, ad, nb, na = s
        bd = bd or 0
        ad = ad or 0
        depth_sum += bd + ad
        n += 1
        if prev is not None:
            p_bb, p_ba, p_bd, p_ad = prev
            if bb > p_bb:
                ofi += bd
            elif bb < p_bb:
                ofi -= p_bd
            if ba < p_ba:
                ofi -= ad
            elif ba > p_ba:
                ofi += p_ad
        prev = (bb, ba, bd, ad)
    depth_medio = depth_sum / n if n else 0.0
    return ofi / depth_medio if depth_medio > 0 else 0.0


def build_por_token(con, token_id):
    """Por token: cada trade x cada horizonte -> features + label markout REAL."""
    cur = con.cursor()
    cur.execute(
        "SELECT ts, best_bid, best_ask, bid_depth, ask_depth, n_bids, n_asks "
        "FROM libro_snapshots WHERE token_id=%s ORDER BY ts",
        (token_id,),
    )
    snaps = cur.fetchall()
    if len(snaps) < 10:
        return []
    ts_list = [s[0] for s in snaps]

    cur.execute(
        "SELECT ts, side, price, coalesce(size,0) size FROM trades_ml "
        "WHERE token_id=%s AND price>0 AND side IN ('BUY','SELL') "
        "ORDER BY ts",
        (token_id,),
    )
    trades = cur.fetchall()
    if len(trades) < MIN_TRADES:
        return []

    feats = []
    for ts, side, price, size in trades:
        s = snap_en_o_antes(snaps, ts_list, ts)
        if s is None:
            continue
        _, bb, ba, bd, ad, nb, na = s
        bd = bd or 0
        ad = ad or 0
        mid = (bb + ba) / 2
        spread = ba - bb
        depth_total = bd + ad

        # momentum en ventanas
        mom = {}
        for w in VENTANAS_MOM:
            sw = snap_en_o_antes(snaps, ts_list, ts - timedelta(seconds=w))
            mom[f"momentum_{w}s"] = (mid - (sw[1] + sw[2]) / 2) if sw else 0.0

        # agresividad: posición del fill dentro del spread
        pos_en_spread = (price - bb) / spread if spread > 0 else 0.5
        if side == "SELL":
            pos_en_spread = 1 - pos_en_spread
        size_rel_depth = size / depth_total if depth_total > 0 else 0.0

        # OFI (Cont et al. 2014)
        ofi = ofi_ventana(snaps, ts_list, ts, VENTANA_OFI)

        # micro-price proxy (Stoikov 2018): weighted mid - mid
        weighted_mid = (bb * ad + ba * bd) / depth_total if depth_total > 0 else mid
        microprice_bias = weighted_mid - mid

        # multi-horizonte: una fila por horizonte con su target
        for h in HORIZONTES:
            sf = snap_en_o_despues(snaps, ts_list,
                                   ts + timedelta(seconds=h))
            target = None
            if sf is not None:
                mid_futuro = (sf[1] + sf[2]) / 2
                if side == "BUY":
                    target = 1 if (mid_futuro - price) > 0 else 0
                else:
                    target = 1 if (price - mid_futuro) > 0 else 0
            row = {
                "ts": ts.isoformat(),
                "ts_epoch": ts.timestamp(),
                "token_id": token_id,
                "horizonte_s": h,
                "side_buy": 1 if side == "BUY" else 0,
                "best_bid": bb,
                "best_ask": ba,
                "mid": mid,
                "spread": spread,
                "spread_rel": spread / mid if mid > 0 else 0,
                "imbalance": (bd - ad) / depth_total if depth_total > 0 else 0.0,
                "depth_total": depth_total,
                "n_total": (nb or 0) + (na or 0),
                "fill_price": price,
                "fill_size": size,
                **mom,
                "pos_en_spread": pos_en_spread,
                "size_rel_depth": size_rel_depth,
                "ofi": ofi,
                "microprice_bias": microprice_bias,
                "target": target,
            }
            feats.append(row)
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-mid", type=float, default=MIN_MID)
    ap.add_argument("--out", default=os.path.join(DATASET_DIR, "ml_features.csv"))
    args = ap.parse_args()

    con = tsdb.conectar()
    cur = con.cursor()
    cur.execute(
        "SELECT token_id, count(*) n FROM trades_ml "
        "WHERE price>0 AND side IN ('BUY','SELL') "
        "GROUP BY token_id HAVING count(*)>=%s ORDER BY n DESC",
        (MIN_TRADES,),
    )
    tokens = [r[0] for r in cur.fetchall()]
    log(f"{len(tokens)} tokens con >= {MIN_TRADES} trades; "
        f"{len(HORIZONTES)} horizontes por fill")

    all_feats = []
    t0 = time.time()
    for i, tok in enumerate(tokens):
        f = build_por_token(con, tok)
        if f:
            mid_med = sorted(x["mid"] for x in f)[len(f) // 2]
            if mid_med < args.min_mid:
                continue
        all_feats.extend(f)
        if (i + 1) % 20 == 0:
            log(f"  {i+1}/{len(tokens)} tokens, {len(all_feats)} filas")

    etiquetadas = [r for r in all_feats if r["target"] is not None]
    log(f"features: {len(all_feats)} filas ({len(etiquetadas)} etiquetadas) "
        f"en {time.time()-t0:.0f}s")
    log(f"distribución target: {dict(Counter(r['target'] for r in etiquetadas))}")

    if all_feats:
        keys = list(all_feats[0].keys())
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(all_feats)
        log(f"dataset -> {args.out} ({os.path.getsize(args.out)} bytes)")
    con.close()


if __name__ == "__main__":
    main()
