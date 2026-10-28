#!/usr/bin/env python3
"""RAW exit markout, trade-to-trade: NO mid, NO book snapshots.

(3rd-round Claude request 2026-10-28: "mover el markout a trade-to-trade —
precio del trade a +h menos precio de entrada, con los mismos aggTrades,
sin snapshot de libro. Es barato y no hereda el desfase.")

Why: the drift/entry decomposition (and every probe before this one) looks
the exit up against the book mid, so both terms inherit the same possibly
lagged mid(t) and the cross-stream trades-vs-snapshots clock offset
(~<=0.5 s, results/markout_entry_lag_probe.json) contaminates everything at
short horizons. This probe never touches the book stream: both the entry
and the exit prices come from the SAME aggTrade stream, so no cross-stream
alignment exists to get wrong.

For each aggTrade at time t (the "entry fill") and horizon h:

    raw(h) = 10000 * sign * (p(t + h) - p(t)) / p(t)   [bps]

with sign = +1 for a buy aggressor (long entry) and -1 for a sell aggressor
(same convention as microstructure/scripts/markout_binance.py). p(t+h) is
the price of the first aggTrade at or after t+h (bisect on the same stream).

Interpretation caveats (declared, not hidden):
  - At small h the trade price toggles between bid-side and ask-side prints,
    so the bid-ask bounce shows up directly. Its size is ~the half-spread
    (SOL ~0.49 bps — the largest of the three symbols): for SOL at small h
    this measure is bounce-dominated and must be read with care. The bounce
    decays as h grows; the SIGN-CONSISTENT drift at large h is the signal.
  - The exit is a future TAKER print, not a maker fill: this is an upper
    bound on adverse selection for a maker who exits passively, and a lower
    bound on nothing in particular. It is a robustness cross-check for the
    mid-based decomposition, not a cost model by itself.

Read-only: the script never writes to the database.

Usage:
    python markout_trade_to_trade_probe.py [--since ...] [--until ...]
        [--symbols BTCUSDT,ETHUSDT,SOLUSDT] [--horizons 0.5,1,2,5,30,300]
        [--json]

Requires: psycopg2, PREDIKT_TS_DSN (or a local .env.capturador file next to
this script), binance_aggtrades populated. Stdlib only (no numpy).
"""
import argparse
import json
import os
import statistics
import sys
import time
from bisect import bisect_left
from pathlib import Path

import psycopg2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

_env = Path(os.path.join(os.path.dirname(__file__), ".env.capturador"))
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
DSN = os.environ.get("PREDIKT_TS_DSN")


def parse_ts(s):
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            import datetime as _dt
            return _dt.datetime.strptime(s, fmt).replace(
                tzinfo=_dt.timezone.utc).timestamp()
        except ValueError:
            pass
    raise ValueError(f"Invalid date format: {s}")


def to_iso(epoch):
    import datetime as _dt
    return (_dt.datetime.fromtimestamp(epoch, tz=_dt.timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ") if epoch else None)


def cargar(since, until, symbols):
    """aggTrades only — the book tables are never queried."""
    con = psycopg2.connect(DSN)
    con.autocommit = True
    cur = con.cursor()
    cur.execute("""
      SELECT symbol, EXTRACT(EPOCH FROM ts), price, aggressor
      FROM binance_aggtrades
      WHERE ts >= to_timestamp(%s) AND ts <= to_timestamp(%s)
        AND symbol = ANY(%s)
      ORDER BY ts
    """, (since, until, symbols))
    trades_por_sym = {}
    for sym, t, px, aggr in cur.fetchall():
        trades_por_sym.setdefault(sym, []).append(
            (float(t), float(px), 1 if aggr == "buy" else -1))
    con.close()
    return trades_por_sym


def analyze_symbol(trades, horizons):
    """raw(h) per horizon, split by aggressor side. Stdlib only.

    Returns ({h: {all/buy/sell: {n, mean_bps, median_bps}}}, n_no_exit)."""
    tt = [t for t, _, _ in trades]
    px = [p for _, p, _ in trades]
    last_t = tt[-1]
    acc = {h: {"all": [], "buy": [], "sell": []} for h in horizons}
    n_no_exit = 0
    for t, price, sign in trades:
        for h in horizons:
            target = t + h
            if target > last_t:
                n_no_exit += 1
                continue
            i = bisect_left(tt, target)
            exit_px = px[i]
            raw = 10000.0 * sign * (exit_px - price) / price
            acc[h]["all"].append(raw)
            acc[h]["buy" if sign > 0 else "sell"].append(raw)

    def stats(vals):
        return {
            "n": len(vals),
            "mean_bps": sum(vals) / len(vals) if vals else None,
            "median_bps": statistics.median(vals) if vals else None,
        }

    out = {h: {side: stats(v) for side, v in acc[h].items()} for h in horizons}
    return out, n_no_exit


def main():
    t0 = time.time()
    p = argparse.ArgumentParser(
        description="RAW trade-to-trade exit markout (no mid, no book)")
    p.add_argument("--since", type=str, default=None)
    p.add_argument("--until", type=str, default=None)
    p.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,SOLUSDT")
    p.add_argument("--horizons", type=str, default="0.5,1,2,5,30,300",
                   help="exit horizons in seconds")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    import datetime as _dt
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    horizons = [float(x) for x in args.horizons.split(",")]

    if args.since:
        since = parse_ts(args.since)
    else:
        since = None
    if args.until:
        until = parse_ts(args.until)
    else:
        until = _dt.datetime.now(_dt.timezone.utc).timestamp()

    if since is None:
        con = psycopg2.connect(DSN)
        con.autocommit = True
        cur = con.cursor()
        cur.execute("SELECT MIN(ts) FROM binance_aggtrades")
        row = cur.fetchone()
        con.close()
        if not row or not row[0]:
            raise RuntimeError("no data")
        since = row[0].timestamp() if hasattr(row[0], "timestamp") else float(row[0])

    trades = cargar(since, until, symbols)
    resumen = {}
    for sym in sorted(trades):
        r, n_no_exit = analyze_symbol(trades[sym], horizons)
        resumen[sym] = {
            "trades_input": len(trades[sym]),
            "n_sin_salida": n_no_exit,
            "por_horizonte_s": r,
        }

    report = {
        "window": {"since": to_iso(since), "until": to_iso(until)},
        "symbols_usados": sorted(trades.keys()),
        "horizontes_s": horizons,
        "definicion": ("raw(h) = 1e4 * sign * (p_trade(t+h) - p_trade(t)) / "
                       "p_trade(t); sign=+1 buy aggressor, -1 sell aggressor; "
                       "p_trade(t+h) = first aggTrade at or after t+h; "
                       "same aggTrade stream for entry and exit — no mid, no "
                       "book snapshot, no cross-stream clock to misalign."),
        "caveats": (
            "small h is bid-ask-bounce dominated (toggle between bid-side and "
            "ask-side prints; size ~ half-spread, largest for SOL ~0.49 bps — "
            "read SOL at small h with care); exit is a future TAKER print, not "
            "a maker fill: a robustness cross-check for the mid-based "
            "decomposition, not a cost model."),
        "resumen_por_simbolo": resumen,
        "lectura": (
            "sin desfase entre streams por construccion: todo el desfase de "
            "cruce (trades-vs-snapshots <=0.5 s) que contamina la "
            "descomposicion drift/entry esta ausente aqui; el rebote bid-ask "
            "aparece en su lugar y decae con h."),
        "duracion_seg": round(time.time() - t0, 3),
    }

    out_path = os.path.join(REPO_ROOT, "results", "markout_trade_to_trade.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print("window:", report["window"]["since"], "->", report["window"]["until"])
    for sym, r in resumen.items():
        print("== %s == trades=%d  sin_salida=%d"
              % (sym, r["trades_input"], r["n_sin_salida"]))
        print("  horizon_s |  mean_raw_bps(all)   median  (buy / sell)")
        for h in horizons:
            row = r["por_horizonte_s"][h]
            print("  %8.1f  |   %+9.4f         %+8.4f   (%+.4f / %+.4f)"
                  % (h, row["all"]["mean_bps"] or 0, row["all"]["median_bps"] or 0,
                     row["buy"]["mean_bps"] or 0, row["sell"]["mean_bps"] or 0))
    print("saved:", out_path)
    print("duration: %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
