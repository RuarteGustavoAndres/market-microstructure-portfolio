#!/usr/bin/env python3
"""Adverse-selection markout on REAL Binance capture data (TimescaleDB).

Reads binance_aggtrades (trades with aggressor side) and rebuilds the mid
from binance_book_top5 (best bid/ask per snapshot), then computes markouts
at +5s/+30s/+60s/+300s in both USD and basis points (bps) relative to the
mid at entry.

Metrics per symbol and horizon:
  - mean/median markout in USD and bps
  - toxic_frac (fraction of trades with negative markout)
  - side breakdown (buy/sell mean bps)
  - decomposition: raw = drift + entry (all in bps)
      raw   = sign * (mid(t+h) - price)
      drift = sign * (mid(t+h) - mid(t))  # market drift after fill
      entry = sign * (mid(t)   - price)   # spread paid at entry
  - mid freshness at exit: (t+h) - ts_of_last_snapshot_at_or_before(t+h)

Usage:
    python markout_binance.py [--since YYYY-MM-DD[THH:MM]] [--until ...]
                              [--symbols BTCUSDT,ETHUSDT] [--json]

Requires:
    - psycopg2
    - PREDIKT_TS_DSN environment variable (postgresql://user:***@host:port/db),
      or a local .env.capturador file next to the script with that variable.
    - Binance capture tables populated: binance_aggtrades, binance_book_top5.

Read-only: the script never writes to the database.
"""
import argparse
import json
import os
import statistics
import sys
import time
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path

import psycopg2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

# Optional local env file (not committed): PREDIKT_TS_DSN=postgresql://...
_env = Path(os.path.join(os.path.dirname(__file__), ".env.capturador"))
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
DSN = os.environ.get("PREDIKT_TS_DSN")

HORIZONS = (5, 30, 60, 300)
DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT"


def parse_args():
    p = argparse.ArgumentParser(description="Adverse-selection markout on Binance capture data")
    p.add_argument("--since", type=str, default=None, help="YYYY-MM-DD[THH:MM] (default: min(ts) in DB)")
    p.add_argument("--until", type=str, default=None, help="YYYY-MM-DD[THH:MM] (default: now)")
    p.add_argument("--symbols", type=str, default=DEFAULT_SYMBOLS, help="Comma-separated symbols")
    p.add_argument("--json", action="store_true", help="Output raw JSON to stdout")
    return p.parse_args()


def parse_ts(s):
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass
    raise ValueError(f"Invalid date format: {s} (expected YYYY-MM-DD[THH:MM])")


def to_iso(epoch):
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if epoch else None


def cargar(since, until, symbols):
    con = psycopg2.connect(DSN)
    con.autocommit = True
    cur = con.cursor()

    if since is None:
        cur.execute("""
          SELECT LEAST(
            (SELECT MIN(ts) FROM binance_aggtrades),
            (SELECT MIN(ts) FROM binance_book_top5)
          )
        """)
        row = cur.fetchone()
        if not row or not row[0]:
            raise RuntimeError("No data found in binance_aggtrades or binance_book_top5")
        since = row[0].timestamp() if hasattr(row[0], "timestamp") else float(row[0])
    if until is None:
        until = datetime.now(timezone.utc).timestamp()

    cur.execute("""
      SELECT b.symbol, EXTRACT(EPOCH FROM b.ts), b.price, a.price
      FROM binance_book_top5 b
      JOIN binance_book_top5 a
        ON a.symbol=b.symbol AND a.ts=b.ts AND a.side='ask' AND a.level=0
      WHERE b.side='bid' AND b.level=0
        AND b.ts >= to_timestamp(%s) AND b.ts <= to_timestamp(%s)
        AND b.symbol = ANY(%s)
      ORDER BY 2
    """, (since, until, symbols))
    mids_por_sym = {}
    for sym, t, bid, ask in cur.fetchall():
        mids_por_sym.setdefault(sym, []).append((float(t), float(bid) / 2.0 + float(ask) / 2.0))

    cur.execute("""
      SELECT symbol, EXTRACT(EPOCH FROM ts), price, aggressor
      FROM binance_aggtrades
      WHERE ts >= to_timestamp(%s) AND ts <= to_timestamp(%s)
        AND symbol = ANY(%s)
      ORDER BY ts
    """, (since, until, symbols))
    trades_por_sym = {}
    for sym, t, px, aggr in cur.fetchall():
        trades_por_sym.setdefault(sym, []).append((float(t), float(px), aggr))
    con.close()
    return trades_por_sym, mids_por_sym, since, until


def analyze(trades, mids, raw_freshness_out=None):
    """Extended markout: bps, median, decomposition and freshness.
    trades: list of (t_sec, price, aggressor).
    mids:   sorted list of (t_sec, mid).
    Returns {horizon: {...metrics...}}"""
    mt = [t for t, _ in mids]
    mv = [v for _, v in mids]
    last_t = mt[-1] if mt else None

    acc = {}
    for h in HORIZONS:
        acc[h] = {
            "usd": [], "bps": [], "buy_bps": [], "sell_bps": [],
            "toxic": 0, "excluded": 0, "raw_bps": [], "drift_bps": [],
            "entry_bps": [], "freshness": []
        }

    for t, price, aggr in trades:
        sign = -1 if aggr == "sell" else 1
        i0 = bisect_right(mt, t) - 1
        if i0 < 0:
            for h in HORIZONS:
                acc[h]["excluded"] += 1
            continue
        mid_t = mv[i0]
        if mid_t <= 0:
            for h in HORIZONS:
                acc[h]["excluded"] += 1
            continue

        for h in HORIZONS:
            th = t + h
            if last_t is None or th > last_t:
                acc[h]["excluded"] += 1
                continue
            i = bisect_right(mt, th) - 1
            if i < 0:
                acc[h]["excluded"] += 1
                continue

            mid_th = mv[i]
            ts_mid_th = mt[i]
            fresh = th - ts_mid_th

            raw_usd = sign * (mid_th - price)
            drift_usd = sign * (mid_th - mid_t)
            entry_usd = sign * (mid_t - price)

            raw_bps = 10000.0 * raw_usd / mid_t
            drift_bps = 10000.0 * drift_usd / mid_t
            entry_bps = 10000.0 * entry_usd / mid_t

            r = acc[h]
            r["usd"].append(raw_usd)
            r["bps"].append(raw_bps)
            if aggr == "buy":
                r["buy_bps"].append(raw_bps)
            elif aggr == "sell":
                r["sell_bps"].append(raw_bps)

            r["raw_bps"].append(raw_bps)
            r["drift_bps"].append(drift_bps)
            r["entry_bps"].append(entry_bps)
            r["freshness"].append(fresh)
            if raw_usd < 0:
                r["toxic"] += 1

    if raw_freshness_out is not None:
        for h in HORIZONS:
            raw_freshness_out.extend(acc[h]["freshness"])

    out = {}
    for h in HORIZONS:
        r = acc[h]
        n = len(r["bps"])
        out[h] = {
            "n": n,
            "excluded": r["excluded"],
            "mean_usd": sum(r["usd"]) / n if n else None,
            "mean_bps": sum(r["bps"]) / n if n else None,
            "median_bps": statistics.median(r["bps"]) if n else None,
            "toxic_frac": r["toxic"] / n if n else None,
            "buy_mean_bps": sum(r["buy_bps"]) / len(r["buy_bps"]) if r["buy_bps"] else None,
            "sell_mean_bps": sum(r["sell_bps"]) / len(r["sell_bps"]) if r["sell_bps"] else None,
            "raw_bps": sum(r["raw_bps"]) / n if n else None,
            "drift_bps": sum(r["drift_bps"]) / n if n else None,
            "entry_bps": sum(r["entry_bps"]) / n if n else None,
            "freshness_mean_s": sum(r["freshness"]) / n if n else None,
            "freshness_median_s": statistics.median(r["freshness"]) if n else None,
        }
    return out


def main():
    t_start = time.time()
    args = parse_args()
    since = parse_ts(args.since)
    until = parse_ts(args.until)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    trades, mids, since, until = cargar(since, until, symbols)

    resumen = {}
    trades_input_totales = 0
    global_freshness = []

    for sym in sorted(trades):
        t_list = trades[sym]
        m_list = mids.get(sym, [])
        trades_input_totales += len(t_list)

        ext = analyze(t_list, m_list, raw_freshness_out=global_freshness)
        resumen[sym] = {
            "trades_input": len(t_list),
            "por_horizonte": ext,
        }

    duracion_seg = time.time() - t_start

    report = {
        "window": {"since": to_iso(since), "until": to_iso(until)},
        "symbols_usados": sorted(trades.keys()),
        "trades_input_totales": trades_input_totales,
        "resumen_por_simbolo": resumen,
        "frescura_mid_mediana_s": statistics.median(global_freshness) if global_freshness else None,
        "duracion_seg": round(duracion_seg, 3),
        "errores": [],
    }

    # Save to file
    report_path = os.path.join(REPO_ROOT, "results", "markout_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    if args.json:
        print(json.dumps(report, indent=2))
        return

    # Human readable table
    print(f"Window: {to_iso(since)} -> {to_iso(until)}")
    print(f"Symbols: {', '.join(sorted(trades.keys()))}")
    print(f"Trades input: {trades_input_totales}")
    print("-" * 110)
    for sym, s in resumen.items():
        print(f"== {sym} == (trades_input={s['trades_input']})")
        for h, v in s["por_horizonte"].items():
            if v["n"] > 0:
                print(f"  +{h:>3}s: n={v['n']:>6} excl={v['excluded']:>5} "
                      f"mean_usd={v['mean_usd']:>10.4f} mean_bps={v['mean_bps']:>8.3f} "
                      f"med_bps={v['median_bps']:>8.3f} toxic={v['toxic_frac']:.1%} "
                      f"[buy={v['buy_mean_bps'] or 0:.2f} sell={v['sell_mean_bps'] or 0:.2f}] "
                      f"drift={v['drift_bps'] or 0:.2f} entry={v['entry_bps'] or 0:.2f} "
                      f"fresh_med={v['freshness_median_s'] or 0:.1f}s")
            else:
                print(f"  +{h:>3}s: n=0 (no data)")
    print("-" * 110)
    print(f"Global freshness median: {report['frescura_mid_mediana_s']:.2f}s")
    print(f"Report saved to: {report_path}")
    print(f"Duration: {duracion_seg:.2f}s")


if __name__ == "__main__":
    main()
