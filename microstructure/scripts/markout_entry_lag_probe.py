#!/usr/bin/env python3
"""Entry-term origin probe: is the flat entry ≈ −0.5 bps a TIMING OFFSET
or REAL price impact? (re-audit Claude 2026-10-28, item: the 10e7d1d probe)

The staleness probe (results/markout_entry_staleness_probe.json) shows the
entry term stays ≈ −0.5 bps for mids of 0.2–2 s of age — it does not grow
with age, which refutes pure exit-mid staleness. But a flat floor is also
exactly what a FIXED timestamp offset between trades and book snapshots
would produce: the mid used at trade time could be the mid measured just
before/after the aggressor arrives by an ingest-latency artifact.

Test (cheap, decisive): recompute the entry term while SHIFTING the mid
lookup by delta seconds.  For each offset d the entry is recomputed as

    entry = sign * (mid(t + d) - price)      [bps of mid]

with mid(t) = last snapshot at or before t.  Sign convention and matching
are identical to microstructure/scripts/markout_binance.py.

Interpretation:
  - offset > 0 emulates SNAPSHOT stamps being delta seconds late
    (the mid we use is the one measured delta earlier, real time).
  - offset < 0 emulates TRADE stamps being |delta| seconds late
    (compensating an ingest-latency lag in the aggTrades stream).
  - If entry(delta) collapses toward the median half-spread (~the cost of
    crossing the spread) for some |delta| <= max tested, a fixed lag
    explains the floor: the claim must read "origin not identified".
  - If entry(delta) stays near the offset-0 value across the whole sweep
    (both directions), a fixed lag is EXCLUDED and the entry term is real
    adverse selection at the fill.

Also reports, for offset 0, the entry-vs-mid-age stratification (the same
reading as the staleness probe) so the artifact and the probe are
cross-checkable in one run.

Read-only: the script never writes to the database.

Usage:
    python markout_entry_lag_probe.py [--since ...] [--until ...]
        [--symbols BTCUSDT] [--offsets -0.5,-0.3,-0.1,0,0.1,0.3,0.5] [--json]

Requires: psycopg2, PREDIKT_TS_DSN (or a local .env.capturador file next to
this script), binance_aggtrades + binance_book_top5 populated.
"""
import argparse
import json
import os
import statistics
import sys
import time
from bisect import bisect_right
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

AGE_BINS = (0.5, 1.0, 2.0, 5.0)  # same stratification as the staleness probe


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
    con = psycopg2.connect(DSN)
    con.autocommit = True
    cur = con.cursor()
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
    mids_por_sym, half_spread_por_sym = {}, {}
    for sym, t, bid, ask in cur.fetchall():
        mids_por_sym.setdefault(sym, []).append((float(t), float(bid) / 2.0 + float(ask) / 2.0))
        mid = float(bid) / 2.0 + float(ask) / 2.0
        if mid > 0:
            half_spread_por_sym.setdefault(sym, []).append(
                10000.0 * (float(ask) - float(bid)) / 2.0 / mid)
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
    return trades_por_sym, mids_por_sym, half_spread_por_sym


def analyze_symbol(trades, mids, half_spread_bps, offsets):
    """Entry term as a function of the mid-lookup offset, stdlib only.

    Returns {offset: {n, mean/median entry_bps, buy/sell, age strata (0 only)}}
    plus the snapshot median half-spread bps as the crossing-cost reference."""
    mt = [t for t, _ in mids]
    mv = [v for _, v in mids]
    acc = {d: {"buy": [], "sell": [], "ages": [], "all": []} for d in offsets}
    n_no_mid = 0
    for t, price, aggr in trades:
        sign = -1 if aggr == "sell" else 1
        for d in offsets:
            i = bisect_right(mt, t + d) - 1
            if i < 0:
                continue
            mid = mv[i]
            if mid <= 0:
                continue
            entry = 10000.0 * sign * (mid - price) / mid
            bucket = acc[d]
            bucket["all"].append(entry)
            if aggr == "buy":
                bucket["buy"].append(entry)
            elif aggr == "sell":
                bucket["sell"].append(entry)
            if d == 0:
                bucket["ages"].append(t - mt[i])
        n_no_mid += 0  # per-offset miss counting kept simple: see n below
    del n_no_mid

    def stats(vals):
        return {
            "n": len(vals),
            "mean_bps": sum(vals) / len(vals) if vals else None,
            "median_bps": statistics.median(vals) if vals else None,
        }

    out = {}
    for d in offsets:
        b = acc[d]
        row = {
            "all": stats(b["all"]),
            "buy": stats(b["buy"]),
            "sell": stats(b["sell"]),
        }
        if d == 0:
            # entry vs mid age, offset 0 (cross-check with the staleness probe)
            strata = {}
            for lo, hi in zip((0,) + AGE_BINS, AGE_BINS + (float("inf"),)):
                vals = [e for e, a in zip(b["all"], b["ages"]) if lo <= a < hi]
                s = stats(vals)
                s["edad_media_s"] = (sum(a for e, a in zip(b["all"], b["ages"])
                                          if lo <= a < hi) / len(vals)
                                     if vals else None)
                strata["%g-%gs" % (lo, hi) if hi != float("inf") else ">%gs" % lo] = s
            row["por_edad_mid_s"] = strata
        out[d] = row
    return out, stats(half_spread_bps)


def main():
    t0 = time.time()
    p = argparse.ArgumentParser(
        description="Entry-term origin probe: timing offset vs real impact")
    p.add_argument("--since", type=str, default=None)
    p.add_argument("--until", type=str, default=None)
    p.add_argument("--symbols", type=str, default="BTCUSDT,ETHUSDT,SOLUSDT")
    p.add_argument("--offsets", type=str, default="-0.5,-0.3,-0.1,0,0.1,0.3,0.5",
                   help="mid-lookup offsets in seconds (see docstring)")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    import datetime as _dt
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    offsets = [float(x) for x in args.offsets.split(",")]

    if args.since:
        since = parse_ts(args.since)
    else:
        since = None
    if args.until:
        until = parse_ts(args.until)
    else:
        until = _dt.datetime.now(_dt.timezone.utc).timestamp()

    con = psycopg2.connect(DSN)
    con.autocommit = True
    cur = con.cursor()
    if since is None:
        cur.execute("""
          SELECT LEAST(
            (SELECT MIN(ts) FROM binance_aggtrades),
            (SELECT MIN(ts) FROM binance_book_top5))
        """)
        row = cur.fetchone()
        if not row or not row[0]:
            raise RuntimeError("no data")
        since = row[0].timestamp() if hasattr(row[0], "timestamp") else float(row[0])
    con.close()

    trades, mids, hs = cargar(since, until, symbols)
    resumen = {}
    for sym in sorted(trades):
        r, hs_stats = analyze_symbol(trades[sym], mids.get(sym, []),
                                     hs.get(sym, []), offsets)
        resumen[sym] = {
            "trades_input": len(trades[sym]),
            "median_half_spread_bps": hs_stats,
            "por_offset_s": r,
        }

    report = {
        "window": {"since": to_iso(since), "until": to_iso(until)},
        "symbols_usados": sorted(trades.keys()),
        "offsets_s": offsets,
        "resumen_por_simbolo": resumen,
        "lectura": (
            "entry(delta) plano en +-0.5 s excluye el desfase fijo como "
            "origen del piso de entry; entry(delta) que colapsa hacia el "
            "medio spread confirma el desfase. Ref: median_half_spread_bps."),
        "duracion_seg": round(time.time() - t0, 3),
    }

    out_path = os.path.join(REPO_ROOT, "results", "markout_entry_lag_probe.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    if args.json:
        print(json.dumps(report, indent=2))
        return
    print("window:", report["window"]["since"], "->", report["window"]["until"])
    for sym, r in resumen.items():
        print("== %s == trades=%d  median half-spread = %.4f bps"
              % (sym, r["trades_input"],
                 r["median_half_spread_bps"]["median_bps"] or -1))
        print("  offset_s |  mean_entry_bps(all)  median  (buy / sell)")
        for d in offsets:
            row = r["por_offset_s"][d]
            print("  %+7.2f  |   %+9.4f          %+8.4f   (%+.4f / %+.4f)"
                  % (d, row["all"]["mean_bps"] or 0, row["all"]["median_bps"] or 0,
                     row["buy"]["mean_bps"] or 0, row["sell"]["mean_bps"] or 0))
        zero = r["por_offset_s"].get(0.0) or r["por_offset_s"].get(0)
        if zero and zero.get("por_edad_mid_s"):
            print("  entry vs mid age (offset 0):")
            for k, s in zero["por_edad_mid_s"].items():
                print("    %-10s n=%-9d mean=%+.4f bps  edad_media=%.2fs"
                      % (k, s["n"], s["mean_bps"] or 0, s["edad_media_s"] or 0))
    print("saved:", out_path)
    print("duration: %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
