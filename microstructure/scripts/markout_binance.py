"""Adverse-selection markout on REAL Binance capture data (TimescaleDB).

Reads binance_aggtrades (trades with aggressor side) and rebuilds the mid
from binance_book_top5 (best bid/ask per snapshot), then computes markouts
at +5s/+30s/+60s using the pure analytics module from this repo
(analytics/adverse_selection.py).

Usage:
    python markout_binance.py [--json]

Requires:
    - psycopg2
    - PREDIKT_TS_DSN environment variable (postgresql://user:pass@host:port/db),
      or a local .env.capturador file next to the script with that variable.
    - Binance capture tables populated: binance_aggtrades, binance_book_top5.

Read-only: the script never writes to the database.
"""
import os
import sys

import psycopg2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

# Optional local env file (not committed): PREDIKT_TS_DSN=postgresql://...
from pathlib import Path
_env = Path(os.path.join(os.path.dirname(__file__), ".env.capturador"))
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
DSN = os.environ.get("PREDIKT_TS_DSN")

from analytics.adverse_selection import markouts  # noqa: E402


def cargar():
    con = psycopg2.connect(DSN)
    con.autocommit = True
    cur = con.cursor()
    # mid: bid0/ask0 pair at the SAME ts (same snapshot)
    cur.execute("""
      SELECT b.symbol, EXTRACT(EPOCH FROM b.ts), b.price, a.price
      FROM binance_book_top5 b
      JOIN binance_book_top5 a
        ON a.symbol=b.symbol AND a.ts=b.ts AND a.side='ask' AND a.level=0
      WHERE b.side='bid' AND b.level=0
      ORDER BY 2
    """)
    mids_por_sym = {}
    for sym, t, bid, ask in cur.fetchall():
        mids_por_sym.setdefault(sym, []).append((t, (bid + ask) / 2.0))
    cur.execute("""
      SELECT symbol, EXTRACT(EPOCH FROM ts), price, aggressor
      FROM binance_aggtrades ORDER BY ts
    """)
    trades_por_sym = {}
    for sym, t, px, aggr in cur.fetchall():
        trades_por_sym.setdefault(sym, []).append((t, px, aggr))
    con.close()
    return trades_por_sym, mids_por_sym


def main():
    trades, mids = cargar()
    resumen = {}
    for sym in sorted(trades):
        r = markouts(trades[sym], mids.get(sym, []))
        resumen[sym] = {
            "trades_usados": r[5]["n"] + r[30]["n"] + r[60]["n"] - r[5]["n"],
            "por_horizonte": {
                str(h): {
                    "n": v["n"], "mean": round(v["mean"], 6) if v["mean"] is not None else None,
                    "toxic_frac": round(v["toxic_frac"], 3) if v["toxic_frac"] is not None else None,
                } for h, v in r.items()
            },
        }
    if "--json" in sys.argv:
        import json
        print(json.dumps(resumen, indent=1))
        return
    for sym, s in resumen.items():
        print(f"== {sym} ==")
        for h, v in s["por_horizonte"].items():
            print(f"  +{h}s: n={v['n']} mean={v['mean']} toxic={v['toxic_frac']}")


if __name__ == "__main__":
    main()
