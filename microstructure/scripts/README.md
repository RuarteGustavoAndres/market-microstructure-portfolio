# scripts

Standalone analysis tools that run against the live TimescaleDB capture
(read-only). They are the exact scripts used to generate the numbers in
this repository's README — no notebooks, no screenshots.

| Script | What it does |
|---|---|
| `markout_binance.py` | Adverse-selection markout at +5s/+30s/+60s per symbol, from `binance_aggtrades` × `binance_book_top5`. Read-only. |

## Running

Requires `psycopg2` and a reachable TimescaleDB with the capture schema.
Set the connection via `PREDIKT_TS_DSN` (environment variable or a local
`.env.capturador` file, not committed):

```bash
export PREDIKT_TS_DSN="postgresql://user:pass@localhost:5434/predikt_ml"
python microstructure/scripts/markout_binance.py --json
```

The script never writes to the database.
