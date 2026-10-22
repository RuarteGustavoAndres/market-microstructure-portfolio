# Market Microstructure: 24/7 Capture Pipeline + Honest Fill-Toxicity Model

> **EN** | [ES](README.es.md)

An end-to-end market microstructure pipeline for prediction-market CLOBs
(Polymarket-style binary order books) and crypto perpetuals: a 24/7 multi-venue
WebSocket order-book capture into TimescaleDB, plus a machine-learning model
that predicts **fill toxicity (adverse selection)** from order-book features —
and a runnable demo of the backtest audit that keeps the numbers honest. Every
number here is real, out-of-sample — **including the parts that did not work**.

**Headline results**

1. **119M+ order-book events and 645k+ real fills** captured 24/7 into
   TimescaleDB hypertables (12,900+ markets tracked, 9,900+ active in the
   last 7 days).
2. **Toxicity classifier: AUC 0.7384 pooled OOS on unseen markets (folds 0.689-0.786, leak-free); PR-AUC 0.506 vs 0.304 prevalence** — a scoring/filter layer, **not** a decision-maker.
3. **Walk-forward verdict: the filter reduces tape adverse selection in maker
   perspective: **+$489 strict temporal OOS @0.7** (the headline; unfiltered
   went from −209 to **+113** as data grew from 286k to 383k rows);
   +$747 per-market folds (quantile-0.5, 50% retention) is an optimistic
   ceiling — that scheme is out-of-time but not out-of-market; queue simulation
   ([PLAN_SIM_COLA.md](PLAN_SIM_COLA.md)) still pending to close the gate** —
   walk-forward efficiency is 35.4% per row (8.0% per total; rows =
   fill×horizon pairs); per-fold optimal thresholds are still **not stable**
   (0.5-0.7), so we run the fixed 0.7 filter. All four horizons positive
   filtered (+81 h1s / +146 h10s / +167 h60s / +95 h300s). The gate stays
   closed and we keep the verdict.

Live capture status (machine-generated daily from the production database):
[`results/CAPTURE_STATUS.md`](results/CAPTURE_STATUS.md).

Contact: open a GitHub issue.

## Highlights
- **119M+ order-book events** (libro_eventos) + **645k+ fills** (trades_full)
  captured 24/7 into TimescaleDB hypertables — **12,900+ markets tracked**
  (verified 2026-10-22).
- **Hundreds of thousands of order-book events per hour, sustained 24/7**, into
  a TimescaleDB hypertable (compression + retention policies).
- **Toxicity classifier: LightGBM AUC 0.7384 pooled OOS on unseen markets
  (leak-free)**, per-fold range **0.689-0.786** — rows are fill×horizon pairs
  (each fill appears once per horizon), so every P&L table sums each fill 4×.
  The pooled/per-market gap is a difference in base rates, and we declare it.
- Caught and removed **feature-leakage traps** (incoming-trade size, stale snapshots, horizon pooling) that had inflated AUC from 0.736 to 0.765.
- Grounded in the microstructure literature: OFI (Cont 2014), micro-price
  (Stoikov 2018), markout, Glosten-Milgrom / Kyle.
- **Backtest Audit Demo** — a runnable, dependency-free teardown
  ([`demo/audit_demo.py`](demo/audit_demo.py), results in
  [`results/AUDIT_DEMO.md`](results/AUDIT_DEMO.md)): a synthetic backtest with
  planted look-ahead, free fills and in-sample tuning reports Sharpe 34; forced
  to obey reality (tradeable information, book-priced fills, walk-forward
  tuning) it loses exactly the spread. Five audit checks, self-verifying.
- **Venue 2 live: Binance USDT-M perpetuals** (2026-10-16) — same pipeline,
  second asset class: aggTrades with aggressor side + top-5 book + funding for
  BTC/ETH/SOL perps, 24/7 systemd capture. Markout / adverse-selection
  analytics in [`analytics/adverse_selection.py`](analytics/adverse_selection.py).

### Data status (declared, as of 2026-10-22)
- Continuous capture **2026-10-14 → present** (trades stream: no gaps).
- **Known outage:** order-book stream down **2026-10-15 ~17:00–20:00 UTC** (~3h,
  systemd services were not `enabled` and died with an interactive session;
  fixed — all capture services are now `systemctl --user enable`-ed with
  linger).
- Any analysis spanning that window must exclude or flag it. We say so because
  an honest dataset declares its holes.
- **Live status:** [`results/CAPTURE_STATUS.md`](results/CAPTURE_STATUS.md) —
  machine-generated daily from the production database, so what you read is
  always <24h old.

## Architecture
```
WebSocket (market channel, per venue)
   │  asyncio producer → queue → consumer  (network never blocks on processing)
   ▼
In-memory order book per instrument (snapshot + incremental diffs)
   │  micro-batch flush: 500 rows / 2s
   ▼
TimescaleDB hypertables (libro_eventos, trades_full, libro_top5, market_meta
                          + binance_* tables for venue 2)
   │  feature engineering (OFI, microprice_bias, momentum, ...)
   ▼
Fill-toxicity classifier (LightGBM)  →  AUC 0.7384 pooled OOS, unseen markets (leak-free)
```

## Pipeline modules
| Module | Purpose |
|---|---|
| `microstructure/capture/http.py` | ISP-fault-tolerant HTTP + pinned-IP DNS fallback (selective NXDOMAIN resilience) |
| `microstructure/data/tsdb.py` | TimescaleDB connection + micro-batch flush helpers |
| `microstructure/model/features_ml.py` | Feature engineering: OFI, micro-price, multi-horizon momentum, markout labels |
| `microstructure/model/entrenar_toxicidad.py` | Classifier training: GroupKFold by market, per-market z-score, leakage-safe features |
| `microstructure/model/backtest_walkforward.py` | Walk-forward backtest of the toxicity filter: temporal splits, 300s embargo, exact fill P&L — published verbatim from the private bot |
| `microstructure/model/backtest_desglose.py` | OOS P&L breakdown of the walk-forward verdict: by horizon, side, fill aggressiveness, spread + maker-realistic adjustment |

## The model (honest numbers)
- **Target:** markout of each real trade at +horizon (book mid vs fill price).
  Markout < 0 = adversely selected.
- **Multi-horizon labeling:** every fill labeled at 1s/10s/60s/300s. The
  freshness-capped walk-forward dataset is **382,582 fill×horizon rows**
  (268,446 of them out-of-sample; each fill contributes one row per horizon —
  these are fill×horizon *pairs*, not distinct fills).
- **Validation:** GroupKFold by market (unseen-market generalization) +
  per-market z-score normalization.
- **Leakage removed:** `pos_en_spread` and `fill_price` reconstruct the target
  by construction — dropped them, AUC fell from 0.702 to the honest 0.6425
  (MLP baseline).
- **Leak-free re-run:** pooled AUC **0.7384** out-of-fold on unseen markets (freshness-capped snapshots, causal standardization, per-market folds);
  (0.751) — better for tabular microstructure features.
- **Honest limitations:** the model ranks toxicity (AUC) but binary accuracy
  sits below the majority baseline → it is a scoring/filter layer, not an
  autonomous decision-maker. Under exact-markout walk-forward the filter
  **does** produce positive net P&L (+$489 @0.7 on the 2026-10-22 refresh) —
  but thresholds per fold are not stable and queue simulation is still
  pending, so the gate stays closed (below).

| Metric | Value |
|---|---|
| AUC-ROC pooled OOS (LightGBM, leak-free, unseen markets) | **0.7384** |
| PR-AUC OOS / prevalence | **0.5057 / 0.304** |
| AUC per temporal fold (range, 5 folds) | **0.689 / 0.786** |
| Walk-forward rows (freshness-capped, 2026-10-22 refresh) | 382,582 (268,446 OOS) |
| **Walk-forward OOS maker P&L (2026-10-22 refresh, 382,582 rows)** | **POSITIVE, still fragile** — exact-markout, temporal splits, 300s embargo: unfiltered **+113.47 USD** · filter @0.7 **+488.98** (231,334 ops) · @0.8 **+216.82** · per-fold-optimal (0.65/0.5/0.7/0.65/0.65 — **not stable**). WFE **35.4% per row** (8.0% per total; rows = fill×horizon pairs). Per-market GroupKFold optimistic estimate: unfiltered **+186** · quantile-0.5 threshold **+746.93** (retention 50%) · @0.7 **+560.97**. All horizons positive filtered (+81 h1s / +146 h10s / +167 h60s / +95 h300s). Gate stays closed pending queue simulation + costs; numbers reported anyway. |

## Venue 2 — Binance USDT-M perpetuals (live since 2026-10-16)
Second venue feeding the same TimescaleDB stack, same order-book sync protocol
(snapshot + diff with update-id chain verification, gap detection, resync —
the official Binance algorithm): aggTrades with aggressor side, top-5 book at
2s cadence, and funding/mark-price every 5 minutes for BTCUSDT/ETHUSDT/SOLUSDT.
Captured 24/7 by a systemd service (public streams, no account involved).
Event→ingest latency ~0.3s. Execution-quality analytics (pure functions,
tested): [`analytics/adverse_selection.py`](analytics/adverse_selection.py) —
markout at multiple horizons, no look-ahead (the mid used is the last one
known at or before t+h; trades whose horizon passes the last quote are
excluded, never extrapolated).

Venue 1 (Polymarket CLOB, above) = discrete-event prediction markets; this
venue 2 = continuous high-frequency books — same schema, same analytics.

### Markout in bps of mid (2026-10-16 22:54 → 2026-10-18 23:42 UTC, 3,111,633 aggressive trades)

| Symbol | mean bps @5s | median bps @5s | toxic frac @5s | mean bps @300s |
|---|---|---|---|---|
| **BTCUSDT** | +0.22 | +0.06 | 47% | +0.34 |
| **ETHUSDT** | +0.43 | +0.27 | 44% | +0.45 |
| **SOLUSDT** | **−0.06** | **−0.48** | **66%** | −0.02 |

**Decomposition (BTCUSDT @5s): drift +0.75 bps vs entry −0.53 bps.** This
changes the honest reading of the markout: the positive USD markout we first
reported over the early capture window was **market drift, not execution
skill** — the taker still **pays the spread at entry**; and for a maker, the
flow the aggressor wins is adverse selection against it (maker P&L =
−(aggressor markout)). SOL is directly toxic
at short horizons (median −0.48 bps, 66% of taker trades lose).

**Declared limitation — exit-mid freshness:** the exit mid is reconstructed
from a book sampled at ~2s cadence; its median staleness is **1.14s** — up to
~23% of the 5s horizon. Short-horizon markouts are therefore a lower bound on
precision.

Regenerate with `microstructure/scripts/markout_binance.py` (read-only, live
DB; latest machine-readable report in
[`results/markout_report.json`](results/markout_report.json)).

## Next steps (scoped)
- ~~MLP → LightGBM~~ **done** — superseded by the leak-free LightGBM re-run (AUC 0.7384 pooled OOS).
- ~~Backtest the toxicity filter against net P&L~~ **done on leak-free features
  (freshness-capped snapshots, causal standardization, per-market folds) —
  the filter reduces tape adverse
  selection in maker perspective: **+$489 OOS under the strict temporal
  walk-forward @0.7** (300s embargo) — the headline (unfiltered +113 vs
  −209 on the previous refresh, as data grew 286k→383k rows); +$747 under
  per-market GroupKFold quantile-0.5 (retention 50%) is an optimistic
  ceiling, because that scheme is out-of-market but not out-of-time.
  Walk-forward efficiency is **35.4% per row** (8.0% per total) but per-fold
  optimal thresholds remain **not stable** (0.5-0.7) and queue
  simulation (PLAN_SIM_COLA.md) is still pending — the gate stays closed.** AUC ranks flow; converting it
  into net maker P&L (queue position, rewards, fees) remains open work.
- Investigate why ranking skill does not translate fully: per-horizon/per-regime
  P&L, queue-position-aware simulation, cost model calibration.

## License
MIT — see [LICENSE](LICENSE). Use freely, attribution appreciated.

## Author
Gustavo Ruarte — market microstructure & ML engineering.
Contact: open a GitHub issue.
