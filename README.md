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
3. **Walk-forward verdict (single horizon h=300s): delta positive, random-filter
   null rejected at 5% (block-bootstrap CI wide, includes 0 — both reported) — the filter's P&L delta over unfiltered is **+$99.24**
   on the 2026-10-22 single-horizon refresh (129,175 rows, one fill = one
   sample at h=300s), while the unfiltered level is **−$106.75** and the
   filtered level @0.7 is **−$7.51** (80,379 ops) — the level @0.7 is still
   negative in single horizon (all figures GROSS under tape-fill assumptions:
   every tape fill counted, no queue position, no fees, no costs). The previous **+$489 @0.7**
   level came from the 382,582-row scheme that summed all four horizons of
   every fill (fill×horizon pairs), inflating the level ~4×; the current
   scheme counts one fill once. Queue simulation
   ([PLAN_SIM_COLA.md](PLAN_SIM_COLA.md)) is still pending to close the gate;
   a first queue-aware run (T2/T3) was **inconclusive** (AUC ~0.50-0.56 on
   approximated features, out-of-distribution); the corrected re-run
   (assumption 3 fixed: persistent quotes, no optimistic auto-cancel) gives a
   **positive delta** (+$9.26 GROSS, both folds) with AUC still near chance
   (~0.556) on a tiny sample (1,580 fills) — direction consistent with the
   real data, magnitude uninformative (details below). Walk-forward efficiency is
   **−0.8% per op** (OOS vs in-sample P&L per op at the fixed threshold —
   below the 30% overfitting-warning line), and per-fold optimal thresholds
   are **not stable** (0.6/0.5/0.7/0.6/0.55), so we run the fixed 0.7 filter.
   The gate stays closed.

Live capture status (machine-generated daily from the production database):
[`results/CAPTURE_STATUS.md`](results/CAPTURE_STATUS.md).

Contact: open a GitHub issue or email gustavoaruarte@gmail.com.

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
| `microstructure/sim/sim_cola.py` + `microstructure/sim/sim_maker_dataset.py` | Queue-fill simulation over the REAL capture with three comparable requote modes (**persist** = honest, no auto-cancel / **cancel** = status quo / **contra** = conservative) + its dataset builder — published verbatim from the private bot |
| `microstructure/model/backtest_walkforward_persist.py` | The same walk-forward (documented adaptations only) run on the persist-mode simulated fills — `results/maker_fills_sim_persist.csv` (1.1 MB, included) → `results/walkforward_toxicidad_h300_persist.json`, fully reproducible |
| `microstructure/model/backtest_nulo_delta.py` | Null test of the filter's delta: random-filter null (same per-fold retention) + market-block bootstrap + permuted-label control — published verbatim from the private bot |

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
- **Leak-free re-run:** pooled AUC **0.7384** out-of-fold on unseen markets
  (freshness-capped snapshots, causal standardization, per-market folds).
- **Honest limitations:** the model ranks toxicity (AUC) but binary accuracy
  sits below the majority baseline → it is a scoring/filter layer, not an
  autonomous decision-maker. Under exact-markout single-horizon walk-forward
  (h=300s, 2026-10-22) the delta is **positive and rejects the random-filter
   null at 5%** (null test 2026-10-23 — row below; the block-bootstrap CI is
   wide and includes 0, both reported) — a **+$99.24** GROSS P&L delta over
   unfiltered under tape-fill assumptions —
  but the absolute level @0.7 is **negative** (−$7.51 over 80,379 ops;
  unfiltered −$106.75), per-op walk-forward efficiency is **−0.8%** (below the
  30% overfitting-warning line), thresholds per fold are not stable, and a
  first queue-aware simulation (T2/T3, 2026-10-18) was **inconclusive**: the
  model's AUC on simulated fills is ~0.50-0.56 (approximated features,
  out-of-distribution), so the tape-based numbers assume ideal fills
  ([PLAN_SIM_COLA.md](PLAN_SIM_COLA.md)). **Queue sim re-run with assumption 3
  corrected** (persistent quotes — the old optimistic auto-cancel removed
  exactly the adverse fills a real maker eats; sim engine published verbatim
  in `microstructure/sim/`): the filter's delta over unfiltered is **+$9.26**
  GROSS (unfiltered −$15.57 → filtered @0.7 −$6.31; 2 of 3 temporal folds,
  1,580 fills, one fill = one sample, h=300s), **positive in both folds** —
  the old T2/T3 negative delta (−$682) was an artifact of that auto-cancel
  assumption. AUC on the sim stays near chance (pooled 0.556, folds
  0.533–0.599; features approximated by design) and the sample is ~80×
  smaller than the real dataset, so this reads as DIRECTION, not magnitude —
  the gate stays closed. The previous +$489 @0.7 level came
  from the 382,582-row scheme that summed all four horizons of every fill
  (fill×horizon pairs) — a previous 4×-counted estimate; the current scheme
  counts one fill once. The gate stays closed.

| Metric | Value |
|---|---|
| AUC-ROC pooled OOS (LightGBM, leak-free, unseen markets) | **0.7384** |
| PR-AUC OOS / prevalence | **0.5057 / 0.304** |
| AUC per temporal fold (range, 5 folds, h=300s) | **0.653 / 0.787** |
| Walk-forward rows (single horizon h=300s, 2026-10-22 refresh) | 129,175 total; 80,379 OOS ops at fixed 0.7 |
| **Walk-forward OOS maker P&L (single horizon h=300s, 2026-10-22 refresh, 129,175 rows — GROSS, tape-fill assumptions; queue+costs pending)** | **DELTA-POSITIVE, LEVEL-NEGATIVE, still fragile** — exact-markout, temporal splits, 300s embargo, one fill = one sample: unfiltered **−106.75 USD** · filter @0.7 **−7.51** (80,379 ops) · **delta over unfiltered +99.24** · per-fold-optimal thresholds 0.6/0.5/0.7/0.6/0.55 (**not stable**) → fixed 0.7 filter. WFE **−0.8% per op** (OOS vs in-sample P&L per op; below the 30% overfitting-warning line). Previous 4×-counted estimate (382,582-row scheme, all four horizons per fill): +488.98 @0.7. Gate stays closed pending queue simulation + costs; numbers reported anyway. |
| **Per-horizon walk-forward (one fill = one sample per horizon, fixed 0.7, GROSS tape-fill assumptions, re-runs 2026-10-23)** | delta is positive at **all four** horizons and grows with the horizon — h=1s: **+18.49** (fold AUC 0.637–0.740) · h=10s: **+56.60** (0.682–0.756) · h=60s: **+89.27** (0.666–0.774) · h=300s: **+99.24** (0.653–0.787). The unfiltered LEVEL flips sign with horizon (+77.84 / +101.32 / +40.62 / −106.75) — adverse selection grows with horizon; the delta is the stable quantity. The pooled **AUC 0.7384** above comes from the fill×horizon scheme (382,582 rows); the per-horizon fold AUCs are the ranges in this row (deltas now null-tested — rows below). JSONs: `results/walkforward_toxicidad_h{1,10,60,300}.json`. |
| **Null test of the delta (decision rule fixed a priori on 2026-10-23, before running: single horizon h=300s, threshold 0.7, statistic = delta; reject at 5% if pooled delta ≥ p95 of a random-filter null AND delta > 0 in ≥4/5 folds)** | **REJECTED at 5%.** Observed pooled delta **+98.23** (GPU re-run; CPU reference +99.24) vs random-filter null (same per-fold retention, 1,000 fixed-seed draws) p95 **+45.27** — the observed delta sits above the null's 100th percentile (p99 +52.01, max +61.02), and is > 0 in **4/5 folds**. Control: 50 re-trainings with labels permuted within market → delta **−1.10 ± 10.21** (min −22.23, max +19.09): the pipeline does not manufacture signal. Secondary (threshold re-drawn per draw): observed delta at p98.9 of that null. **Honest caveat:** the market-block bootstrap CI of the delta, **[−20.45, +121.74], includes 0** (high sampling volatility; 2 of 5 folds nearly zero) — both results are reported. Gate stays closed (level negative, WFE negative, queue+costs pending). Code and JSON: `microstructure/model/backtest_nulo_delta.py`, `results/nulo_delta_h300.json`. |
| **The auditor audited: Monte Carlo of the 5 leak detectors (2026-10-23, alpha=0.01, 999 permutations/test)** | Over 200 edge-free null strategies (4 types × 50 seeds, 90-day markets, spread 2×40 bps): **false-positive rate per detector — capturabilidad 0%, fills-at-book 0%, tuning-in-sample 0%, kfold-cv 1.5%, survivorship 5.0%; union 6.0%**. **Power: 100% (union) on all three look-ahead traps** (full look-ahead caught by tuning/kfold/survivorship; 1-bar look-ahead by all five; 50% partial look-ahead: capturabilidad 86%). An honest but tiny edge is flagged **0%** of the time — the detectors catch plumbing, not the existence of edge. Known limitation: capturabilidad cannot see a full-hold look-ahead (a 1-bar delay removes only 1/30 of the peek); the other four detectors cover it. JSON: `results/monte_carlo_fp_power.json`. |
| **The auditor audited itself: five external strategies × two price scales (2026-10-23, 3 honest + 2 look-ahead traps, p0=0.5 vs p0=100)** | **Scale invariance restored and verified**: the harness now scales the round-trip cost with the market scale (constant 80 bps: `half_spread = HALF_SPREAD * p0 / P0`), and the audit rng is seeded identically per strategy across scales — 5/5 strategies return **identical verdicts at both scales**, pnl ratio exactly 200.0×. Verdicts: the 2 look-ahead traps FAIL/WARN in both scales (tuning-in-sample, kfold, fills-at-book all fire); the 3 honest strategies PASS except `spread_sniper`, flagged by survivorship at both scales — a **known detector limitation, documented in the JSON** (`limitaciones_conocidas`): every losing trade pays the same fixed round-trip cost (a point mass at −2·half_spread), which compresses the IQR and mechanically inflates the right-tail outlier count (`nr > nl` in all 7 flagged of 24 market seeds; 29.2% false-flag rate vs alpha=0.01; honest controls momentum/ruido: 0/16 flagged; same rate with relative pnl — not a units artifact). Study script + JSON: `demo/estudio_limitacion_supervivencia.py`, `results/estudio_limitacion_supervivencia.json`. The false edge previously reported for `momentum_chaser` at p0=100 (mean +0.086/trade) came from an additive spread over lognormal prices — the cost-scaling fix removes it. |

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
- ~~Backtest the toxicity filter against P&L under tape-fill assumptions~~ **done on leak-free features
  (freshness-capped snapshots, causal standardization, temporal folds) —
  under the strict single-horizon walk-forward (h=300s, one fill = one
  sample, 300s embargo) the delta is positive (random-filter null rejected
  at 5%; block-bootstrap CI wide, includes 0 — both reported):
  **+$99.24 GROSS P&L delta over unfiltered** (level @0.7: −$7.51; gross,
  tape-fill assumptions; 2026-10-22 refresh, 129,175 rows). The earlier
  4×-counted scheme (fill×horizon pairs, 382,582 rows) had shown a +$376
  delta / +$489 level — same qualitative story, inflated level. WFE per op
  is **−0.8%** and per-fold optimal thresholds remain **not stable**
  (0.6/0.5/0.7/0.6/0.55); queue simulation (PLAN_SIM_COLA.md) is still
  pending — the gate stays closed.** AUC ranks flow; converting it into
  profitable maker P&L (queue position, rewards, fees) remains open work.
- Investigate why ranking skill does not translate fully: per-horizon/per-regime
  P&L, queue-position-aware simulation, cost model calibration.

## License
MIT — see [LICENSE](LICENSE). Use freely, attribution appreciated.

## Author
Gustavo Ruarte — market microstructure & ML engineering.
Contact: open a GitHub issue or email gustavoaruarte@gmail.com.
