# Model status — weekly re-train (PENDING HUMAN REVIEW)

> Machine-generated on 2026-10-18 17:10 UTC from the live capture DB. This file is the raw
> weekly result; the reviewed number only moves into the README after human
> verification (gate discipline: a number without provenance is marketing).

> **Note (2026-10-24):** the current README headline is the 2026-10-22/23 refresh —
> single-horizon h=300s walk-forward + null test of the delta. Sources:
> `results/walkforward_toxicidad_h300.json`, `results/walkforward_toxicidad_h{1,10,60}.json`,
> `results/nulo_delta_h300.json`. The table below is the 2026-10-19 review, kept
> on record with its provenance.

| Metric | Leak-free re-run (2026-10-19) | Last reviewed (2026-10-16)* |
|---|---|---|
| LightGBM AUC OOS (pooled) | 0.7356 | 0.765 (inflated by horizon + leaks) |
| AUC per temporal fold (range) | 0.675 - 0.796 | - |
| PR-AUC OOS / prevalence | 0.514 / 0.314 | - |
| Labeled rows / markets | 285,764 / 215 | 224,520 / 119 |
| Walk-forward OOS maker P&L, unfiltered | -208.84 USD | -225 (aggressor perspective) |
| Walk-forward OOS maker P&L, filter @0.7 | +111.08 USD | -514 (aggressor perspective) |
| Walk-forward efficiency (maker) | 2.1% | - |
| Per-market GroupKFold upper bound, quantile-0.5 | +512.34 USD, 5/5 folds positive | - |

*2026-10-16 numbers measured the AGGRESSOR side of each fill and used leaky
features (incoming-trade size features, stale snapshots, non-causal scaling);
the maker convention is P&L_maker = -(aggressor markout). The old 2026-10-18
row (0.770 / -8.46 / -536.81) was the leaky run and is superseded.

Artifacts regenerated: `walkforward_toxicidad.json`, `desglose_walkforward.json`
(this directory, overwritten weekly). Full log on the capture host.

**Verdict rule (unchanged):** the trading gate stays CLOSED until the OOS
walk-forward maker P&L with the filter is positive AND the threshold is stable
across folds (with queue position and costs modeled). Status: OOS maker P&L is
positive (+111 temporal / +512 per-market) but WFE 2.1% and thresholds are
unstable - the gate stays CLOSED. An AUC improvement alone never reopens it.
