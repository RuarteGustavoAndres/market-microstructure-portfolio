# Model status — weekly re-train (PENDING HUMAN REVIEW)

> Machine-generated on 2026-10-18 17:10 UTC from the live capture DB. This file is the raw
> weekly result; the reviewed number only moves into the README after human
> verification (gate discipline: a number without provenance is marketing).

| Metric | This week | Last reviewed (2026-10-16) |
|---|---|---|
| LightGBM AUC out-of-fold | 0.770 | 0.765 |
| Labeled fills / markets | 397777 / 218 | 224,520 / 119 |
| Walk-forward OOS P&L, unfiltered | 8.46 USD | -225 |
| Walk-forward OOS P&L, filter @0.7 | -536.81 USD | -514 |

Artifacts regenerated: `walkforward_toxicidad.json`, `desglose_walkforward.json`
(this directory, overwritten weekly). Full log on the capture host.

**Verdict rule (unchanged):** the trading gate stays CLOSED until the OOS
walk-forward P&L with the filter is positive AND the threshold is stable
across folds. An AUC improvement alone never reopens it.
