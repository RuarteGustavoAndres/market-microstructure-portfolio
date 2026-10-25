# Model status — weekly re-train (PENDING HUMAN REVIEW)

> Machine-generated on 2026-10-25 17:30 UTC from the live capture DB. This file is the raw
> weekly result. The reviewed file is results/MODEL_STATUS.md (NOT touched by
> this job): a number only moves there — or into the READMEs — after human
> verification of the artifacts below. Gate discipline: a number without
> provenance is marketing.

| Metric | This week (raw) | Last reviewed (2026-10-16) |
|---|---|---|
| LightGBM AUC out-of-fold | **0.7007** (artifact) — see note below | 0.765 |
| Labeled fills / markets | 530452 / 321 | 224,520 / 119 |
| Walk-forward OOS P&L, unfiltered | -305.57 USD | -225 |
| Walk-forward OOS P&L, filter @0.7 | 292.77 USD | -514 |

Artifacts (this re-train): `results/pending_retrain/modelo_toxicidad_metricas.json`
(auc_oof source), `results/pending_retrain/walkforward_toxicidad.json`,
`results/pending_retrain/desglose_walkforward.json` (+ h300) — copied from the
capture host, unreviewed. Full log: `bot/dataset/retrain_*.log`.

**Flagged for review (2026-10-25):** the training log prints "LightGBM AUC
out-of-fold: 0.745", but the artifact from the same run records
`auc_oof: 0.7007` (per-fold max is 0.7459 — fold 3). The published number is
the artifact's 0.7007; the log line is treated as a reporting bug until
reconciled. Numbers above are NOT reviewed; the 292.77 USD filtered P&L with
an unstable per-fold best threshold (0.5/0.65/0.65/0.8/0.5) does not reopen
the gate on its own.

**Verdict rule (unchanged):** the trading gate stays CLOSED until the OOS
walk-forward P&L with the filter is positive AND the threshold is stable
across folds. An AUC improvement alone never reopens it.
