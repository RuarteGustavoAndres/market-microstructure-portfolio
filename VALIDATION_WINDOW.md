# Reserved validation window (pre-registered)

**Pre-registered: 2026-10-25.** This document freezes, in writing and before
looking at any new data, how the next out-of-sample evaluation of the
fill-toxicity filter will be run. It exists to keep the 2026-11-08 verdict
honest: the decision rule is fixed *before* the window's data exists, and the
result is published whatever it is.

## 1. Freeze date

**2026-10-25.** Any data with timestamp **≥ 2026-10-26T00:00:00Z** was *not*
seen when any model, threshold, or decision rule was chosen. Everything
reported in this repo (results/, READMEs, charts) was computed strictly from
data with ts < 2026-10-26T00:00:00Z.

## 2. Evaluation date (single)

**2026-11-08** — 13 days of fresh capture (2026-10-26T00:00:00Z through
2026-11-08T00:00:00Z, exclusive of the evaluation instant). The evaluation
runs **once**, on that date, on data captured strictly inside that window.

## 3. Decision rule (pre-registered, unchanged from 2026-10-23)

Identical to the rule pre-registered on 2026-10-23 for the h=300s walk-forward
(null test in `results/nulo_delta_h300.json`):

- **Scheme:** single horizon h=300s, one fill = one sample, 300s embargo
  between folds; temporal folds over the fresh window.
- **Threshold:** fixed at **0.7** — the same fixed threshold used in the
  published walk-forward. No per-fold re-optimization.
- **Re-tuning:** only on data strictly *older* than the window (ts <
  2026-10-26T00:00:00Z). Nothing in the window may inform any choice.
- **Statistic:** pooled P&L delta of the fixed-0.7 filter over unfiltered
  (GROSS, same tape-fill assumptions as the published walk-forward).
- **Rejection rule:** reject the "no edge" null at the 5% level iff
  **delta pooled ≥ p95 of the random-filter null (null A)** **and**
  **delta > 0 in ≥ 4 of 5 folds**. The null A baseline (p95 = +45.27 USD on
  the 2026-10-22 dataset, `results/nulo_delta_h300.json`) is re-drawn on the
  fresh window's own folds with the same fixed seed and same per-fold
  retention, so the comparison stays within-dataset.

## 4. Publication

The result is published **once, unconditionally** — green or red:

- machine-readable: `results/validacion_reservada_2026-11-08.json`
- one line in the READMEs (EN + ES) next to this document.

No re-runs, no "one more check", no threshold changes after seeing the
window. If the rule rejects, the gate may be reconsidered under
`PLAN_SIM_COLA.md`; if it does not, the gate stays closed and the negative
result stands as the headline.

## 5. What NOT to do until 2026-11-08

- **No** re-tuning of thresholds, features, or model hyperparameters using
  any data with ts ≥ 2026-10-26T00:00:00Z.
- **No** peeking at metrics computed on the window (P&L, AUC, hit rate,
  markout) before the evaluation date.
- **No** changing this document's decision rule after 2026-10-25. If a
  legitimate correction is needed, it is appended as a dated addendum *before*
  2026-11-08, never by editing the original rule.
- The capture pipeline itself keeps running normally (it is not a
  model-facing decision); only *analysis* of the window is reserved.

---
*Scope: this pre-registration covers the toxicity filter's economic verdict
on fresh data. It does not cover the capture pipeline, the backtest-audit
service, or the markout report, which keep their own update cadence.*
