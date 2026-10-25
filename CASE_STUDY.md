# Case Study — Backtest Audit Service

**Portfolio:** market-microstructure-portfolio
**Service:** Backtest audit for trading systems
**Audience:** a quant/dev who wants to audit their own backtest

## Problem

A backtest with a positive paper P&L is not evidence of an edge. Most "profitable" backtests are leaking: the model has seen the future, the labels are contaminated, or the walk-forward is not actually out-of-sample. A trader who ships such a backtest to live capital is trading a measurement artifact, not a strategy. The question is not "does it make money on paper?" but "is this number a real, reproducible, out-of-sample result — or a leak?"

## Method

The audit is a fixed, pre-registered procedure, not a judgment call:

- **Strict temporal walk-forward.** h = 300 s, 1 fill = 1 sample. No shuffling; no future leakage by construction.
- **Pre-registered null test of P&L.** The decision rule is fixed before any result is computed, then the observed delta is compared against a null distribution.
- **Five leak detectors.** Each detector's false-positive rate is measured by Monte Carlo; synthetic price-scale-invariant controls check that the detectors do not fire on structure-free data.
- **Reproducibility as a deliverable.** Folds are re-run and compared byte-for-byte.

## Findings

On the reference pipeline (the portfolio's own pipeline, which the method was developed against), the audit produces:

- **Walk-forward, h = 300 s, 1 fill = 1 sample:** unfiltered P&L −106.75 USD, filtered @0.7 −7.51 USD, delta +99.24 USD GROSS.
- **Pre-registered null test (rule fixed a priori):** observed delta +98.23 vs p95 of the random-filter null +45.27 — rejected at 5%, 4/5 folds positive. Permutation-label control: −1.10 ± 10.21.
- **Central caveat (this is what makes the audit credible):** the IC block-bootstrap CI is [−20.45, +121.74] and includes 0. The positive delta is *not* proven as a robust gain. The trading gate is **CLOSED** and declared as such.
- **Toxicity classifier:** AUC 0.7276 pooled OOS out-of-time (PR-AUC 0.4963 vs prevalence 0.304); folds reproduced byte-identical (CPU, seed 42).
- **Audit of the auditor:** 5 leak detectors with Monte Carlo-measured FP rates (union 1.87% — 131/7,000, Wilson 95% CI [1.58, 2.22] — vs alpha 1%), plus synthetic price-scale-invariant controls.

**Honest bottom line:** the pipeline does *not* have an edge yet, and the audit says so. What is being sold is the *method* of audit — the detectors, the pre-registered nulls, the reproducibility — not a profitable model.

## Deliverable

A backtest audit report containing:
1. Leak scan with the 5 detectors.
2. Pre-registered null test of the P&L.
3. Verified reproducibility.
4. Report with findings and severities.

## Price and timeline

- **Price:** fixed, $250–500 USD per backtest.
- **Timeline:** delivered in 5 business days.

## Contratame

You have a backtest with a positive paper P&L and no way to tell whether the number is real or a leak. I will run the same pre-registered audit on it and report, in writing, whether the result is a leak or a reproducible out-of-sample number — including the caveat that if it is a leak, I will say so.

- **Email:** gustavoaruarte@gmail.com
- **GitHub:** open an issue on the market-microstructure-portfolio repo.

Fixed price, 5 business days, no profitability promises — just a measured answer.
