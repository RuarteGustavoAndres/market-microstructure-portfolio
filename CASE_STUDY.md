# Case Study — Backtest Audit Service

**Portfolio:** market-microstructure-portfolio
**Service:** Backtest audit for trading systems
**Audience:** a quant/dev who wants to audit their own backtest

## Problem

A backtest with a positive paper P&L is not evidence of an edge. Most "profitable" backtests are leaking: the model has seen the future, the labels are contaminated, or the walk-forward is not actually out-of-sample. A trader who ships such a backtest to live capital is trading a measurement artifact, not a strategy. The question is not "does it make money on paper?" but "is this number a real, reproducible, out-of-sample result — or a leak?"

## Method

The audit is a fixed, pre-registered procedure, not a judgment call. **Input: the trade list exactly as the client reports it** — `(entry_time, direction, pnl)` with their fills, their costs, their universe — plus, where available, per-bar mid prices and the raw signal series.

**What runs on any trade list (always delivered):**

- **Leak detectors that need only the reported fills** — capturability (does the edge survive a 1-bar entry delay?), spread stress (does it survive realistic pricing?), and survivorship (is the reported universe missing its left tail?). Each detector's false-positive rate is measured by Monte Carlo; synthetic price-scale-invariant controls check that the detectors do not fire on structure-free data.
- **Pre-registered null test of P&L.** The decision rule is fixed before any result is computed, then the observed delta is compared against a null distribution.

**What additionally runs when the client submits prices + the signal series** (i.e. a threshold strategy on a signal — the detector's native domain):

- **Re-tuning walk-forward:** the threshold is re-fit on the past half and reported on the future (in-sample tuning check).
- **Validation-protocol check:** shuffled K-fold vs temporal + embargo comparison.

If a client's backtest is not a threshold strategy over a signal, those two checks are reported as **SKIP with a stated reason** — never as a pass, and the scope is agreed before the audit starts. Strict temporal walk-forward (h = 300 s, 1 fill = 1 sample, no shuffling) applies to everything. **Reproducibility is a deliverable:** folds are re-run and compared against the artifact.

## Findings

On the reference pipeline (the portfolio's own pipeline, which the method was developed against), the audit produces:

- **Walk-forward, h = 300 s, 1 fill = 1 sample:** unfiltered P&L −106.75 USD, filtered @0.7 −7.51 USD, delta +99.24 USD GROSS.
- **Pre-registered null test (rule fixed a priori):** observed delta +98.23 vs p95 of the random-filter null +45.27 — rejected at 5%, 4/5 folds positive. Permutation-label control: −1.10 ± 10.21.
- **Central caveat (this is what makes the audit credible):** the IC block-bootstrap CI is [−20.45, +121.74] and includes 0. The positive delta is *not* proven as a robust gain. The trading gate is **CLOSED** and declared as such.
- **Toxicity classifier:** AUC 0.7276 pooled OOS out-of-time (PR-AUC 0.4963 vs prevalence 0.304); folds reproduced on an independent re-run — AUCs identical to 3–4 decimals, unfiltered P&L identical in all 5 folds, filtered P&L differs ~1% from GPU histogram non-determinism, verdict unchanged (seed 42).
- **Audit of the auditor:** 5 leak detectors with Monte Carlo-measured FP rates — 8,000 edge-free null strategies (8 types × 1,000 seeds): union FP 2.39% (191/8,000), Wilson 95% CI [2.08, 2.75]; every detector ≤ the 10% FP budget; power 100% (union) on all three look-ahead traps. **Known limitation, measured:** an honest strategy with a small positive edge *net of declared costs* and ≥200 trades enforced (observed min 2,808) is flagged 7.2% (72/1,000) [CI 5.8, 9.0] — under the 10% budget, above the 5% nominal, almost all from the temporal kfold-CV check reacting to a real (small) edge; separating a real small edge from noise takes more history than 5 CV blocks provide.

**Honest bottom line:** the pipeline does *not* have an edge yet, and the audit says so. What is being sold is the *method* of audit — the detectors, the pre-registered nulls, the reproducibility — not a profitable model.

## Deliverable

A backtest audit report containing:

1. Leak scan: the detectors that apply to your trade list — the three fill-level detectors (capturability, spread stress, survivorship) always; the two signal-level detectors (in-sample re-tuning, validation protocol) when your backtest is a threshold strategy over a signal, **SKIP with a stated reason otherwise**.
2. Pre-registered null test of the P&L.
3. Verified reproducibility: folds re-run and compared against the artifact.
4. Report with findings and severities — including the caveat section when a result is not proven robust.

Before the audit starts, the scope is confirmed in writing: which detectors apply to your backtest type and what inputs each one needs, so there are no surprise SKIPs in the final report.

## Price and timeline

- **Price:** fixed, $250–500 USD per backtest.
- **Timeline:** delivered in 5 business days.

## Contratame

You have a backtest with a positive paper P&L and no way to tell whether the number is real or a leak. I will run the same pre-registered audit on it and report, in writing, whether the result is a leak or a reproducible out-of-sample number — including the caveat that if it is a leak, I will say so.

- **Email:** gustavoaruarte@gmail.com
- **GitHub:** open an issue on the market-microstructure-portfolio repo.

Fixed price, 5 business days, no profitability promises — just a measured answer.
