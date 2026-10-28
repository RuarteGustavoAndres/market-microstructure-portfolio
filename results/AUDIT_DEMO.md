# Backtest Audit Demo — planted biases, quantified

Deterministic synthetic market (seed 7, 90 days of 1-min bars, standard-library-only script: [`demo/audit_demo.py`](../demo/audit_demo.py)).

The market is a pure random walk: it contains **no edge by construction**. Two clients report on it: a TRAP client with every classic bias planted on purpose, and a CLEAN control with honest plumbing. The audit runs the same five checks on both — each one a detector with a formal statistical test (permutation / exact binomial at alpha=0.01), not a hand-picked threshold.

## Case 1 — the trap client — every classic bias planted on purpose

### As the client reports it

```
n=5253  mean=0.0049  Sharpe=98.33 (overlap-corrected)  win=75%   (threshold 2.00 tuned in-sample, fills at mid, trades on future data, worst decile dropped)
```

### Audit findings

1. Edge capturability (re-run with a 1-bar lag) — **WARN**
   Reported edge +0.00494/trade; re-executed 1 bar later (same direction, same horizon) +0.00169 (p <= 1e-03, 999 permutations, H1: reported > delayed). With a 1-bar delay the edge drops to 34% of the reported value: look-ahead cannot be distinguished from fast alpha with prices+trades alone — ask for the signal-entry timestamps.

2. Fills priced at the book (spread on every round trip) — **FAIL**
   Round trip costs 2 x 40 bps. Reported mean trade 0.0049; spread-charged -0.0031 (p <= 1e-03, 999 permutations, H0: charged mean = 0). The reported edge does not survive the book: every round trip was a donation to the market makers.

3. In-sample parameter tuning (threshold fit on the reported window) — **FAIL**
   SCOPE: applies only to threshold-over-signal strategies (grid and annualization are demo-tuned). Walk-forward: tune the threshold on the first half only (thr 1.30), report the second half with honest plumbing: Sharpe -104.18, mean trade -0.0079 — vs reported Sharpe 95.06, mean 0.0049 (p <= 1e-03, 999 permutations). Tuning on the reported window is fitting the answer.

4. Random K-fold CV on temporal data — **FAIL**
   SCOPE: applies only to threshold-over-signal strategies (grid and annualization are demo-tuned). Shuffled 5-fold 'validation' reports Sharpe 95.74 — but shuffling P&L and averaging fold Sharpes is commutative: it returns the full-sample number and validates nothing. The audit's validation: contiguous temporal blocks (block 0 skipped: no tuning history), hold-length embargo, threshold re-tuned on the past only, honest plumbing: Sharpe -119.07, mean trade -0.0080 (p <= 1e-03, 999 permutations). Random K-fold on temporal data happily validates what honest execution denies.

5. Survivorship (universe filtered by outcome) — **FAIL**
   Reported list n=5253; tested on the 2433 non-overlapping (independent) units: deep tails beyond median +/- 1.5*IQR hold left 0 vs right 145 trades (exact binomial, p <= 2e-44). The left tail is missing: a universe filtered by outcome. Trimming the worst decile of any sample always raises its mean — the trim itself is arithmetic; the missing tail is the evidence.

## Case 2 — the clean control — same market, honest plumbing

### As the client reports it

```
n=10336  mean=-0.0078  Sharpe=-106.72 (overlap-corrected)  win=19%   (threshold 1.40 tuned walk-forward (first half only), spread paid, causal signal)
```

### Audit findings

1. Edge capturability (re-run with a 1-bar lag) — **PASS**
   Reported edge -0.00782/trade; re-executed 1 bar later (same direction, same horizon) +0.00018 (p = 1.000, 999 permutations, H1: reported > delayed). The edge survives a 1-bar delay: capturable at realistic latency (or there is no edge to lose).

2. Fills priced at the book (spread on every round trip) — **PASS**
   Round trip costs 2 x 40 bps. Reported mean trade -0.0078; spread-charged -0.0158 (p <= 1e-03, 999 permutations, H0: charged mean = 0). The client reports no positive edge to begin with — a reported loss is not a leak, nothing to catch here (the small p only says the charged mean sits below zero, as expected).

3. In-sample parameter tuning (threshold fit on the reported window) — **PASS**
   SCOPE: applies only to threshold-over-signal strategies (grid and annualization are demo-tuned). Walk-forward: tune the threshold on the first half only (thr 1.40), report the second half with honest plumbing: Sharpe -106.72, mean trade -0.0078 — vs reported Sharpe -104.35, mean -0.0078 (p = 0.289, 999 permutations).

4. Random K-fold CV on temporal data — **PASS**
   SCOPE: applies only to threshold-over-signal strategies (grid and annualization are demo-tuned). Shuffled 5-fold 'validation' reports Sharpe -106.32 — but shuffling P&L and averaging fold Sharpes is commutative: it returns the full-sample number and validates nothing. The audit's validation: contiguous temporal blocks (block 0 skipped: no tuning history), hold-length embargo, threshold re-tuned on the past only, honest plumbing: Sharpe -138.14, mean trade -0.0080 (p = 0.184, 999 permutations).

5. Survivorship (universe filtered by outcome) — **PASS**
   Reported list n=10336; tested on the 1616 non-overlapping (independent) units: deep tails beyond median +/- 1.5*IQR hold left 68 vs right 64 trades (exact binomial, p = 0.668).

## What the demo proves

> The audit **detects**: the trap client — Sharpe 98.33 and a plausible win rate — gets flagged by every check (4 FAIL + 1 WARN: the capturability check reports WARN where look-ahead and fast alpha are indistinguishable from prices+trades, and asks for signal-entry timestamps); the clean control — a losing strategy on a no-edge market — passes every one. The audit punishes broken plumbing, not losing strategies: an honest backtest of a no-edge market loses exactly the spread and says so.

> Every claim must be checked against the plumbing, not the numbers. This demo is the checklist; the service runs it against YOUR backtest — `--csv` takes your trades and signal series, your fills, your costs.
