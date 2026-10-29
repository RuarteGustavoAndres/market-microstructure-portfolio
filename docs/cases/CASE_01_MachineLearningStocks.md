# Case 01 — Re-auditing a public ML stock-picking backtest

**Audited:** [`robertmartin8/MachineLearningStocks`](https://github.com/robertmartin8/MachineLearningStocks) at commit `1c3712a` (2021-02-04, MIT licence, unmaintained since Feb 2021). An educational project by Robert Martin.
**Question:** the project's own backtest reports a large edge over the S&P 500. How much of it survives an honest train/test split?
**Note on credit:** the author already warns that `backtesting.py` has "a fatal flaw" and that the result "looks too good to be true". This case does not claim to have discovered a hidden error; it **measures** the disclosed one.

## What the project reports
Sample output in the project's README: accuracy 0.81, precision 0.75, 177 trades, average return 37.8% vs 9.2% for the index, **+28.6 percentage points**.

## What I did
Same model (`RandomForestClassifier`, 100 trees), same 41 features, same label ("beats the S&P 500 by ≥10 points over the next 12 months") and same accounting (buy every stock the model flags; compare with the index over the same rows). **Only the way data is split changes.** The audit itself executes none of the project's code: it reads only the public data file (`keystats.csv`, 8,682 rows, 3,384 after the project's own `dropna`). As an independent anchor, the project's own `backtesting.py` was also run under 8 identical random splits and gives exactly the same accuracy, precision, trade count and edge as our re-implementation, on an identical input matrix ([`anchor.py`](../../cases/01_machinelearningstocks/anchor.py)).

| Split | Precision (base rate) | Return of flagged stocks | Index | Edge (pts) |
|---|---|---|---|---|
| Random 80/20, as in the project (100 repeats; 5–95%: 25.8–33.1) | 0.77 (0.35) | 39.1% | 9.9% | **+29.2** |
| Random, evaluated on the same rows as the time-ordered test | 0.74 (0.31) | 45.7% | 20.1% | +25.7 |
| Held-out **companies** (no ticker in both train and test; dates still mixed) | 0.49 (0.35) | 24.6% | 12.6% | +12.1 |
| **Time-ordered walk-forward, 365-day embargo** | **0.31 (0.31)** | 22.5% | 20.2% | **+2.28** |

*Precision* = share of flagged stocks that really beat the index by ≥10 points. *Base rate* = that share if stocks were picked at random. *Embargo* = a gap between training and test so the 12-month labels cannot overlap. *Walk-forward* = train on the past, test on the following year, move forward.

## Reading
- The random split reproduces the project's published numbers (accuracy 0.79 / precision 0.77 / +29 pts vs 0.81 / 0.75 / +28.6).
- More than half of the apparent edge (+29.2 → +12.1 pts) disappears just by not letting the same companies sit in both train and test; most of the rest disappears when time is not mixed (+2.3 pts).
- With a leak-free split, precision (0.313) is indistinguishable from the base rate (0.306). **No edge is detectable**: the point estimate is +2.3 pts but the interval below includes 0.
- Uncertainty (block bootstrap by calendar month, 35 blocks, 2,000 resamples): edge **+2.28 pts, 95% CI [−3.1, +8.5]**; precision minus base rate **+0.006, CI [−0.060, +0.066]**.

## Limits (what this does *not* show)
- With 291 flagged stocks over four test years (2009, 2011, 2012, 2013) a small real edge (up to about +8 pts, the top of the interval) cannot be ruled out. The right wording is "no edge detected", not "no edge exists".
- The project's backtest silently drops 61% of the rows (`dropna`), and the usable data starts in 2006; years 2006–2008 and 2010 were skipped because they leave too little training data (2006–2008) or too few test rows (2010).
- One model, as shipped, with no tuning. 12-month returns overlap, so the effective sample is smaller than the row count. Survivorship in the company list was not examined.

## Reproduce
```bash
git clone https://github.com/robertmartin8/MachineLearningStocks && cd MachineLearningStocks && git checkout 1c3712a
python3 -I audit_mls.py keystats.csv mls_audit.json      # ~3 min, deterministic
python3 -I boot_mls.py  keystats.csv mls_bootstrap.json  # ~1 min
```
Environment: Python 3.13, scikit-learn 1.9.1, pandas 3.0.5, numpy 2.5.4. A second run reproduced every number exactly. Data file: `keystats.csv` from the repo at that commit, sha256 `7eaf46245c422d2373b9059c1517b36bff631ec45ab7ec5f94d36b92ea1b822f` (8,682 rows, 3,384 after the project's own `dropna`). Files: [`audit_mls.py`](../../cases/01_machinelearningstocks/audit_mls.py), [`boot_mls.py`](../../cases/01_machinelearningstocks/boot_mls.py), [`anchor.py`](../../cases/01_machinelearningstocks/anchor.py), [`mls_audit.json`](../../cases/01_machinelearningstocks/mls_audit.json), [`mls_bootstrap.json`](../../cases/01_machinelearningstocks/mls_bootstrap.json).

## Correction log
- 2026-10-29: the first version fed the model 42 inputs by mistake: the script also included `SP500_p_change` (the index's own 12-month return, which the project's model does not use). Corrected to the project's 41 inputs and every number re-derived; the conclusion (no edge detected with a leak-free split) is unchanged, the point estimates moved (time-ordered edge +0.24 → 2.28 pts).
