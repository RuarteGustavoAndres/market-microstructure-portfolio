# Backtest Audit Demo — planted biases, quantified

Deterministic synthetic backtest (seed 7, 90 days of 1-min bars,
standard-library-only script: [`demo/audit_demo.py`](../demo/audit_demo.py)).

The market is a pure random walk: it contains **no edge by construction**.
The 'client backtest' below has biases planted on purpose.

## The backtest as the client would report it

```
n=5836  mean=0.0034  Sharpe=34.03  win=68%   (threshold 2.00 tuned in-sample, fills at mid, trades on future data)
```

## Audit findings

### 1. Decision-time computability (signal uses data that does not exist yet) — **FAIL**

The indicator at bar t is bar t+1's return. Entering at bar t trades information that does not exist yet. Forced to enter at the first executable bar (t+1): Sharpe 34.03 -> 1.13, mean trade 0.0034 -> 0.0001. The 'edge' was unexecutable information.

### 2. Fills priced at the book (spread on every round trip) — **FAIL**

With honest fills, mean trade 0.0001 -> -0.0079: exactly the spread (2 x 40bps). On a market with no edge, every round trip is a donation to the market makers.

### 3. In-sample parameter tuning (threshold fit on the reported window) — **FAIL**

Tuned in-sample: Sharpe 34.03 reported. Walk-forward (tune on first half, report second half, honest plumbing): Sharpe -63.46. Tuning on the reported window is fitting the answer.

### 4. Random K-fold CV on temporal data — **FAIL**

Shuffled 5-fold 'validation' reports Sharpe 34.01 for the look-ahead pipeline. It happily validates unexecutable information. Time-series validation must respect time order and embargo.

### 5. Survivorship (universe filtered by outcome) — **FAIL**

Dropping the worst decile of trades flips mean trade 0.0001 -> 0.0017. If the asset universe was chosen knowing which markets survived, the report is self-fulfilling.

## What survives the audit

```
honest walk-forward:  n=3694  mean=-0.0078  Sharpe=-63.46  win=20%   (threshold 1.90)
```

> The reported Sharpe was an artifact of the plumbing. On a market with no
> edge, an honest backtest loses exactly the spread. That gap — between what
> the backtest claims and what survives the audit — is what we measure.
> This demo is the checklist; the service runs it against YOUR backtest,
> with your data, your fills, your costs.
