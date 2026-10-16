#!/usr/bin/env python3
"""Backtest Audit Demo — the teardown, in a box.

A working demonstration of the backtest-audit service: it builds a synthetic
"client backtest" with KNOWN, PLANTED biases, then runs the audit checks and
quantifies each one. The reported Sharpe collapses once the strategy is forced
to obey reality: tradeable information, book-priced fills, out-of-sample tuning.

What it shows:
  - A backtest can report Sharpe 30+ and still be worthless.
  - Every claim must be checked against the plumbing, not the numbers.

Usage:
    python3 demo/audit_demo.py            # prints verdicts
    python3 demo/audit_demo.py --write    # also writes results/AUDIT_DEMO.md

Deterministic (fixed seed). Standard library only — no dependencies.
"""
import json
import math
import os
import random
import sys

SEED = 7
rng = random.Random(SEED)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# ──────────────────────────────────────────────────────────────────────
# 1. Synthetic market: 90 days of 1-minute bars, geometric random walk.
#    By construction it contains NO edge: past prices say nothing about
#    future returns. Any strategy "finding" edge here is an artifact.
# ──────────────────────────────────────────────────────────────────────
N_MIN = 60 * 24 * 90
P0 = 0.50
SIGMA = 0.0020

px = [P0]
for _ in range(N_MIN - 1):
    px.append(px[-1] * math.exp(rng.gauss(0.0, SIGMA)))
HALF_SPREAD = 0.004                     # book half-spread (bid/ask around mid)

# ──────────────────────────────────────────────────────────────────────
# 2. The "client backtest" — planted biases
#    (a) look-ahead: the "indicator" at bar t is the return of bar t+1 —
#        information that does not exist yet when the trade is placed
#    (b) fills at mid: no spread, no fees
#    (c) threshold tuned in-sample, on the very window being reported
# ──────────────────────────────────────────────────────────────────────
signal = [(px[i + 1] / px[i] - 1.0) / SIGMA for i in range(N_MIN - 1)] + [0.0]

def run_backtest(sig, thr, entry_lag=0, price_book=False, indices=None):
    """Long/short on |signal|>thr, 30-minute hold.

    entry_lag=0    : enter at bar t  (naive: trades on bar t+1's return)
    entry_lag=1    : enter at bar t+1 (first EXECUTABLE bar — you can only
                     trade the signal one bar after it is knowable)
    price_book     : pay the spread — round trip costs 2 x half-spread,
                     symmetric for longs and shorts
    indices        : restrict reported trades to a bar range (walk-forward)
    """
    n = len(sig)
    hold = 30
    pnl = []
    for t in range(n - hold - 1):
        if abs(sig[t]) <= thr:
            continue
        pos = 1.0 if sig[t] > 0 else -1.0
        e_t = t + entry_lag
        x_t = t + entry_lag + hold
        if x_t >= n:
            continue
        e, x = px[e_t], px[x_t]
        cost = 2 * HALF_SPREAD if price_book else 0.0
        pnl_t = pos * (x - e) - cost
        if indices is None or (indices[0] <= t < indices[1]):
            pnl.append(pnl_t)
    return pnl

def metrics(trades):
    tr = [t for t in trades if t != 0]
    if len(tr) < 3:
        return {"n": len(tr), "mean": 0.0, "sharpe": 0.0, "win": 0.0}
    m = sum(tr) / len(tr)
    sd = math.sqrt(sum((t - m) ** 2 for t in tr) / (len(tr) - 1))
    return {
        "n": len(tr),
        "mean": m,
        "sharpe": m / sd * math.sqrt(252 * 26),   # per-30-min -> annualized
        "win": sum(1 for t in tr if t > 0) / len(tr),
    }

def tune_in_sample(sig, lo=0.3, hi=2.0, steps=18):
    best_thr, best_sh = lo, -9e9
    for i in range(steps):
        thr = lo + i * (hi - lo) / (steps - 1)
        sh = metrics(run_backtest(sig, thr))["sharpe"]
        if sh > best_sh:
            best_sh, best_thr = sh, thr
    return best_thr

best_thr = tune_in_sample(signal)                 # (c) tuned on the reported window
pnl_naive = run_backtest(signal, best_thr)        # (a)+(b) untouched
m_naive = metrics(pnl_naive)

# ──────────────────────────────────────────────────────────────────────
# 3. The audit — same signal, honest plumbing
# ──────────────────────────────────────────────────────────────────────
findings = []

# CHECK 1 — decision-time computability. The signal at bar t uses bar t+1's
# return. You cannot trade it at bar t; the earliest executable entry is t+1.
pnl_exec = run_backtest(signal, best_thr, entry_lag=1)
m_exec = metrics(pnl_exec)
findings.append({
    "check": "Decision-time computability (signal uses data that does not exist yet)",
    "verdict": "FAIL" if m_exec["mean"] < m_naive["mean"] / 2 else "PASS",
    "detail": ("The indicator at bar t is bar t+1's return. Entering at bar t "
               "trades information that does not exist yet. Forced to enter at "
               "the first executable bar (t+1): Sharpe %.2f -> %.2f, mean trade "
               "%.4f -> %.4f. The 'edge' was unexecutable information."
               % (m_naive["sharpe"], m_exec["sharpe"], m_naive["mean"], m_exec["mean"])),
})

# CHECK 2 — book pricing: round trips pay the spread, longs and shorts alike.
pnl_book = run_backtest(signal, best_thr, entry_lag=1, price_book=True)
m_book = metrics(pnl_book)
findings.append({
    "check": "Fills priced at the book (spread on every round trip)",
    "verdict": "FAIL" if m_book["mean"] < 0 else "PASS",
    "detail": ("With honest fills, mean trade %.4f -> %.4f: exactly the spread "
               "(2 x %dbps). On a market with no edge, every round trip is a "
               "donation to the market makers."
               % (m_exec["mean"], m_book["mean"], HALF_SPREAD * 1e4)),
})

# CHECK 3 — in-sample tuning: honest version tunes on H1, reports H2.
tune_end = len(signal) // 2
thr_oos = tune_in_sample(signal[:tune_end])
pnl_oos = run_backtest(signal, thr_oos, entry_lag=1, price_book=True,
                       indices=(tune_end, len(signal)))
m_oos = metrics(pnl_oos)
findings.append({
    "check": "In-sample parameter tuning (threshold fit on the reported window)",
    "verdict": "FAIL" if m_naive["sharpe"] > m_oos["sharpe"] * 2 else "PASS",
    "detail": ("Tuned in-sample: Sharpe %.2f reported. Walk-forward (tune on "
               "first half, report second half, honest plumbing): Sharpe %.2f. "
               "Tuning on the reported window is fitting the answer."
               % (m_naive["sharpe"], m_oos["sharpe"])),
})

# CHECK 4 — random K-fold CV on temporal data: shuffling "validates" the
# look-ahead pipeline. This is the trap most backtests never notice.
idx = list(range(len(pnl_naive)))
rng.shuffle(idx)
kfold = []
for f in range(5):
    fold = [pnl_naive[i] for i in idx[f::5]]
    if fold:
        kfold.append(metrics(fold)["sharpe"])
kf = sum(kfold) / len(kfold)
findings.append({
    "check": "Random K-fold CV on temporal data",
    "verdict": "FAIL" if abs(kf) > 0.5 else "PASS",
    "detail": ("Shuffled 5-fold 'validation' reports Sharpe %.2f for the "
               "look-ahead pipeline. It happily validates unexecutable "
               "information. Time-series validation must respect time order "
               "and embargo." % kf),
})

# CHECK 5 — survivorship: drop the worst decile of trades, watch it jump.
# Applied to the executable pipeline (mean ~ 0): removing the worst tail is
# exactly what an outcome-filtered universe silently does to the report.
tr = sorted(t for t in pnl_exec if t != 0)
n_drop = max(1, len(tr) // 10)
m_surv = metrics(tr[n_drop:])            # drop the WORST decile (list is ascending)
findings.append({
    "check": "Survivorship (universe filtered by outcome)",
    "verdict": "FAIL" if m_surv["mean"] > 0 else "PASS",
    "detail": ("Dropping the worst decile of trades flips mean trade "
               "%.4f -> %.4f. If the asset universe was chosen knowing which "
               "markets survived, the report is self-fulfilling."
               % (m_exec["mean"], m_surv["mean"])),
})

# ──────────────────────────────────────────────────────────────────────
# 4. Self-check: on a market with no edge, the honest pipeline must be a
#    spread-donor. If it is not, the demo itself is broken.
# ──────────────────────────────────────────────────────────────────────
assert m_book["mean"] < 0, "honest pipeline should lose the spread, got %r" % m_book["mean"]
assert abs(m_book["mean"] + 2 * HALF_SPREAD) < 1e-3, \
    "honest mean should be ~ -(2 x half_spread), got %r" % m_book["mean"]
assert m_oos["mean"] < 0, "walk-forward should also lose the spread, got %r" % m_oos["mean"]

# ──────────────────────────────────────────────────────────────────────
# 5. Report
# ──────────────────────────────────────────────────────────────────────
def fmt(m):
    return "n=%d  mean=%.4f  Sharpe=%.2f  win=%.0f%%" % (
        m["n"], m["mean"], m["sharpe"], m["win"] * 100)

L = []
L.append("# Backtest Audit Demo — planted biases, quantified\n")
L.append("Deterministic synthetic backtest (seed %d, 90 days of 1-min bars,\n"
         "standard-library-only script: [`demo/audit_demo.py`](../demo/audit_demo.py)).\n" % SEED)
L.append("The market is a pure random walk: it contains **no edge by construction**.\n"
         "The 'client backtest' below has biases planted on purpose.\n")
L.append("## The backtest as the client would report it\n")
L.append("```\n%s   (threshold %.2f tuned in-sample, fills at mid, trades on future data)\n```\n"
         % (fmt(m_naive), best_thr))
L.append("## Audit findings\n")
for i, f in enumerate(findings, 1):
    L.append("### %d. %s — **%s**\n" % (i, f["check"], f["verdict"]))
    L.append("%s\n" % f["detail"])
L.append("## What survives the audit\n")
L.append("```\nhonest walk-forward:  %s   (threshold %.2f)\n```\n" % (fmt(m_oos), thr_oos))
L.append("> The reported Sharpe was an artifact of the plumbing. On a market with no\n"
         "> edge, an honest backtest loses exactly the spread. That gap — between what\n"
         "> the backtest claims and what survives the audit — is what we measure.\n"
         "> This demo is the checklist; the service runs it against YOUR backtest,\n"
         "> with your data, your fills, your costs.\n")

report = "\n".join(L)
print(report)
print("naive:", json.dumps(m_naive))
print("honest:", json.dumps(m_oos))

if "--write" in sys.argv:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "AUDIT_DEMO.md")
    with open(out, "w") as fh:
        fh.write(report)
    print("written:", out)