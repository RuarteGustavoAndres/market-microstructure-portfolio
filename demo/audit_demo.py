#!/usr/bin/env python3
"""Backtest Audit Demo — the teardown, in a box.

Two synthetic clients, one market (a pure random walk with NO edge by
construction):

  TRAP client  — the classic broken backtest: look-ahead signal, fills at
                 mid, threshold tuned in-sample on the reported window,
                 "validated" with shuffled K-fold CV, and a trade universe
                 quietly filtered by outcome (worst decile dropped).
  CLEAN client — the same market, audited honestly: causal signal, first
                 executable entry, spread paid, walk-forward tuning,
                 temporal validation with embargo, full trade list.

The audit runs five checks on each. On the TRAP client every check fires
with a formal statistical test (not a hand-picked threshold); on the CLEAN
client every check passes — the audit punishes broken plumbing, not losing
strategies. On a no-edge market an honest backtest loses exactly the spread,
and says so.

Each check is a detector with a formal test (PASO 4a-4b of the audit):

  1. Capturabilidad       — every reported trade re-executed with a 1-bar
                            delay, same direction (permutation test,
                            H1: reported > delayed). FAIL if the edge
                            VANISHES with one bar of latency; WARN if it
                            falls sharply but survives (look-ahead and fast
                            alpha are indistinguishable from prices+trades
                            alone — ask the client for signal-entry
                            timestamps); PASS if the edge survives latency.
  2. Fills at mid         — the reported edge re-priced with the spread;
                            one-sided sign-flip test that the charged mean
                            is <= 0. Firing only if the client CLAIMS an
                            edge (a reported loss is not a leak).
  3. In-sample tuning     — reported window vs a real walk-forward (tune on
                            the past half only), one-sided permutation test.
  4. Shuffled K-fold CV   — the client's shuffle "validation" vs a real
                            temporal partition: contiguous blocks, hold-length
                            embargo, threshold re-tuned on the past only;
                            one-sided permutation test. Shuffling P&L and
                            averaging fold Sharpes is commutative — it
                            returns the full-sample number and validates
                            nothing.
  5. Survivorship         — tail-symmetry test on the reported trade list
                            (exact binomial): under a roughly symmetric P&L
                            null the deep tails beyond median +/- 1.5*IQR hold
                            the same number of trades. A missing LEFT tail
                            means the universe was filtered by outcome.
                            Trimming the worst decile of any sample always
                            raises its mean — the trim itself proves nothing;
                            the missing tail is the evidence.

Sharpe is annualized from NON-OVERLAPPING trades: a 30-minute hold sampled
every bar overlaps itself ~30x, and averaging overlapping trade returns
inflates the effective sample size (and the Sharpe with it).

CLI (the service, in a box) — audit the client's own data:
    python3 demo/audit_demo.py --csv trades.csv --prices prices.csv --spread 0.0004

  trades CSV  (one row per reported trade):
      entry_bar,exit_bar,direction,entry_price,exit_price
  prices CSV  (one row per bar; the optional `signal` column enables the
  re-tuning checks 3 and 4):
      bar,price[,signal]

  --spread is the HALF-spread per side (a round trip costs 2 x spread).
  --fills-net-of-costs : the client already reports fills net of costs —
                         check 2 is SKIPPED (charging twice would be our bug).
  --stop-loss          : the strategy uses stops — the left tail is truncated
                         by design, check 5 is SKIPPED (asymmetry expected).
  Without --prices only the checks that need no re-execution run (5);
  the rest report SKIP. The audit re-derives the canonical rule (enter when
  |signal| > threshold, direction = sign(signal), hold = median holding
  period) — a production service would parse the client's declared ruleset.
  SCOPE: checks 3 and 4 re-tune a threshold over a standardized signal
  (grid 0.3-2.0, annualization for 30-min holds) — they only apply to that
  strategy class; for other strategies they report SKIP in production use.

Demo mode:
    python3 demo/audit_demo.py            # prints TRAP vs CLEAN verdicts
    python3 demo/audit_demo.py --write    # also writes results/AUDIT_DEMO.md
    python3 demo/audit_demo.py --seed 4   # vary the market + permutation seed

Deterministic for a given seed (SEED = 7 by default, override with --seed).
Standard library only — no dependencies.
"""
import argparse
import csv
import json
import math
import os
import random
import sys

SEED = 7
ALPHA = 0.01          # verdict threshold for every statistical test
N_PERM = 999          # permutation resamples per test
POOL_CAP = 1500       # max trades per pool fed to a permutation test

N_DAYS = 90
P0 = 0.50
SIGMA = 0.0020        # 1-minute bar volatility
HOLD = 30             # holding period, in 1-minute bars
HALF_SPREAD = 0.004   # half-spread of the synthetic book

# 24/7 market: 48 non-overlapping 30-minute trades per day, 365 days/year.
ANNUALIZATION = math.sqrt(365 * 48)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


# ──────────────────────────────────────────────────────────────────────
# Statistics — stdlib only, deterministic (seeded permutations)
# ──────────────────────────────────────────────────────────────────────
def _mean(xs):
    return sum(xs) / len(xs)


def _sd(xs):
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _subsample(xs, cap, rng):
    xs = list(xs)
    return rng.sample(xs, cap) if len(xs) > cap else xs


def perm_test_two_pool(a, b, rng, n_perm=N_PERM):
    """One-sided permutation test. H0: mean(a) == mean(b). H1: mean(a) > mean(b).

    Exact under exchangeability; with overlapping trades the sampling units
    are not perfectly independent — noted in the report, acceptable for a
    detector at ALPHA = 1% with effects this large. Returns p in (0, 1],
    or None if either pool is too small to test."""
    a = _subsample(a, POOL_CAP, rng)
    b = _subsample(b, POOL_CAP, rng)
    na, nb = len(a), len(b)
    if na < 30 or nb < 30:
        return None
    obs = _mean(a) - _mean(b)
    pool = a + b
    total = sum(pool)
    labels = [1.0] * na + [-1.0] * nb
    c1 = 0.5 * (1.0 / na + 1.0 / nb)          # diff = c1*S_diff + c2*S_total
    c2 = 0.5 * (1.0 / na - 1.0 / nb)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(labels)
        s_diff = sum(l * x for l, x in zip(labels, pool))
        if c1 * s_diff + c2 * total >= obs:
            hits += 1
    return (1 + hits) / (n_perm + 1)


def perm_test_mean(xs, mu0, side, rng, n_perm=N_PERM):
    """One-sided sign-flip permutation test. H0: mean(xs) == mu0.
    side='greater': H1 mean > mu0; side='less': H1 mean < mu0.
    Assumes the centered values are symmetric under H0 (stated per check).
    Returns p in (0, 1], or None if too few trades."""
    xs = _subsample(xs, POOL_CAP, rng)
    n = len(xs)
    if n < 30:
        return None
    obs = _mean(xs) - mu0
    hits = 0
    for _ in range(n_perm):
        s = sum(x if rng.random() < 0.5 else -x for x in xs)
        m = s / n
        if (side == "greater" and m >= obs) or (side == "less" and m <= obs):
            hits += 1
    return (1 + hits) / (n_perm + 1)


def binom_cdf_le(k, n):
    """P(X <= k) for X ~ Bin(n, 0.5). Exact to n=500, normal approx above."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if n <= 500:
        return sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    mu = n / 2.0
    sd = math.sqrt(n) / 2.0
    return 0.5 * (1.0 + math.erf((k + 0.5 - mu) / (sd * math.sqrt(2.0))))


def tail_symmetry_test(pnls):
    """Survivorship detector (check 5). Under a roughly symmetric P&L null,
    the trades beyond median - d and median + d (d = 1.5*IQR) are a binomial
    coin-flip: a missing deep-LEFT tail means the universe was filtered by
    outcome. Returns (n_left, n_right, p) or None if too few trades."""
    xs = sorted(p for p in pnls if p != 0)
    n = len(xs)
    if n < 200:
        return None
    med = xs[n // 2]
    q1, q3 = xs[n // 4], xs[(3 * n) // 4]
    d = 1.5 * (q3 - q1)
    if d <= 0:
        return None
    nl = sum(1 for x in xs if x < med - d)
    nr = sum(1 for x in xs if x > med + d)
    return nl, nr, binom_cdf_le(nl, nl + nr)


def _fmt_p(p, n_perm):
    if p is None:
        return "n/a"
    if p <= 1.0 / n_perm:
        return "p <= %.0e" % p
    return "p = %.3f" % p


# ──────────────────────────────────────────────────────────────────────
# Synthetic market and re-executable strategy plumbing
# ──────────────────────────────────────────────────────────────────────
def build_market(n_days=N_DAYS, sigma=SIGMA, p0=P0, seed=SEED):
    """n_days of 1-minute bars, geometric random walk. No edge by
    construction: past prices say nothing about future returns."""
    rng = random.Random(seed)
    n = 60 * 24 * n_days
    px = [p0]
    for _ in range(n - 1):
        px.append(px[-1] * math.exp(rng.gauss(0.0, sigma)))
    return px


def look_ahead_signal(px, sigma=SIGMA):
    """The planted bias: the 'indicator' at bar t is bar t+1's return —
    information that does not exist yet when the trade is placed."""
    return [(px[i + 1] / px[i] - 1.0) / sigma for i in range(len(px) - 1)] + [0.0]


def causal_signal(px, lookback=5, sigma=SIGMA):
    """A honest signal: momentum over the PAST `lookback` bars only. On a
    random walk it has no predictive value — the honest control case."""
    out = [0.0]
    for t in range(1, len(px)):
        out.append((px[t] / px[t - lookback] - 1.0) / (sigma * math.sqrt(lookback)))
    return out


def run_backtest(px, sig, thr, entry_lag=0, price_book=False, half_spread=0.0,
                 indices=None, hold=HOLD):
    """Long/short on |signal|>thr, `hold` bars. Returns trades as
    (decision_bar, direction, pnl).

    entry_lag=0 : enter at the decision bar (naive: trades on bar t+1's return)
    entry_lag=1 : enter at the first EXECUTABLE bar (you can only trade the
                  signal one bar after it is knowable)
    price_book  : pay the spread — a round trip costs 2 x half-spread
    indices     : restrict reported trades to a decision-bar range
    """
    n = len(sig)
    out = []
    for t in range(n - hold - 1):
        if abs(sig[t]) <= thr:
            continue
        pos = 1.0 if sig[t] > 0 else -1.0
        e_t, x_t = t + entry_lag, t + entry_lag + hold
        if x_t >= n:
            continue
        pnl = pos * (px[x_t] - px[e_t]) - (2 * half_spread if price_book else 0.0)
        if indices is None or (indices[0] <= t < indices[1]):
            out.append((t, pos, pnl))
    return out


def nonoverlapping_pnls(trades, hold=HOLD):
    """Greedy selection of trades that do not overlap in time (entry >=
    previous exit): the sample Sharpe is computed on these, not on the
    overlapping full list."""
    out, last_exit = [], -1
    for t, _pos, pnl in sorted(trades):
        if t >= last_exit:
            out.append(pnl)
            last_exit = t + hold
    return out


def metrics(trades, hold=HOLD):
    """Headline Sharpe = non-overlapping trades (honest annualization,
    PASO 4c); `sharpe_all` keeps the uncorrected number for comparison."""
    tr = [p for _t, _pos, p in trades if p != 0]
    if len(tr) < 3:
        return {"n": len(tr), "mean": 0.0, "sharpe": 0.0,
                "sharpe_all": 0.0, "win": 0.0}
    m, sd = _mean(tr), _sd(tr)
    sharpe_all = m / sd * ANNUALIZATION
    no = [p for p in nonoverlapping_pnls(trades, hold) if p != 0]
    if len(no) >= 30:
        sharpe = _mean(no) / _sd(no) * ANNUALIZATION
    else:
        sharpe = sharpe_all
    return {"n": len(tr), "mean": m, "sharpe": sharpe,
            "sharpe_all": sharpe_all, "win": sum(1 for t in tr if t > 0) / len(tr)}


def tune_threshold(px, sig, indices=None, entry_lag=1, price_book=True,
                   half_spread=0.0, lo=0.3, hi=2.0, steps=18):
    """Grid-search the entry threshold. The AUDIT always tunes with honest
    plumbing (first executable entry, spread paid) and, for walk-forward
    use, on a restricted past window."""
    best_thr, best_sh = lo, -9e9
    for i in range(steps):
        thr = lo + i * (hi - lo) / (steps - 1)
        sh = metrics(run_backtest(px, sig, thr, entry_lag=entry_lag,
                                  price_book=price_book,
                                  half_spread=half_spread, indices=indices))["sharpe"]
        if sh > best_sh:
            best_sh, best_thr = sh, thr
    return best_thr


# ──────────────────────────────────────────────────────────────────────
# The audit engine — five detectors, each with a formal statistical test
# ──────────────────────────────────────────────────────────────────────
def run_audit(booked, prices=None, sig=None, half_spread=0.0, hold=HOLD,
              rng=None, n_perm=N_PERM, fills_net_of_costs=False,
              stop_loss=False):
    """Audit a client's reported trade list.

    booked : list of (entry_bar, direction, pnl) exactly as the client
             reports them (their fills, their costs, their universe).
    prices : per-bar mid prices (re-execution checks 1/3/4). None = SKIP.
    sig    : per-bar signal series, if the client submitted one (re-tuning
             checks 3/4). None = SKIP.
    half_spread : half-spread charged by check 2 (0 = SKIP).
    fills_net_of_costs : True = client's fills already net of costs —
             check 2 SKIPPED (charging twice would be our bug).
    stop_loss : True = strategy uses stops — left tail truncated by design,
             check 5 SKIPPED (asymmetry expected, not survivorship).

    Returns five findings: {"check", "verdict" (FAIL/PASS/SKIP), "detail"}.
    Verdict = FAIL only when the formal test rejects at ALPHA *and* the
    effect runs in the damning direction.
    """
    if rng is None:
        rng = random.Random(SEED)
    m_booked = metrics(booked, hold)
    pnls = [p for _t, _pos, p in booked]
    f = []

    # CHECK 1 — capturabilidad: does the edge survive a 1-bar entry delay?
    # Reworked twice (round 2 + 3, review Claude 19-oct):
    #  - testing bar+1 edge vs 0 flagged REAL short-term edges (false +);
    #  - the share fix had a UNITS bug: relative return / price-unit P&L is
    #    inflated by 1/px (fires 2x hard at px=0.5, never fires above $1).
    #  Final form: re-execute every reported trade ONE bar later, same
    #  direction, SAME price units as the reported P&L:
    #    FAIL  — delayed edge <= 0: only the bar the signal saw paid.
    #    WARN  — delayed edge falls sharply but survives: look-ahead vs fast
    #            alpha is indistinguishable from prices+trades alone → ask
    #            the client for signal-entry timestamps.
    #    PASS  — edge survives latency (what a client can actually trade).
    if prices is None:
        f.append({"check": "Capturabilidad del edge (re-ejecución con 1 bar de retraso)",
                  "verdict": "SKIP",
                  "detail": "Needs the price series (--prices) to re-execute."})
    else:
        n = len(prices)
        rep, dly = [], []
        for t, pos, pnl in booked:
            k = t + hold + 1
            if k >= n:
                continue
            rep.append(pnl)
            dly.append(pos * (prices[k] - prices[t + 1]))  # same price units
        obsR = _mean(rep) if rep else 0.0
        obsD = _mean(dly) if dly else 0.0
        p = perm_test_two_pool(rep, dly, rng, n_perm)
        ratio = obsD / obsR if obsR > 0 else 1.0
        if p is not None and p <= ALPHA and obsR > 0 and obsD <= 0:
            verdict, extra = "FAIL", (
                " Con 1 bar de retraso el edge DESAPARECE: solo paga la "
                "barra que la señal pudo ver — la dirección anticipa "
                "información que no existía al decidir.")
        elif p is not None and p <= ALPHA and obsR > 0 and ratio < 0.5:
            verdict, extra = "WARN", (
                " Con 1 bar de retraso el edge cae a %.0f%% del reportado: "
                "no se distingue look-ahead de alfa rápido con solo "
                "precios+trades — pedir timestamps de las entradas de la "
                "señal." % (100.0 * ratio))
        else:
            verdict, extra = "PASS", (
                " El edge sobrevive a 1 bar de retraso: es capturable a "
                "latencia realista (o no hay edge que perder).")
        f.append({
            "check": "Capturabilidad del edge (re-ejecución con 1 bar de retraso)",
            "verdict": verdict,
            "detail": ("Edge reportado %+.5f/trade; re-ejecutado 1 bar más "
                       "tarde (misma dirección, mismo horizonte) %+.5f "
                       "(%s, %d permutations, H1: reportado > retrasado)."
                       "%s"
                       % (obsR, obsD, _fmt_p(p, n_perm), n_perm, extra)),
        })

    # CHECK 2 — book pricing: does the claimed edge survive the spread?
    if half_spread <= 0:
        f.append({"check": "Fills priced at the book (spread on every round trip)",
                  "verdict": "SKIP",
                  "detail": "No spread supplied (--spread); the round-trip "
                            "cost cannot be charged."})
    elif fills_net_of_costs:
        f.append({"check": "Fills priced at the book (spread on every round trip)",
                  "verdict": "SKIP",
                  "detail": "Client declares fills already NET of costs "
                            "(--fills-net-of-costs); charging the spread "
                            "again would double-count — our bug, not theirs."})
    else:
        charged = [p - 2 * half_spread for p in pnls]
        p = perm_test_mean(charged, 0.0, "less", rng, n_perm)
        mean_charged = _mean(charged)
        fail = (p is not None and p <= ALPHA
                and m_booked["mean"] > 0 and mean_charged <= 0)
        if fail:
            extra = (" The reported edge does not survive the book: every "
                     "round trip was a donation to the market makers.")
        elif m_booked["mean"] <= 0:
            extra = (" The client reports no positive edge to begin with — a "
                     "reported loss is not a leak, nothing to catch here "
                     "(the small p only says the charged mean sits below "
                     "zero, as expected).")
        else:
            extra = " The edge survives the spread."
        f.append({
            "check": "Fills priced at the book (spread on every round trip)",
            "verdict": "FAIL" if fail else "PASS",
            "detail": ("Round trip costs 2 x %d bps. Reported mean trade "
                       "%.4f; spread-charged %.4f (%s, %d permutations, "
                       "H0: charged mean = 0).%s"
                       % (half_spread * 1e4, m_booked["mean"], mean_charged,
                          _fmt_p(p, n_perm), n_perm, extra)),
        })

    # CHECK 3 — in-sample tuning: re-tune on the past, report the future.
    if prices is None or sig is None:
        f.append({"check": "In-sample parameter tuning (threshold fit on the reported window)",
                  "verdict": "SKIP",
                  "detail": "Needs prices + the signal series to re-tune the "
                            "threshold walk-forward."})
    else:
        half = len(sig) // 2
        thr_oos = tune_threshold(prices, sig, indices=(0, half),
                                 half_spread=half_spread)
        oos = run_backtest(prices, sig, thr_oos, entry_lag=1, price_book=True,
                           half_spread=half_spread, indices=(half, len(sig)))
        m_oos = metrics(oos, hold)
        p = perm_test_two_pool(pnls, [x[2] for x in oos], rng, n_perm)
        fail = (p is not None and p <= ALPHA
                and m_oos["mean"] < m_booked["mean"])
        f.append({
            "check": "In-sample parameter tuning (threshold fit on the reported window)",
            "verdict": "FAIL" if fail else "PASS",
            "detail": ("SCOPE: applies only to threshold-over-signal "
                       "strategies (grid and annualization are demo-tuned). "
                       "Walk-forward: tune the threshold on the first half "
                       "only (thr %.2f), report the second half with honest "
                       "plumbing: Sharpe %.2f, mean trade %.4f — vs reported "
                       "Sharpe %.2f, mean %.4f (%s, %d permutations).%s"
                       % (thr_oos, m_oos["sharpe"], m_oos["mean"],
                          m_booked["sharpe_all"], m_booked["mean"],
                          _fmt_p(p, n_perm), n_perm,
                          "" if not fail else
                          " Tuning on the reported window is fitting the "
                          "answer.")),
        })

    # CHECK 4 — validation protocol: shuffled K-fold vs temporal + embargo.
    if prices is None or sig is None:
        f.append({"check": "Random K-fold CV on temporal data",
                  "verdict": "SKIP",
                  "detail": "Needs prices + the signal series to run the "
                            "embargoed temporal validation."})
    else:
        idx = list(range(len(booked)))
        rng.shuffle(idx)
        folds = [metrics([booked[i] for i in idx[j::5]], hold)["sharpe"]
                 for j in range(5) if idx[j::5]]
        kf_claim = _mean(folds) if folds else 0.0
        n = len(sig)
        k = 5
        edges = [i * n // k for i in range(k + 1)]
        audit_cv = []
        for i in range(k):
            lo, hi = edges[i], edges[i + 1]
            if i == 0:
                thr_b = 1.0                    # no history: neutral default
            else:
                thr_b = tune_threshold(prices, sig,
                                       indices=(0, max(lo - hold, 1)),
                                       half_spread=half_spread)  # embargo = hold
            audit_cv.extend(run_backtest(prices, sig, thr_b, entry_lag=1,
                                         price_book=True,
                                         half_spread=half_spread,
                                         indices=(lo, hi)))
        m_cv = metrics(audit_cv, hold)
        p = perm_test_two_pool(pnls, [x[2] for x in audit_cv], rng, n_perm)
        fail = (p is not None and p <= ALPHA
                and m_cv["mean"] < m_booked["mean"])
        f.append({
            "check": "Random K-fold CV on temporal data",
            "verdict": "FAIL" if fail else "PASS",
            "detail": ("SCOPE: applies only to threshold-over-signal "
                       "strategies (grid and annualization are demo-tuned). "
                       "Shuffled 5-fold 'validation' reports Sharpe %.2f — "
                       "but shuffling P&L and averaging fold Sharpes is "
                       "commutative: it returns the full-sample number and "
                       "validates nothing. The audit's validation: contiguous "
                       "temporal blocks, hold-length embargo, threshold "
                       "re-tuned on the past only, honest plumbing: Sharpe "
                       "%.2f, mean trade %.4f (%s, %d permutations).%s"
                       % (kf_claim, m_cv["sharpe"], m_cv["mean"],
                          _fmt_p(p, n_perm), n_perm,
                          "" if not fail else
                          " Random K-fold on temporal data happily validates "
                          "what honest execution denies.")),
        })

    # CHECK 5 — survivorship: is the reported universe missing its left tail?
    if stop_loss:
        f.append({"check": "Survivorship (universe filtered by outcome)",
                  "verdict": "SKIP",
                  "detail": "Strategy uses stops (--stop-loss): the left "
                            "tail is truncated BY DESIGN — tail asymmetry "
                            "here is expected, not survivorship evidence."})
        ts = None
    else:
        ts = tail_symmetry_test(pnls)
    if ts is None and not stop_loss:
        f.append({"check": "Survivorship (universe filtered by outcome)",
                  "verdict": "PASS" if len(pnls) < 200 else "SKIP",
                  "detail": "Too few trades (%d) to test tail symmetry."
                            % len(pnls)})
    elif ts is not None:
        nl, nr, p = ts
        fail = p <= ALPHA and nl < nr
        f.append({
            "check": "Survivorship (universe filtered by outcome)",
            "verdict": "FAIL" if fail else "PASS",
            "detail": ("Reported trade list (n=%d). Deep tails beyond "
                       "median +/- 1.5*IQR: left %d vs right %d trades "
                       "(exact binomial, %s).%s"
                       % (len(pnls), nl, nr, _fmt_p(p, n_perm),
                          "" if not fail else
                          " The left tail is missing: a universe filtered by "
                          "outcome. Trimming the worst decile of any sample "
                          "always raises its mean — the trim itself is "
                          "arithmetic; the missing tail is the evidence.")),
        })
    return f, m_booked


# ──────────────────────────────────────────────────────────────────────
# The two demo cases (PASO 4d): a trap client and a clean control
# ──────────────────────────────────────────────────────────────────────
def drop_worst_decile(trades, frac=10):
    """Plant the survivorship bias: the client's universe was quietly
    filtered by outcome — the worst decile never made it into the report."""
    tr = sorted(trades, key=lambda x: x[2])
    return tr[max(1, len(tr) // frac):]


def build_cases(rng, n_days=N_DAYS, seed=SEED):
    """Build the trap client and the clean control on the same market."""
    px = build_market(n_days=n_days, seed=seed)

    # TRAP: look-ahead signal, threshold tuned in-sample on the full window,
    # fills at mid, and a worst-decile-filtered trade universe.
    sig = look_ahead_signal(px)
    best_thr = tune_threshold(px, sig, entry_lag=0, price_book=False,
                              half_spread=0.0)
    trap_booked = drop_worst_decile(run_backtest(px, sig, best_thr))

    # CLEAN: causal signal, threshold tuned on the first half only, first
    # executable entries, spread paid, full trade list reported.
    csig = causal_signal(px)
    half = len(px) // 2
    thr_clean = tune_threshold(px, csig, indices=(0, half),
                               half_spread=HALF_SPREAD)
    clean_booked = run_backtest(px, csig, thr_clean, entry_lag=1,
                                price_book=True, half_spread=HALF_SPREAD,
                                indices=(half, len(px)))
    return {
        "px": px,
        "trap": {"booked": trap_booked, "sig": sig, "thr": best_thr,
                 "desc": ("threshold %.2f tuned in-sample, fills at mid, "
                          "trades on future data, worst decile dropped"
                          % best_thr)},
        "clean": {"booked": clean_booked, "sig": csig, "thr": thr_clean,
                  "desc": ("threshold %.2f tuned walk-forward (first half "
                           "only), spread paid, causal signal" % thr_clean)},
    }


def self_check(trap_findings, clean_findings, m_trap, m_clean):
    """The demo's own invariant (PASO 4d): the trap client must be CAUGHT by
    every check (FAIL, or WARN where look-ahead vs fast alpha is
    indistinguishable — capturabilidad); the clean control must pass every
    one. If not, the demo is broken."""
    bad = [f["check"] for f in trap_findings if f["verdict"] == "PASS"]
    assert not bad, "trap client must not PASS any check, passed: %r" % bad
    bad = [f["check"] for f in clean_findings if f["verdict"] != "PASS"]
    assert not bad, "clean control must pass every check, failed: %r" % bad
    assert m_clean["mean"] < 0, \
        "honest pipeline on a no-edge market should lose the spread, got %r" % m_clean["mean"]
    assert abs(m_clean["mean"] + 2 * HALF_SPREAD) < 2e-3, \
        "honest mean should be ~ -(2 x half_spread), got %r" % m_clean["mean"]


# ──────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────
def _fmt(m):
    return ("n=%d  mean=%.4f  Sharpe=%.2f (overlap-corrected)  win=%.0f%%"
            % (m["n"], m["mean"], m["sharpe"], m["win"] * 100))


def build_report(cases):
    """cases: list of (title, client_desc, findings, m_reported)."""
    L = []
    L.append("# Backtest Audit Demo — planted biases, quantified\n")
    L.append("Deterministic synthetic market (seed %d, %d days of 1-min bars, "
             "standard-library-only script: "
             "[`demo/audit_demo.py`](../demo/audit_demo.py)).\n"
             % (SEED, N_DAYS))
    L.append("The market is a pure random walk: it contains **no edge by "
             "construction**. Two clients report on it: a TRAP client with "
             "every classic bias planted on purpose, and a CLEAN control "
             "with honest plumbing. The audit runs the same five checks on "
             "both — each one a detector with a formal statistical test "
             "(permutation / exact binomial at alpha=%.2f), not a "
             "hand-picked threshold.\n" % ALPHA)
    for i, (title, desc, findings, m) in enumerate(cases, 1):
        L.append("## Case %d — %s\n" % (i, title))
        L.append("### As the client reports it\n")
        L.append("```\n%s   (%s)\n```\n" % (_fmt(m), desc))
        L.append("### Audit findings\n")
        for j, fd in enumerate(findings, 1):
            L.append("%d. %s — **%s**\n   %s\n" % (j, fd["check"], fd["verdict"], fd["detail"]))
    L.append("## What the demo proves\n")
    L.append("> The audit **detects**: the trap client — Sharpe %s and a "
             "plausible win rate — gets flagged by every check (4 FAIL + "
             "1 WARN: the capturabilidad check reports WARN where look-ahead "
             "and fast alpha are indistinguishable from prices+trades, and "
             "asks for signal-entry timestamps); the clean control — a "
             "losing strategy on a no-edge market — passes every one. The "
             "audit punishes broken plumbing, not losing strategies: an "
             "honest backtest of a no-edge market loses exactly the spread "
             "and says so.\n" % _sh(cases[0][3]))
    L.append("> Every claim must be checked against the plumbing, not the "
             "numbers. This demo is the checklist; the service runs it "
             "against YOUR backtest — `--csv` takes your trades and signal "
             "series, your fills, your costs.\n")
    return "\n".join(L)


def _sh(m):
    return "%.2f" % m["sharpe"]


# ──────────────────────────────────────────────────────────────────────
# CLI: audit a client's CSV (the sellable service)
# ──────────────────────────────────────────────────────────────────────
def load_trades_csv(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("trades CSV %s has no rows" % path)
    need = {"entry_bar", "exit_bar", "direction", "entry_price", "exit_price"}
    missing = need - set(rows[0])
    if missing:
        raise SystemExit("trades CSV missing columns: %s (need: %s)"
                         % (sorted(missing), sorted(need)))
    def _int(v):
        return int(float(v))     # tolerate "1.0"-style integers

    booked = []
    holds = []
    for r in rows:
        t, x, pos = (_int(r["entry_bar"]), _int(r["exit_bar"]),
                     _int(r["direction"]))
        booked.append((t, pos, pos * (float(r["exit_price"]) - float(r["entry_price"]))))
        holds.append(x - t)
    holds.sort()
    return booked, holds[len(holds) // 2]


def load_prices_csv(path):
    """bar,price[,signal] -> (prices, signals or None)."""
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("prices CSV %s has no rows" % path)
    cols = set(rows[0])
    has_sig = "signal" in cols
    nb = max(int(r["bar"]) for r in rows)
    px = [0.0] * (nb + 1)
    sig = [0.0] * (nb + 1) if has_sig else None
    for r in rows:
        b = int(r["bar"])
        px[b] = float(r["price"])
        if has_sig:
            sig[b] = float(r["signal"])
    return px, sig


# ──────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(
        description="Backtest audit demo: TRAP vs CLEAN, or --csv to audit "
                    "a client's own trades CSV.")
    ap.add_argument("--csv", help="client trades CSV (entry_bar,exit_bar,"
                                  "direction,entry_price,exit_price)")
    ap.add_argument("--prices", help="client prices CSV (bar,price[,signal])")
    ap.add_argument("--spread", type=float, default=0.0,
                    help="half-spread per side, in price units "
                         "(round trip = 2 x spread)")
    ap.add_argument("--n-perm", type=int, default=N_PERM,
                    help="permutation resamples per test (default %d)" % N_PERM)
    ap.add_argument("--seed", type=int, default=SEED,
                    help="seed for the synthetic market and the permutation "
                         "tests (default %d)" % SEED)
    ap.add_argument("--fills-net-of-costs", action="store_true",
                    help="client fills are already net of costs: SKIP check 2")
    ap.add_argument("--stop-loss", action="store_true",
                    help="strategy uses stops: SKIP check 5 (left tail by design)")
    ap.add_argument("--write", action="store_true",
                    help="also write the report to results/")
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)

    if args.csv:
        booked, hold = load_trades_csv(args.csv)
        px = sig = None
        if args.prices:
            px, sig = load_prices_csv(args.prices)
        findings, m = run_audit(booked, px, sig, half_spread=args.spread,
                                hold=hold, rng=rng, n_perm=args.n_perm,
                                fills_net_of_costs=args.fills_net_of_costs,
                                stop_loss=args.stop_loss)
        report = build_report([("Client submission — %s" % os.path.basename(args.csv),
                                "as reported by the client", findings, m)])
        print(report)
        if args.write:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            out = os.path.join(RESULTS_DIR, "AUDIT_CLIENT.md")
            with open(out, "w") as fh:
                fh.write(report)
            print("written:", out)
        return 0

    # Demo mode: trap vs clean on the same synthetic market.
    cases_raw = build_cases(rng, seed=args.seed)
    trap_f, m_trap = run_audit(cases_raw["trap"]["booked"], cases_raw["px"],
                               cases_raw["trap"]["sig"],
                               half_spread=HALF_SPREAD, rng=rng, n_perm=args.n_perm)
    clean_f, m_clean = run_audit(cases_raw["clean"]["booked"], cases_raw["px"],
                                 cases_raw["clean"]["sig"],
                                 half_spread=HALF_SPREAD, rng=rng, n_perm=args.n_perm)
    self_check(trap_f, clean_f, m_trap, m_clean)
    cases = [
        ("the trap client — every classic bias planted on purpose",
         cases_raw["trap"]["desc"], trap_f, m_trap),
        ("the clean control — same market, honest plumbing",
         cases_raw["clean"]["desc"], clean_f, m_clean),
    ]
    report = build_report(cases)
    print(report)
    print("trap:", json.dumps(m_trap))
    print("clean:", json.dumps(m_clean))
    if args.write:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = os.path.join(RESULTS_DIR, "AUDIT_DEMO.md")
        with open(out, "w") as fh:
            fh.write(report)
        print("written:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
