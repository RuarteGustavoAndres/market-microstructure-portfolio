#!/usr/bin/env python3
"""Monte Carlo of the audit DETECTORS — false positive (FP) rate and power.

This is not the null test of the model's P&L (that is `monte_carlo_audit.py`):
here the quality of the audit itself is measured, i.e. the five detectors of
`audit_demo.run_audit` as measuring instruments:

  FP rate (false positives) — N NULL strategies: edge-free market
      (random walk) + signal with no predictive power in EIGHT levels
      (pure iid noise, causal momentum, noise + micro-drift, AR(1) noise,
      declared mechanical stop-loss, mechanical asymmetric take-profit,
      small real fast alpha, frequent net-positive edge), HONEST plumbing
      end to end (walk-forward tuning, first executable entry, paid spread,
      full trade list — the same "clean" client as the demo). Per detector:
      % of null strategies each one flags (FAIL or WARN) with a Wilson 95%
      interval + % flagged by at least one.

  Power (sensitivity) — N TRAP strategies with variable intensity:
        lookahead_completo — the signal sees the FULL hold return
                             (30-bar peek); broken demo plumbing.
        lookahead_1bar     — the exact demo trap (1-bar peek).
        lookahead_50pct    — 1-bar peek on 50% of the trades; in the
                             other 50% the signal is pure noise (same
                             marginals: tuning cannot distinguish them).
        edge_realo_pequeno — NOT a trap: genuine real edge of +10% of the
                             half-spread per trade (predictable component
                             of the return, alpha observable before entry),
                             honest plumbing. Hard "no false positive" case:
                             low detection there is CORRECT behavior, not a
                             failure (the detectors hunt plumbing biases,
                             not the existence of edge).
      The plumbing of the three traps mirrors the demo trap: in-sample
      tuning, fills at mid, worst decile dropped. Per intensity: %
      detected by each detector and by the set.

  The 3 new nulls (P2 24-10-2026):
      stop_loss           — mechanical exits: exits early if the adverse
                            drawdown exceeds 2 x sigma_path; DECLARED
                            (stop_loss=True → check 5 SKIP by design,
                            the left tail is truncated on purpose).
      pagos_asimetricos   — mechanical take-profit at +3 x gross half-spread,
                            losses run to the full hold; NOT
                            declared (mechanical exits, nothing filtered:
                            tests whether check 5 confuses mechanical payoff
                            with survivorship — the truncated tail here is
                            the RIGHT one, not the left).
      alpha_rapido_real   — small real edge (+10% of the half-spread) through
                            the honest pipeline; it is the null case of the
                            detector: a genuinely profitable client should
                            NOT be flagged (identical to the
                            edge_realo_pequeno intensity, counted on the FP
                            side).

  Specificity control (task_3, 27-10-2026) — edge_neto_positivo:
      the original alpha_rapido_real barely operates (815/1000 runs with
      0 trades; the remaining 185 LOSE the spread: gross edge 0.1 x
      half_spread << round-trip cost) — it proves nothing. This null is
      the missing control: an HONEST client with a small real
      NET-POSITIVE edge after costs and >= 200 reported trades. Market with
      real alpha (predictable component = SPEC_S_EPS_FRAC x SIGMA per
      bar, observable with a 2-bar lead), calibrated half-spread
      (SPEC_HALF_SPREAD) so the net edge is positive but small
      (~30% of the round-trip cost), signal normalized by the
      alpha scale, walk-forward tuning, paid spread, full
      list, fills declared NET of costs (--fills-net-of-costs:
      the audit cannot charge the spread twice). A good and
      honest backtest must NOT be flagged: if the MC flags it, the audit
      wrongly accuses a legitimate client (GATE pilot Upwork).

Honesty: if a detector has high FP or low power it is reported as is
— that is exactly what must be known before offering the service.

Deterministic: fixed seeds, all reported in the JSON. Stdlib only.

Parallel + checkpoints (P2): --jobs N splits the runs across N processes
(CPU only, stdlib); each finished run is appended to a JSONL checkpoint
(flush + fsync), so a cut resumes with --resume without repeating work.

Wilson 95% interval (P2b): for k flagged out of n runs with z=1.959964:
    center = (p + z^2/2n) / (1 + z^2/n)
    half   = z*sqrt(p(1-p)/n + z^2/4n^2) / (1 + z^2/n)
With k=0, n=50 the upper bound is ~7.1%; with n=1000, ~1.1%. The interval
is computed in this script and travels to the JSON alongside each rate.

CLI:
    python3 demo/monte_carlo_fp_power.py [--n-perm 999] [--per-type 50]
                                         [--per-intensity 100] [--no-write]
                                         [--jobs N] [--checkpoint PATH]
                                         [--resume]

Writes results/monte_carlo_fp_power.json and a summary table to stdout.
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import audit_demo as ad  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results")

# Five detectors, in the order run_audit emits them.
DETS = ["capturabilidad", "fills_al_book", "tuning_in_sample", "kfold_cv",
        "supervivencia"]
FLAG = ("FAIL", "WARN")   # a detector "flags" if it returns FAIL or WARN

# Base seeds (contiguous blocks; see "seeds" in the JSON).
NULL_SEED_BASE = 2_000_000
TRAP_SEED_BASE = 3_000_000
AUDIT_RNG_BASE = 10_000_000

NULL_TYPES = ["ruido_puro", "momento_causal", "ruido_micro_drift", "ruido_ar1",
              "stop_loss", "pagos_asimetricos", "alpha_rapido_real",
              "edge_neto_positivo"]
INTENSITIES = ["lookahead_completo", "lookahead_1bar", "lookahead_50pct",
               "edge_realo_pequeno"]

WILSON_Z = 1.959963984540054   # z for the 95% CI (two-sided)
WILSON_FORMULA = ("center = (p + z^2/(2n)) / (1 + z^2/n); "
                  "half = z*sqrt(p(1-p)/n + z^2/(4n^2)) / (1 + z^2/n); "
                  "z = 1.959963984540054 (95% two-sided)")

# Parameterization of the mechanical exits of the new nulls.
STOP_MULT = 2.0            # stop-loss at 2 x sigma_path (sigma * sqrt(hold))
TP_MULT_HALF_SPREAD = 3.0  # take-profit at +3 x gross half_spread

# Specificity control (task_3): honest client, small net-positive edge.
# s_eps = SPEC_S_EPS_FRAC * SIGMA (real alpha observable with a 2-bar lead);
# SPEC_HALF_SPREAD calibrated for net edge ~ +30% of the round-trip cost with
# the threshold the tuner picks (see pilot). Declared trades floor.
SPEC_S_EPS_FRAC = 0.5
SPEC_HALF_SPREAD = 0.00045
SPEC_MIN_TRADES = 200


# ──────────────────────────────────────────────────────────────────────
# Signals
# ──────────────────────────────────────────────────────────────────────
def null_signal(kind, px, rng):
    """Signal with no predictive power over the random walk, 4 base levels."""
    if kind == "ruido_puro":
        return [rng.gauss(0.0, 1.0) for _ in px]
    if kind == "momento_causal":
        # momentum of the past 5 bars: honest, and powerless in a random walk
        return ad.causal_signal(px)
    if kind == "ruido_micro_drift":
        # noise + constant bias in the signal (no edge: the biased
        # direction predicts nothing in a random walk, it only pays spread)
        return [rng.gauss(0.0, 1.0) + 0.25 for _ in px]
    if kind == "ruido_ar1":
        # persistent stationary AR(1) noise N(0,1): realistic, powerless
        phi, out, x = 0.5, [], 0.0
        for _ in px:
            x = phi * x + rng.gauss(0.0, math.sqrt(1.0 - phi * phi))
            out.append(x)
        return out
    raise ValueError(kind)


def trap_signal(kind, px, rng):
    """Trap signal per intensity."""
    if kind == "lookahead_completo":
        # peek of the FULL hold: the "signal" at t is the return from
        # t to t+hold, normalized — information that does not exist at
        # decision time
        H, s, n = ad.HOLD, ad.SIGMA, len(px)
        return [(px[i + H] / px[i] - 1.0) / (s * math.sqrt(H))
                if i + H < n else 0.0 for i in range(n)]
    if kind == "lookahead_1bar":
        return ad.look_ahead_signal(px)
    if kind == "lookahead_50pct":
        # 1-bar peek on 50% of the bars; the rest, pure noise.
        # Same N(0,1) marginals: statistically indistinguishable from the
        # leak-free level for any in-sample tuning.
        base = ad.look_ahead_signal(px)
        return [b if rng.random() < 0.5 else rng.gauss(0.0, 1.0) for b in base]
    raise ValueError(kind)


def build_market_alpha(n, p0, s_eps, sigma_z, seed):
    """Random walk + predictable component eps[b] ~ N(0, s_eps^2) in each
    log-return. eps[b] is modeled as observable with a 2-bar lead (the
    alpha known before the bar that pays — e.g. order flow).
    Returns (px, eps)."""
    rng = random.Random(seed)
    px, eps = [p0], [0.0]
    for _ in range(n - 1):
        e = rng.gauss(0.0, s_eps)
        px.append(px[-1] * math.exp(e + rng.gauss(0.0, sigma_z)))
        eps.append(e)
    return px, eps


# ──────────────────────────────────────────────────────────────────────
# Backtests with mechanical exits (stop_loss / pagos_asimetricos nulls)
# ──────────────────────────────────────────────────────────────────────
def backtest_stop(px, sig, thr, stop_level, entry_lag=1, half_spread=ad.HALF_SPREAD,
                  indices=None, hold=ad.HOLD):
    """Mirror of ad.run_backtest with a mechanical stop-loss: the position
    exits on the FIRST bar where the adverse drawdown reaches -stop_level
    (gross); if it never touches it, it holds to the full hold. Spread paid."""
    n = len(sig)
    out = []
    for t in range(n - hold - 1):
        if abs(sig[t]) <= thr:
            continue
        pos = 1.0 if sig[t] > 0 else -1.0
        e_t = t + entry_lag
        x_t = e_t + hold
        if x_t >= n:
            continue
        exit_bar = x_t
        for j in range(1, hold + 1):
            if pos * (px[e_t + j] - px[e_t]) <= -stop_level:
                exit_bar = e_t + j
                break
        pnl = pos * (px[exit_bar] - px[e_t]) - 2 * half_spread
        if indices is None or (indices[0] <= t < indices[1]):
            out.append((t, pos, pnl))
    return out


def backtest_tp(px, sig, thr, tp_level, entry_lag=1, half_spread=ad.HALF_SPREAD,
                indices=None, hold=ad.HOLD):
    """Mirror of ad.run_backtest with a mechanical take-profit: exits on the
    FIRST bar with gross gain >= +tp_level; if it never touches it, it holds
    to the full hold (losses run). Spread paid. Nothing is filtered."""
    n = len(sig)
    out = []
    for t in range(n - hold - 1):
        if abs(sig[t]) <= thr:
            continue
        pos = 1.0 if sig[t] > 0 else -1.0
        e_t = t + entry_lag
        x_t = e_t + hold
        if x_t >= n:
            continue
        exit_bar = x_t
        for j in range(1, hold + 1):
            if pos * (px[e_t + j] - px[e_t]) >= tp_level:
                exit_bar = e_t + j
                break
        pnl = pos * (px[exit_bar] - px[e_t]) - 2 * half_spread
        if indices is None or (indices[0] <= t < indices[1]):
            out.append((t, pos, pnl))
    return out


def tune_threshold_mech(px, sig, bt, indices=None, lo=0.3, hi=2.0, steps=18):
    """Mirror of ad.tune_threshold for a backtest with mechanical exits:
    same grid (lo..hi, 18 steps), same criterion (non-overlapping Sharpe),
    honest tuning over the given window (walk-forward)."""
    best_thr, best_sh = lo, -9e9
    for i in range(steps):
        thr = lo + i * (hi - lo) / (steps - 1)
        sh = ad.metrics(bt(px, sig, thr, indices=indices))["sharpe"]
        if sh > best_sh:
            best_sh, best_thr = sh, thr
    return best_thr


# ──────────────────────────────────────────────────────────────────────
# Client pipelines (honest null / broken trap)
# ──────────────────────────────────────────────────────────────────────
def build_null_client(px, sig, half_spread=ad.HALF_SPREAD):
    """Honest NULL client: walk-forward tuning (first half), first
    executable entry, spread paid, reports the full second half.
    Same pipeline as the demo's 'clean' control."""
    half = len(px) // 2
    thr = ad.tune_threshold(px, sig, indices=(0, half),
                            half_spread=half_spread)
    booked = ad.run_backtest(px, sig, thr, entry_lag=1, price_book=True,
                             half_spread=half_spread,
                             indices=(half, len(px)))
    return booked, sig, thr


def build_trap_client(px, sig):
    """TRAP client (mirror of the demo trap): in-sample tuning over the
    full window, fills at mid, worst decile dropped."""
    thr = ad.tune_threshold(px, sig, entry_lag=0, price_book=False,
                            half_spread=0.0)
    booked = ad.drop_worst_decile(
        ad.run_backtest(px, sig, thr, entry_lag=0, price_book=False))
    return booked, sig, thr


def audit_flags(booked, px, sig, rng, n_perm, stop_loss=False,
                half_spread=ad.HALF_SPREAD, fills_net=False):
    """Runs run_audit and returns (flags per detector 0/1, verdicts F/W/P/S,
    reported metrics). fills_net=True declares fills NET of costs
    (check 2 cannot charge the spread twice)."""
    f, m = ad.run_audit(booked, px, sig, half_spread=half_spread,
                        rng=rng, n_perm=n_perm, stop_loss=stop_loss,
                        fills_net_of_costs=fills_net)
    return ([1 if x["verdict"] in FLAG else 0 for x in f],
            "".join(x["verdict"][0] for x in f), m)


# ──────────────────────────────────────────────────────────────────────
# Jobs and workers (shared by the serial and parallel paths)
# ──────────────────────────────────────────────────────────────────────
def build_jobs(per_type, per_int):
    """Deterministic list of jobs (group, kind, i, seed) in block order:
    nulls first (by type), traps after (by intensity)."""
    jobs = []
    for ti, kind in enumerate(NULL_TYPES):
        for j in range(per_type):
            i = ti * per_type + j
            jobs.append(("null", kind, i, NULL_SEED_BASE + i))
    for it, kind in enumerate(INTENSITIES):
        for j in range(per_int):
            i = it * per_int + j
            jobs.append(("trap", kind, i, TRAP_SEED_BASE + it * 1000 + j))
    return jobs


def run_one(job, n_perm):
    """Runs ONE Monte Carlo audit. Returns a serializable dict with
    group/kind/seed/flags/verdicts/mean/time. Deterministic by seed."""
    group, kind, i, seed = job
    t0 = time.time()
    n = ad.N_DAYS * 60 * 24
    if group == "null" and kind in ("stop_loss", "pagos_asimetricos"):
        # mechanical exits on the null market + signal with no power
        px = ad.build_market(seed=seed)
        rng = random.Random(seed + 500_000)
        sig = [rng.gauss(0.0, 1.0) for _ in px]
        half = len(px) // 2
        if kind == "stop_loss":
            stop_level = STOP_MULT * ad.SIGMA * math.sqrt(ad.HOLD)
            bt = lambda p, s, th, indices: backtest_stop(   # noqa: E731
                p, s, th, stop_level, indices=indices)
            thr = tune_threshold_mech(px, sig, bt, indices=(0, half))
            booked = bt(px, sig, thr, indices=(half, len(px)))
            flags, verdicts, m = audit_flags(booked, px, sig,
                                             random.Random(AUDIT_RNG_BASE + i),
                                             n_perm, stop_loss=True)
        else:
            tp_level = TP_MULT_HALF_SPREAD * ad.HALF_SPREAD
            bt = lambda p, s, th, indices: backtest_tp(     # noqa: E731
                p, s, th, tp_level, indices=indices)
            thr = tune_threshold_mech(px, sig, bt, indices=(0, half))
            booked = bt(px, sig, thr, indices=(half, len(px)))
            flags, verdicts, m = audit_flags(booked, px, sig,
                                             random.Random(AUDIT_RNG_BASE + i),
                                             n_perm, stop_loss=False)
    elif group == "null" and kind == "alpha_rapido_real":
        # small real edge through the HONEST pipeline (same engine as the
        # edge_realo_pequeno intensity, counted on the FP side)
        s_eps = 0.1 * ad.HALF_SPREAD * math.sqrt(math.pi / 2.0)
        sigma_z = math.sqrt(ad.SIGMA ** 2 - s_eps ** 2)
        px, eps = build_market_alpha(n, ad.P0, s_eps, sigma_z, seed)
        sig = [eps[t + 2] / ad.SIGMA if t + 2 < n else 0.0
               for t in range(n)]
        booked, sig, _thr = build_null_client(px, sig)
        flags, verdicts, m = audit_flags(booked, px, sig,
                                         random.Random(AUDIT_RNG_BASE + i),
                                         n_perm)
    elif group == "null" and kind == "edge_neto_positivo":
        # Specificity control: honest client with a small real edge
        # NET-POSITIVE after costs and >= 200 trades. Fills declared net
        # (--fills-net-of-costs): the audit does not charge the spread twice.
        s_eps = SPEC_S_EPS_FRAC * ad.SIGMA
        sigma_z = math.sqrt(ad.SIGMA ** 2 - s_eps ** 2)
        px, eps = build_market_alpha(n, ad.P0, s_eps, sigma_z, seed)
        sig = [eps[t + 2] / s_eps if t + 2 < n else 0.0
               for t in range(n)]
        booked, sig, _thr = build_null_client(px, sig,
                                              half_spread=SPEC_HALF_SPREAD)
        flags, verdicts, m = audit_flags(booked, px, sig,
                                         random.Random(AUDIT_RNG_BASE + i),
                                         n_perm,
                                         half_spread=SPEC_HALF_SPREAD,
                                         fills_net=True)
    elif group == "null":
        px = ad.build_market(seed=seed)
        sig = null_signal(kind, px, random.Random(seed + 500_000))
        booked, sig, _thr = build_null_client(px, sig)
        flags, verdicts, m = audit_flags(booked, px, sig,
                                         random.Random(AUDIT_RNG_BASE + i),
                                         n_perm)
    else:
        if kind == "edge_realo_pequeno":
            s_eps = 0.1 * ad.HALF_SPREAD * math.sqrt(math.pi / 2.0)
            sigma_z = math.sqrt(ad.SIGMA ** 2 - s_eps ** 2)
            px, eps = build_market_alpha(n, ad.P0, s_eps, sigma_z, seed)
            sig = [eps[t + 2] / ad.SIGMA if t + 2 < n else 0.0
                   for t in range(n)]
            booked, sig, _thr = build_null_client(px, sig)
        else:
            px = ad.build_market(seed=seed)
            sig = trap_signal(kind, px, random.Random(seed + 500_000))
            booked, sig, _thr = build_trap_client(px, sig)
        flags, verdicts, m = audit_flags(booked, px, sig,
                                         random.Random(AUDIT_RNG_BASE + 100_000 + i),
                                         n_perm)
    return {"group": group, "kind": kind, "i": i, "seed": seed,
            "flags": flags, "verdicts": verdicts, "mean": m["mean"],
            "n": m["n"], "t": round(time.time() - t0, 2)}


def _pool_job(args):
    group, kind, i, seed, n_perm = args
    return run_one((group, kind, i, seed), n_perm)


# ──────────────────────────────────────────────────────────────────────
# FP and power parts (serial, legacy path)
# ──────────────────────────────────────────────────────────────────────
def run_fp(n_perm, per_type):
    """per_type x len(NULL_TYPES) honest null strategies through the audit
    pipeline. Returns a list of (type, seed, flags, verdicts, mean)."""
    out = []
    jobs = [j for j in build_jobs(per_type, 0)]
    for k, (group, kind, i, seed) in enumerate(jobs):
        t0 = time.time()
        r = run_one((group, kind, i, seed), n_perm)
        out.append((r["kind"], r["seed"], r["flags"], r["verdicts"], r["mean"]))
        if k % 20 == 0:
            print("  [FP %3d/%d] %s seed=%d flags=%s %.1fs"
                  % (k + 1, len(jobs), kind, seed, r["verdicts"],
                     time.time() - t0), flush=True)
    return out


def run_power(n_perm, per_int):
    """4 intensities x per_int trap strategies. Returns a list of
    (intensity, seed, flags, verdicts, mean)."""
    out = []
    jobs = build_jobs(0, per_int)
    for k, (group, kind, i, seed) in enumerate(jobs):
        t0 = time.time()
        r = run_one((group, kind, i, seed), n_perm)
        out.append((r["kind"], r["seed"], r["flags"], r["verdicts"], r["mean"]))
        if k % 25 == 0:
            print("  [POW %3d/%d] %s seed=%d flags=%s %.1fs"
                  % (k + 1, len(jobs), kind, seed, r["verdicts"],
                     time.time() - t0), flush=True)
    return out


# ──────────────────────────────────────────────────────────────────────
# Parallel + checkpoints
# ──────────────────────────────────────────────────────────────────────
def run_jobs_parallel(jobs, n_perm, jobs_n, ckpt_path, resume):
    """Runs jobs in a Pool of jobs_n processes; each result is appended to
    the checkpoint JSONL (flush + fsync) so it can be resumed after a cut.
    Returns the full list of result dicts (resumed + new)."""
    done = {}
    if resume and os.path.exists(ckpt_path):
        with open(ckpt_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                done[(r["group"], r["kind"], r["i"])] = r
        print("  resume: %d runs already in checkpoint" % len(done), flush=True)
    todo = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    print("  jobs: %d to run, %d already done" % (len(todo), len(done)),
          flush=True)
    results = list(done.values())
    if todo:
        t0 = time.time()
        todo5 = [(g, k, i, s, n_perm) for (g, k, i, s) in todo]
        with mp.Pool(processes=jobs_n) as pool:
            for k, r in enumerate(pool.imap_unordered(_pool_job, todo5), 1):
                results.append(r)
                with open(ckpt_path, "a") as fh:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                    fh.flush()
                    os.fsync(fh.fileno())
                if k % 50 == 0 or k == len(todo):
                    el = time.time() - t0
                    print("  [%4d/%4d] %.1f runs/s, ETA %.0f min"
                          % (k, len(todo), k / el,
                             (len(todo) - k) * el / k / 60.0), flush=True)
    return results


# ──────────────────────────────────────────────────────────────────────
# Aggregation and reporting
# ──────────────────────────────────────────────────────────────────────
def wilson(k, n, z=WILSON_Z):
    """Wilson 95% interval for a proportion k/n. Returns
    (lo, hi, p). With k=0: the upper bound caps FP < ~3/n."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half), p)


def _block(rows):
    """rows: list of dicts with 'flags' and 'verdicts'. Returns (n, flags per
    detector, union, verdicts distribution)."""
    n = len(rows)
    per = [sum(1 for r in rows if r["flags"][d]) for d in range(5)]
    union = sum(1 for r in rows if any(r["flags"]))
    dist = {}
    for r in rows:
        dist[r["verdicts"]] = dist.get(r["verdicts"], 0) + 1
    return n, per, union, dist


def _det_entry(rows, d):
    n, per, _union, _dist = _block(rows)
    lo, hi, p = wilson(per[d], n)
    return {"n": n, "marcadas": per[d], "rate": p,
            "ci95": [round(lo, 4), round(hi, 4)]}


def summarize(rows, groups):
    """Aggregates per detector with a per-group (type/intensity) breakdown
    and a Wilson 95% CI on each rate."""
    n, per, union, dist = _block(rows)
    out = {}
    for d, name in enumerate(DETS):
        out[name] = _det_entry(rows, d)
    lo, hi, p = wilson(union, n)
    out["_union"] = {"n": n, "marcadas": union, "rate": p,
                     "ci95": [round(lo, 4), round(hi, 4)]}
    out["_verdicts_dist"] = dist
    out["_por_grupo"] = {}
    for g in groups:
        sub = [r for r in rows if r["kind"] == g]
        gn, gper, gunion, _ = _block(sub)
        out["_por_grupo"][g] = {
            name: _det_entry(sub, d) for d, name in enumerate(DETS)}
        lo, hi, p = wilson(gunion, gn)
        out["_por_grupo"][g]["_union"] = {"n": gn, "marcadas": gunion,
                                          "rate": p,
                                          "ci95": [round(lo, 4), round(hi, 4)]}
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(
        description="Monte Carlo FP/power of the audit detectors.")
    ap.add_argument("--n-perm", type=int, default=ad.N_PERM,
                    help="permutations per test (default %d)" % ad.N_PERM)
    ap.add_argument("--per-type", type=int, default=50,
                    help="null strategies per signal type (default 50)")
    ap.add_argument("--per-intensity", type=int, default=100,
                    help="trap strategies per intensity (default 100)")
    ap.add_argument("--no-write", dest="write", action="store_false",
                    default=True, help="do not write the JSON")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel processes (default 1 = serial legacy)")
    ap.add_argument("--checkpoint", default=None,
                    help="JSONL with finished runs (default: "
                         "results/mc_fp_power_checkpoint.jsonl)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the checkpoint (--jobs > 1)")
    args = ap.parse_args(argv)

    ckpt = args.checkpoint or os.path.join(RESULTS_DIR,
                                           "mc_fp_power_checkpoint.jsonl")
    t_start = time.time()
    jobs = build_jobs(args.per_type, args.per_intensity)
    n_nulls = args.per_type * len(NULL_TYPES)
    n_traps = args.per_intensity * len(INTENSITIES)
    print("FP: %d null strategies (%d x %d types) + POT: %d traps "
          "(%d x %d) = %d audits, jobs=%d"
          % (n_nulls, args.per_type, len(NULL_TYPES), n_traps,
             args.per_intensity, len(INTENSITIES), len(jobs), args.jobs),
          flush=True)

    if args.jobs > 1:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        rows = run_jobs_parallel(jobs, args.n_perm, args.jobs, ckpt,
                                 args.resume)
    else:
        rows = []
        for k, (group, kind, i, seed) in enumerate(jobs):
            t0 = time.time()
            r = run_one((group, kind, i, seed), args.n_perm)
            rows.append(r)
            if r["group"] == "null" and k % 20 == 0:
                print("  [FP %4d/%d] %s seed=%d flags=%s %.1fs"
                      % (k + 1, n_nulls, kind, seed, r["verdicts"],
                         time.time() - t0), flush=True)
            elif r["group"] == "trap" and (k - n_nulls) % 25 == 0:
                print("  [POW %3d/%d] %s seed=%d flags=%s %.1fs"
                      % (k - n_nulls + 1, n_traps, kind, seed, r["verdicts"],
                         time.time() - t0), flush=True)
    elapsed = time.time() - t_start

    fp_rows = [r for r in rows if r["group"] == "null"]
    pot_rows = [r for r in rows if r["group"] == "trap"]
    assert len(fp_rows) == n_nulls and len(pot_rows) == n_traps, \
        "missing runs: %d nulls + %d traps of %d + %d expected" % (
            len(fp_rows), len(pot_rows), n_nulls, n_traps)
    fp = summarize(fp_rows, NULL_TYPES)
    pot = {k: summarize([r for r in pot_rows if r["kind"] == k], [k])
           for k in INTENSITIES}

    # ── specificity: profitable honest client (net edge+, >=200 trades) ──
    spec_rows = [r for r in fp_rows if r["kind"] == "edge_neto_positivo"]
    spec = None
    if spec_rows:
        ns = [r.get("n", 0) or 0 for r in spec_rows]
        flagged = sum(1 for r in spec_rows if any(r["flags"]))
        lo, hi, p = wilson(flagged, len(spec_rows))
        srt = sorted(ns)
        spec = {
            "n_runs": len(spec_rows),
            "min_trades": min(ns),
            "median_trades": srt[len(srt) // 2],
            "runs_ge_min": sum(1 for x in ns if x >= SPEC_MIN_TRADES),
            "min_trades_declarado": SPEC_MIN_TRADES,
            "mean_net_per_trade": round(
                sum(r["mean"] for r in spec_rows) / len(spec_rows), 6),
            "flagged": flagged,
            "flag_rate": p,
            "ci95": [round(lo, 4), round(hi, 4)],
            "s_eps_frac_de_sigma": SPEC_S_EPS_FRAC,
            "half_spread": SPEC_HALF_SPREAD,
            "fills": "net of costs, declared (--fills-net-of-costs)",
        }

    # ── summary tables ──────────────────────────────────────────────
    L = []
    L.append("=" * 78)
    L.append("MONTE CARLO OF DETECTORS — FP and power  (%.0fs of CPU)"
             % elapsed)
    L.append("alpha=%.2f, %d permutations/test, market %d days without edge, "
             "spread 2x%d bps" % (ad.ALPHA, args.n_perm, ad.N_DAYS,
                                  ad.HALF_SPREAD * 1e4))
    L.append("Wilson 95%% CI: %s" % WILSON_FORMULA)
    L.append("=" * 78)
    L.append("")
    L.append("FALSE POSITIVES — %d honest null strategies (no edge, "
             "clean plumbing), %d x %d types" % (len(fp_rows), args.per_type,
                                                 len(NULL_TYPES)))
    L.append("  %-18s %10s %7s %-22s per signal type (n=%d each)"
             % ("detector", "flag.", "FP % [CI95]", "", args.per_type))
    for d, name in enumerate(DETS):
        por_tipo = "  ".join(
            "%s=%.0f%%" % (g[:6], 100 * fp["_por_grupo"][g][name]["rate"])
            for g in NULL_TYPES)
        L.append("  %-18s %6d %5.1f%% [%4.1f,%4.1f]  %s"
                 % (name, fp[name]["marcadas"], 100 * fp[name]["rate"],
                    100 * fp[name]["ci95"][0], 100 * fp[name]["ci95"][1],
                    por_tipo))
    L.append("  %-18s %6d %5.1f%% [%4.1f,%4.1f]  <- flagged by at least one "
             "detector" % ("UNION", fp["_union"]["marcadas"],
                           100 * fp["_union"]["rate"],
                           100 * fp["_union"]["ci95"][0],
                           100 * fp["_union"]["ci95"][1]))
    if spec:
        L.append("")
        L.append("SPECIFICITY — honest client with NET-POSITIVE edge and "
                 "enough trades (task_3 control)")
        L.append("  edge_neto_positivo: %d/%d flagged (%.1f%% [CI95 %.1f%%,"
                 "%.1f%%]); trades: min %d, median %d; mean net %+.5f/trade"
                 % (spec["flagged"], spec["n_runs"], 100 * spec["flag_rate"],
                    100 * spec["ci95"][0], 100 * spec["ci95"][1],
                    spec["min_trades"], spec["median_trades"],
                    spec["mean_net_per_trade"]))
    L.append("")
    L.append("POWER — trap strategies by intensity (n=%d each)"
             % args.per_intensity)
    L.append("  %-20s " % "intensity" +
             " ".join("%13s" % d for d in DETS) + " %10s" % "UNION")
    for kind in INTENSITIES:
        L.append("  %-20s " % kind +
                 " ".join("%5d/%-4d" % (pot[kind][d]["marcadas"],
                                        pot[kind][d]["n"]) for d in DETS) +
                 " %5d/%-4d" % (pot[kind]["_union"]["marcadas"],
                                pot[kind]["_union"]["n"]))
    L.append("")

    # ── verdict (built from the data, not hand-written) ──
    fp_max = max(fp[d]["rate"] for d in DETS)
    fp_ok = all(fp[d]["rate"] <= 0.10 for d in DETS) and \
        fp["_union"]["rate"] <= 0.10
    pot_obvios = {k: pot[k]["_union"]["rate"] for k in INTENSITIES[:3]}
    pot_ok = all(v >= 0.80 for v in pot_obvios.values())
    pot_edge = pot["edge_realo_pequeno"]["_union"]["rate"]
    c1_full = pot["lookahead_completo"]["capturabilidad"]["rate"]
    c2_full = pot["lookahead_completo"]["fills_al_book"]["rate"]
    notas = []
    if c1_full < 0.5:
        notas.append("c1(capturabilidad) does not catch full-hold look-ahead "
                     "(the 1-bar delay only removes 1/30 of the peek; "
                     "it detects the 1-bar peek: %d%%)"
                     % int(100 * pot["lookahead_1bar"]["capturabilidad"]["rate"]))
    if c2_full < 0.5:
        notas.append("c2(fills al book) does not fire when the edge exceeds "
                     "the round-trip cost (full edge > 2x spread)")
    if pot_edge <= 0.10:
        notas.append("small real edge not detected (%.0f%%) — correct: "
                     "it is not a trap and the detectors hunt plumbing "
                     "biases, not the existence of edge" % (100 * pot_edge))
    veredicto = ("AUDIT OK: FP <= 10%% per detector (max %.1f%%) and power >= "
                 "80%% on obvious traps (min %.0f%%); union FP %.1f%%, "
                 "union trap power %.0f%%"
                 % (100 * fp_max, 100 * min(pot_obvios.values()),
                    100 * fp["_union"]["rate"], 100 * min(pot_obvios.values()))
                 if (fp_ok and pot_ok)
                 else "CHECK: %s" % (
                     "high FP on %s" % [d for d in DETS if fp[d]["rate"] > 0.10]
                     if not fp_ok else
                     "low power on %s" % [k for k, v in pot_obvios.items()
                                          if v < 0.80]))
    if notas:
        veredicto += "; honest limitations: " + " | ".join(notas)
    if spec:
        veredicto += ("; specificity net-edge+ (%d trades+): %.1f%% "
                      "flagged [CI95 %.1f%%,%.1f%%]"
                      % (spec["min_trades_declarado"], 100 * spec["flag_rate"],
                         100 * spec["ci95"][0], 100 * spec["ci95"][1]))

    # ── JSON ────────────────────────────────────────────────────────
    result = {
        "params": {
            "alpha": ad.ALPHA, "n_perm": args.n_perm, "n_days": ad.N_DAYS,
            "bars": ad.N_DAYS * 60 * 24, "hold": ad.HOLD,
            "half_spread": ad.HALF_SPREAD, "sigma": ad.SIGMA,
            "null_types": NULL_TYPES, "intensities": INTENSITIES,
            "n_null_per_type": args.per_type,
            "n_trap_per_intensity": args.per_intensity,
            "flag_definition": "verdict FAIL or WARN",
            "null_client": "tuning walk-forward, entry_lag=1, spread paid, "
                           "reports the full second half (honest control)",
            "spec_s_eps_frac_de_sigma": SPEC_S_EPS_FRAC,
            "spec_half_spread": SPEC_HALF_SPREAD,
            "spec_min_trades": SPEC_MIN_TRADES,
            "spec_fills": "declared net of costs (the spread is not charged "
                          "twice)",
            "trap_client": "tuning in-sample, entry_lag=0, fills at mid, "
                           "drop_worst_decile (mirror of the demo trap)",
            "stop_loss_null": "mechanical exits: stop at %.1f x sigma_path "
                              "(sigma*sqrt(hold) = %.5f gross); DECLARED "
                              "(stop_loss=True → check 5 SKIP by design)"
                              % (STOP_MULT, STOP_MULT * ad.SIGMA
                                 * math.sqrt(ad.HOLD)),
            "pagos_asimetricos_null": "mechanical take-profit at +%.1f x "
                                      "gross half_spread (+%.4f), losses "
                                      "run to the hold; NOT declared "
                                      "(truncated tail = RIGHT, nothing "
                                      "filtered: tests whether check 5 "
                                      "confuses mechanical payoff with "
                                      "survivorship)"
                                      % (TP_MULT_HALF_SPREAD,
                                         TP_MULT_HALF_SPREAD * ad.HALF_SPREAD),
            "alpha_rapido_real_null": "genuine edge = 0.1 x half_spread/trade "
                                      "gross (E|eps| = sqrt(2/pi) s_eps), "
                                      "honest plumbing; same engine as the "
                                      "edge_realo_pequeno intensity, "
                                      "counted on the FP side",
            "edge_realo_pequeno": "genuine edge = 0.1 x half_spread/trade "
                                  "gross (E|eps| = sqrt(2/pi) s_eps), "
                                  "honest plumbing; NOT a trap",
            "wilson_ci": WILSON_FORMULA,
            "elapsed_s": round(elapsed, 1),
            "jobs": args.jobs,
            "cpu": "CPU only, stdlib",
        },
        "seeds": {
            "null_market": {"start": NULL_SEED_BASE,
                            "count": len(fp_rows),
                            "note": "seed i-th = start + i, per type "
                                    "(blocks of n_null_per_type)"},
            "trap_market": {"start": TRAP_SEED_BASE,
                            "blocks": {k: {"start": TRAP_SEED_BASE + t * 1000,
                                           "count": args.per_intensity}
                                       for t, k in enumerate(INTENSITIES)},
                            "note": "each intensity has a block of 1000"},
            "audit_rng": AUDIT_RNG_BASE,
            "signal_rng": "market_seed + 500000",
        },
        "fp_rate_por_detector": {
            name: fp[name] for name in DETS
        },
        "fp_rate_por_tipo": {
            g: {name: fp["_por_grupo"][g][name] for name in DETS}
            | {"union": fp["_por_grupo"][g]["_union"]}
            for g in NULL_TYPES
        },
        "fp_union": fp["_union"],
        "especificidad": spec,
        "fp_verdicts_distribution": fp["_verdicts_dist"],
        "potencia_por_intensidad": {
            k: {name: pot[k][name] for name in DETS}
            | {"union": pot[k]["_union"],
               "verdicts_distribution": pot[k]["_verdicts_dist"]}
            for k in INTENSITIES
        },
        "veredicto": veredicto,
    }

    for line in L:
        print(line)
    print("FP_RATE_BY_DETECTOR: " + "; ".join(
        "%s=%.1f%% (%d/%d) CI95=[%.1f%%, %.1f%%]"
        % (d, 100 * fp[d]["rate"], fp[d]["marcadas"], fp[d]["n"],
           100 * fp[d]["ci95"][0], 100 * fp[d]["ci95"][1]) for d in DETS)
        + "; union=%.1f%% (%d/%d) CI95=[%.1f%%, %.1f%%]"
        % (100 * fp["_union"]["rate"], fp["_union"]["marcadas"],
           fp["_union"]["n"], 100 * fp["_union"]["ci95"][0],
           100 * fp["_union"]["ci95"][1]))
    print("POWER_BY_INTENSITY: " + "; ".join(
        "%s: union=%.1f%% [%s]" % (
            k, 100 * pot[k]["_union"]["rate"],
            "+".join("%s=%d%%" % (d[:4], 100 * pot[k][d]["rate"])
                     for d in DETS))
        for k in INTENSITIES))
    print("VERDICT: " + veredicto)

    if args.write:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = os.path.join(RESULTS_DIR, "monte_carlo_fp_power.json")
        with open(out, "w") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        print("written:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
