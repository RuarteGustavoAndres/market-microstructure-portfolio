#!/usr/bin/env python3
"""Audit of THIRD-PARTY STRATEGIES through the audit pipeline (4h).

Does not measure the detectors (that is `monte_carlo_fp_power.py`):
demonstrates the SERVICE — a third-party client submits 5 distinct
strategies on the demo market (1-min random walk, no edge by
construction) and the audit separates them: the honest ones come out
with benign verdicts (no false alarm, not even on pure noise) and the
cheating ones come out WARN/FAIL. Then the full batch is repeated at
p0=0.5 and p0=100: the scale invariance the reviewer asked for (the
audit tests are ratios of statistics, so they must give the same
verdict at any scale).

The 5 strategies (each with a documented setup and fixed seed):

  1. momentum_chaser (hONEST, naive) — enters after k=3 consecutive
     ups (downs): signal = z-scored 3-bar momentum with sign ±1 only if
     the last 3 bars had the same sign, 0 otherwise. Naive by design
     (chasing momentum in a random walk predicts nothing), but the
     plumbing is honest end to end.
  2. spread_sniper (hONEST, mean reversion) — enters when price touches
     the extreme: z = (px - mean_120)/sd_120, signal = -z (fade of the
     extreme; long when it touches the minimum, short when it touches
     the maximum). The |signal|>thr threshold is chosen walk-forward on
     the 0.3-2.0 grid.
  3. ruido_puro (hONEST, null control) — random entry: signal iid
     N(0,1). Must give benign verdicts with P&L ~ -2 x half_spread
     (pays the spread and has no edge: exactly what an honest backtest
     of a no-edge market produces). It is the "no false alarm" case.
  4. lookahead_parcial (TRAP) — uses information 1 bar ahead in 50% of
     the trades (the other 50%, pure noise N(0,1): same marginals,
     indistinguishable for any in-sample tuning).
  5. lookahead_completo (TRAP, the classic demo trap) — the "indicator"
     at t is the full hold return (30 bars), normalized: information
     that does not exist at decision time.

Plumbing by type (documented, same as the demo and the Monte Carlo):
  honest — walk-forward tuning (0.3-2.0 grid on the first half),
           entry_lag=1 (first executable entry), price_book=True with
           half_spread scaled with the market scale (constant bps:
           half_spread = HALF_SPREAD * p0 / P0, round trip = 2 x
           half_spread), reports the COMPLETE second half.
  cheating — exact mirror of the demo TRAP client: in-sample tuning on
             the full window, entry_lag=0 (enters on the decision bar),
             fills at mid (price_book=False), and the worst decile of
             the trade list dropped (drop_worst_decile).

The audit runs the 5 detectors with run_audit (prices + signal
supplied, half_spread scaled with the market scale — constant bps —,
alpha=0.01, 999 permutations).
Global verdict per strategy: FAIL if any detector fails, WARN if the
worst is WARN, OK (benign) if all PASS.

Honesty: if an honest one comes out flagged (false alarm) or a trap
passes clean (undetected trap), it is reported AS-IS in the verdict.

Deterministic: fixed seeds (market 7; signals 201-205; audit rng
40_000_000+i, IDENTICAL at both scales so the invariance comparison is
exact — no permutation noise), all reported in the JSON. Stdlib only.

CLI:
    python3 demo/audit_controles_sinteticos.py [--n-perm 999] [--no-write]

Writes results/audit_controles_sinteticos.json and a summary table to
stdout. At the end it prints VERDICTS_BY_STRATEGY and VERDICT (one line
each).
"""
import argparse
import json
import math
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

SEED_MERCADO = 7          # same market for the 5 strategies and both scales
P0S = [0.50, 100.0]       # scale invariance (reviewer request)
SIG_SEED_BASE = 201       # signal of the i-th strategy = base + i
AUDIT_RNG_BASE = 40_000_000
K_MOMENTUM = 3            # momentum_chaser: k consecutive ups/downs
WIN_SNIPER = 120          # spread_sniper: rolling window of the extreme


# ──────────────────────────────────────────────────────────────────────
# Signals of the 5 strategies (each documented above)
# ──────────────────────────────────────────────────────────────────────
def sig_momentum_chaser(px, k=K_MOMENTUM):
    """±z-scored k-bar momentum, only if the last k bars had the same
    sign (3 ups -> long, 3 downs -> short), 0 otherwise. Naive: in a
    random walk it predicts nothing."""
    n = len(px)
    out = [0.0] * n
    s3 = ad.SIGMA * math.sqrt(k)
    for t in range(k, n):
        r1, r2, r3 = (px[t - 2] / px[t - 3], px[t - 1] / px[t - 2],
                      px[t] / px[t - 1])
        if r1 > 1 and r2 > 1 and r3 > 1:
            out[t] = (px[t] / px[t - k] - 1.0) / s3
        elif r1 < 1 and r2 < 1 and r3 < 1:
            out[t] = -(px[t] / px[t - k] - 1.0) / s3
    return out


def sig_spread_sniper(px, window=WIN_SNIPER):
    """Mean reversion: z = (px - mean_window)/sd_window over the last
    `window` bars; signal = -z (fade: price touches the upper extreme ->
    short, lower -> long). In a random walk there is no reversion."""
    n = len(px)
    out = [0.0] * n
    s = sq = 0.0
    for t in range(n):
        s += px[t]
        sq += px[t] * px[t]
        if t >= window:
            old = px[t - window]
            s -= old
            sq -= old * old
        if t >= window:
            m = s / window
            v = max(sq / window - m * m, 1e-12)
            out[t] = -(px[t] - m) / math.sqrt(v)
    return out


def sig_ruido_puro(px, seed):
    """Random entry: signal iid N(0,1) with no relation to the market."""
    rng = random.Random(seed)
    return [rng.gauss(0.0, 1.0) for _ in px]


def sig_lookahead_parcial(px, seed):
    """TRAP: in 50% of the bars the "indicator" is the 1-bar-ahead
    return (1-bar peek, normalized by sigma); in the other 50%, pure
    noise N(0,1). Same N(0,1) marginals: in-sample tuning cannot tell
    the two sets apart."""
    rng = random.Random(seed)
    base = ad.look_ahead_signal(px)
    return [b if rng.random() < 0.5 else rng.gauss(0.0, 1.0) for b in base]


def sig_lookahead_completo(px):
    """Classic TRAP (the demo one): the "indicator" at t is the full
    hold return (30 bars), normalized — information that does not exist
    when the trade is decided."""
    H, s, n = ad.HOLD, ad.SIGMA, len(px)
    return [(px[i + H] / px[i] - 1.0) / (s * math.sqrt(H))
            if i + H < n else 0.0 for i in range(n)]


# (function, type, documented setup) — the order defines the seed index.
STRATEGIES = [
    ("momentum_chaser", "honest", sig_momentum_chaser,
     "enters after k=3 consecutive ups (downs): signal = ±z-momentum of 3 "
     "bars only if the last 3 have the same sign (naive)"),
    ("spread_sniper", "honest", sig_spread_sniper,
     "mean reversion: enters when price touches the rolling 120-bar "
     "extreme (|z|>thr), direction = fade of the extreme"),
    ("ruido_puro", "honest", sig_ruido_puro,
     "random entry: signal iid N(0,1) — null control, expected P&L "
     "~ -2 x half_spread"),
    ("lookahead_parcial", "trap", sig_lookahead_parcial,
     "TRAP: uses 1-bar-ahead info in 50% of the trades; the other "
     "50% is N(0,1) noise (same marginals)"),
    ("lookahead_completo", "trap", sig_lookahead_completo,
     "classic TRAP: the signal at t is the FULL hold return "
     "(30 bars) — the demo TRAP client"),
]
HONEST = {n for n, _t, _f, _d in STRATEGIES if _t == "honest"}


# ──────────────────────────────────────────────────────────────────────
# Client pipelines (same as demo/audit_demo.py and the Monte Carlo)
# ──────────────────────────────────────────────────────────────────────
def build_honest_client(px, sig, hs):
    """HONEST: walk-forward tuning (first half), first executable entry,
    spread paid, reports the complete second half.
    hs = half_spread in price units of THIS scale (constant bps across
    scales: hs = HALF_SPREAD * p0 / P0)."""
    half = len(px) // 2
    thr = ad.tune_threshold(px, sig, indices=(0, half), entry_lag=1,
                            price_book=True, half_spread=hs)
    booked = ad.run_backtest(px, sig, thr, entry_lag=1, price_book=True,
                             half_spread=hs,
                             indices=(half, len(px)))
    return booked, thr


def build_trap_client(px, sig):
    """TRAP (mirror of the demo TRAP client): in-sample tuning on the
    full window, fills at mid, entry on the decision bar, worst decile
    dropped."""
    thr = ad.tune_threshold(px, sig, entry_lag=0, price_book=False,
                            half_spread=0.0)
    booked = ad.drop_worst_decile(
        ad.run_backtest(px, sig, thr, entry_lag=0, price_book=False))
    return booked, thr


# ──────────────────────────────────────────────────────────────────────
# Batch: 5 strategies x 2 scales
# ──────────────────────────────────────────────────────────────────────
def run_one(name, tipo, sig_fn, i, p0, n_perm):
    """Builds the strategy on the market (fixed seed, same for both
    scales: the geometric walk scales exactly by p0), runs it through
    its client pipeline and audits it."""
    px = ad.build_market(seed=SEED_MERCADO, p0=p0)
    # The book spread is a fraction of price (constant bps), not an
    # absolute amount: at p0=0.5 the round trip is 80 bps, at p0=100
    # too. Without this, a fixed price-unit spread breaks scale
    # invariance in the checks that charge it over the pnl
    # (tuning_in_sample, fills_al_book).
    hs = ad.HALF_SPREAD * p0 / ad.P0
    seed_sig = SIG_SEED_BASE + i
    # only ruido_puro and lookahead_parcial consume their own signal seed
    sig = sig_fn(px, seed_sig) if sig_fn in (sig_ruido_puro,
                                             sig_lookahead_parcial) \
        else sig_fn(px)
    if tipo == "honest":
        booked, thr = build_honest_client(px, sig, hs)
    else:
        booked, thr = build_trap_client(px, sig)
    # identical audit rng per strategy at both scales: with the scaled
    # spread, the p0=100 pnls are exactly 200x the p0=0.5 ones (same
    # scaled walk, same signals/trades), so the rank-based detectors and
    # the permutations reproduce the SAME verdict — the invariance is
    # demonstrated exactly, not on average.
    rng = random.Random(AUDIT_RNG_BASE + i)
    findings, m = ad.run_audit(booked, px, sig, half_spread=hs,
                               rng=rng, n_perm=n_perm)
    verdicts = {d: f["verdict"] for d, f in zip(DETS, findings)}
    flagged = any(v in FLAG for v in verdicts.values())
    worst = "FAIL" if "FAIL" in verdicts.values() else (
        "WARN" if "WARN" in verdicts.values() else "PASS")
    return {
        "thr": round(thr, 4),
        "n_trades": m["n"],
        "mean_pnl": m["mean"],
        "total_pnl": sum(p for _t, _p, p in booked),
        "sharpe": m["sharpe"],
        "win": m["win"],
        "flags": verdicts,
        "verdicts": "".join(v[0] for v in verdicts.values()),
        "global": worst,
        "flagged": flagged,
    }


def run_lote(n_perm):
    """Returns {name: {setup, tipo, p0: {result}}} for the whole batch."""
    out = {}
    for i, (name, tipo, sig_fn, setup) in enumerate(STRATEGIES):
        rec = {"tipo": tipo, "setup": setup,
               "seed_senal": SIG_SEED_BASE + i, "p0": {}}
        for p0 in P0S:
            t0 = time.time()
            rec["p0"][p0] = run_one(name, tipo, sig_fn, i, p0, n_perm)
            print("  [%s p0=%-6g] %s n=%d mean=%+.5f flags=%s %.1fs"
                  % (name, p0, rec["p0"][p0]["verdicts"],
                     rec["p0"][p0]["n_trades"], rec["p0"][p0]["mean_pnl"],
                     rec["p0"][p0]["verdicts"], time.time() - t0),
                  flush=True)
        out[name] = rec
    return out


# ──────────────────────────────────────────────────────────────────────
# Verdicts (built from the data, not written by hand)
# ──────────────────────────────────────────────────────────────────────
def expected_ok(rec):
    """The strategy comes out as expected: honest -> unflagged; trap -> flagged — at BOTH scales."""
    p0s = [rec["p0"][p0] for p0 in P0S]
    if rec["tipo"] == "honest":
        return all(not r["flagged"] for r in p0s)
    return all(r["flagged"] for r in p0s)


def escala_identica(rec):
    """Per-detector verdict identical at p0=0.5 and p0=100; if it
    differs, returns the list of checks that change."""
    a, b = rec["p0"][P0S[0]]["flags"], rec["p0"][P0S[1]]["flags"]
    return [d for d in DETS if a[d] != b[d]]


def build_veredicto(lote):
    """The audit is OK if: every honest strategy comes out benign at the 2 scales (no
    false alarm outside the documented known limitations) and
    every trap comes out flagged at the 2 scales. Deviations are
    reported as-is."""

    # Known audit limitations: the only historical one
    # (spread_sniper x survivorship) was RESOLVED by the forensic audit
    # of 2026-10-24 — the root cause was the sample size (the level
    # selective sniper enters in BURSTS: overlapping entries treated
    # as independent draws), not the fixed cost (the test is
    # rank-based, invariant to translation: net vs gross gives (nl, nr, p)
    # identical 48/48). Check 5 now runs over the NON
    # overlapping trades. The sniper must come out benign; if it fires again
    # it is a real false alarm, with no whitelist to justify it:
    LIMITACIONES_CONOCIDAS = {}

    problemas = []
    documentadas = []
    falsas = [n for n, r in lote.items() if r["tipo"] == "honest"
              and expected_ok(r) is False]
    trampas_libres = [n for n, r in lote.items() if r["tipo"] == "trap"
                      and expected_ok(r) is False]
    for n in falsas:
        r = lote[n]
        for p0 in P0S:
            marc = [d for d in DETS if r["p0"][p0]["flags"][d] in FLAG]
            extras = [d for d in marc
                      if d not in LIMITACIONES_CONOCIDAS.get(n, set())]
            if extras:
                problemas.append("false alarm: %s flagged by %s at p0=%g"
                                 % (n, extras or "?", p0))
            elif marc:
                documentadas.append("%s flagged by %s at p0=%g "
                                    "(known limitation of the detector, "
                                    "see limitaciones_conocidas)"
                                    % (n, marc, p0))
    for n in trampas_libres:
        r = lote[n]
        for p0 in P0S:
            if not r["p0"][p0]["flagged"]:
                problemas.append("TRAP NOT DETECTED: %s passes clean at "
                                 "p0=%g" % (n, p0))

    # Scale invariance: with the spread scaled with the price (constant
    # bps) and the identical audit rng, the full pipeline is
    # scale-invariant and the SAME per-detector verdict is expected at
    # p0=0.5 and p0=100; any difference is invariance BROKEN.
    diffs = {n: escala_identica(lote[n]) for n in lote}
    diffs = {n: d for n, d in diffs.items() if d}

    if not problemas:
        v = ("AUDIT OK: the audit separates the honest ones from the %d "
             "traps (WARN/FAIL at the 2 scales); the 3 honest ones "
             "give a benign verdict at the 2 scales, no false "
             "alarms — including spread_sniper since the fix of "
             "check 5 (tail test over NON-overlapping trades; "
             "forensic audit 2026-10-24)"
             % (len(STRATEGIES) - len(HONEST)))
        if documentadas:
            v += (" — with %d documented known limitation(s) of the "
                  "detector: %s" % (len(documentadas), "; ".join(documentadas)))
    else:
        v = "REVISAR: " + "; ".join(problemas)

    if diffs:
        det = "; ".join("%s changes in %s" % (n, d) for n, d in sorted(diffs.items()))
        v += "; scale invariance BROKEN: %s" % det
    return v


# ──────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────
def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(
        description="Audit of 5 synthetic controls x 2 scales (p0) "
                    "through the audit pipeline.")
    ap.add_argument("--n-perm", type=int, default=ad.N_PERM,
                    help="permutations per test (default %d)" % ad.N_PERM)
    ap.add_argument("--no-write", dest="write", action="store_false",
                    default=True, help="do not write the JSON")
    args = ap.parse_args(argv)

    t_start = time.time()
    print("Batch: %d strategies x %d scales p0=%s"
          % (len(STRATEGIES), len(P0S), P0S), flush=True)
    lote = run_lote(args.n_perm)
    elapsed = time.time() - t_start
    veredicto = build_veredicto(lote)

    # ── summary table ───────────────────────────────────────────────
    L = []
    L.append("=" * 100)
    L.append("AUDIT OF THIRD-PARTY STRATEGIES — 5 strategies x 2 scales  "
             "(%.0fs of CPU)" % elapsed)
    L.append("market: random walk %d days, seed %d, sigma %g; alpha=%.2f, "
             "%d perm/test, spread 2x%d bps, hold %d bars"
             % (ad.N_DAYS, SEED_MERCADO, ad.SIGMA, ad.ALPHA, args.n_perm,
                ad.HALF_SPREAD * 1e4, ad.HOLD))
    L.append("=" * 100)
    L.append("")
    L.append("%-20s %-6s %6s %8s %12s %8s %7s  %s  %s"
             % ("strategy", "type", "p0", "n_tr", "P&L/trade", "Sharpe",
                "win%", "flags(1-5)", "GLOBAL"))
    for name, rec in lote.items():
        for p0 in P0S:
            r = rec["p0"][p0]
            L.append("%-20s %-6s %6g %8d %+12.5f %8.2f %6.0f%%  %s  %s%s"
                     % (name, rec["tipo"][:6], p0, r["n_trades"],
                        r["mean_pnl"], r["sharpe"], r["win"] * 100,
                        r["verdicts"], r["global"],
                        "  <- flagged" if r["flagged"] else ""))
        L.append("  setup: %s (thr tuned=%s)" % (rec["setup"],
                                                 rec["p0"][P0S[0]]["thr"]))
        L.append("")
    L.append("honest plumbing: walk-forward, entry_lag=1, paid spread, "
             "reports full 2nd half")
    L.append("trap plumbing: in-sample tuning, entry_lag=0, fills at mid, "
             "worst decile dropped (mirror of the demo TRAP client)")
    L.append("")

    # ── JSON ────────────────────────────────────────────────────────
    result = {
        "params": {
            "alpha": ad.ALPHA, "n_perm": args.n_perm, "n_days": ad.N_DAYS,
            "bars": ad.N_DAYS * 60 * 24, "hold": ad.HOLD,
            "half_spread": ad.HALF_SPREAD, "sigma": ad.SIGMA,
            "half_spread_modelo": "constant bps across scales: "
                                  "half_spread_efectivo = HALF_SPREAD * p0 / P0",
            "p0s": P0S, "seed_mercado": SEED_MERCADO,
            "seeds_senal": {n: SIG_SEED_BASE + i
                            for i, (n, _t, _f, _d) in enumerate(STRATEGIES)},
            "audit_rng_base": AUDIT_RNG_BASE,
            "honest_plumbing": "walk-forward tuning (1st half), entry_lag=1, "
                               "price_book=True, half_spread, reports full "
                               "2nd half",
            "trap_plumbing": "in-sample tuning, entry_lag=0, price_book="
                             "False, drop_worst_decile (mirror TRAP client)",
            "global_definition": "worst verdict among the 5 detectors; "
                                 "flagged = at least one FAIL/WARN",
            "elapsed_s": round(elapsed, 1), "cpu": "CPU only, stdlib",
        },
        "estrategias": {
            n: {
                "tipo": r["tipo"], "setup": r["setup"],
                "seed_senal": r["seed_senal"],
                "p0": {
                    str(p0): {
                        "thr": rr["thr"], "n_trades": rr["n_trades"],
                        "mean_pnl": rr["mean_pnl"],
                        "total_pnl": rr["total_pnl"],
                        "sharpe": rr["sharpe"], "win": rr["win"],
                        "flags": rr["flags"], "verdicts": rr["verdicts"],
                        "global": rr["global"], "flagged": rr["flagged"],
                    } for p0, rr in r["p0"].items()
                },
                "checks_con_escala_diferente": escala_identica(r),
                "esperado": ("benign (unflagged)" if r["tipo"] == "honest"
                             else "flagged (WARN/FAIL)"),
                "ok": expected_ok(r),
            } for n, r in lote.items()
        },
        "limitaciones_conocidas": {
            "spread_sniper x supervivencia": {
                "estado": "RESOLVED (forensic audit 2026-10-24)",
                "diagnostico_anterior_y_erroneo": "detector limitation: "
                               "every losing trade pays the same fixed "
                               "round-trip cost (point mass at "
                               "−2·half_spread), which compresses the IQR "
                               "of the pnls and mechanically inflates the "
                               "count of right-tail outliers (nr > nl)",
                "por_que_era_erroneo": "pnl_net = pnl_gross − 2·half_spread "
                               "is a PURE translation and tail_symmetry_test "
                               "is rank-based: invariant to translation. "
                               "Verified: (nl, nr, p) identical net vs "
                               "gross in 48/48 runs (24 seeds × 2 "
                               "scales), and the sniper's FP is identical "
                               "7/24 with a 200× different relative cost "
                               "(p0=0.5 vs p0=100) — the cost cannot be "
                               "the cause",
                "causa_raiz": "violation of sample independence: a "
                               "level-selective sniper enters in BURSTS "
                               "(one price excursion generates a run of "
                               "entries whose hold windows share the same "
                               "realized path); the binomial test treated "
                               "the 9.7k–54k reported rows as independent "
                               "draws (worst seed: a single cluster "
                               "concentrates 61% of the trades) — FP 29.2% "
                               "(7/24) on honest clients vs 1% nominal",
                "solucion": "check 5 runs tail_symmetry_test over the "
                               "NON-overlapping trades (ad.nonoverlapping_"
                               "pnls, same sampling units as the honest "
                               "Sharpe); with <200 independent units it "
                               "reports SKIP, never a verdict on an "
                               "inflated sample",
                "potencia_preservada": "re-run of the MC with the "
                               "corrected detector: power intact on the "
                               "look-ahead traps (results/"
                               "monte_carlo_fp_power.json) and the demo "
                               "trap still FAIL (self_check OK); the "
                               "24 seeds × 2 scales study shows "
                               "abs vs abs_noov per seed (results/"
                               "estudio_limitacion_supervivencia.json)",
                "lectura_correcta": "unchanged: a FAIL in check 5 still "
                               "detects tail asymmetry — now over "
                               "independent units; the "
                               "PERMUTACION-DE-POSICIONES (null with "
                               "level structure) remains proposed as an "
                               "optional improvement",
            },
        },
        "veredicto": veredicto,
    }

    for line in L:
        print(line)
    print("VERDICTS_BY_STRATEGY: " + " | ".join(
        "%s[%s]: %s@%g/%s@%g -> %s"
        % (n, r["tipo"], r["p0"][P0S[0]]["verdicts"], P0S[0],
           r["p0"][P0S[1]]["verdicts"], P0S[1],
           "OK" if expected_ok(r) else "PROBLEM")
        for n, r in lote.items()))
    print("VERDICT: " + veredicto)

    if args.write:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = os.path.join(RESULTS_DIR, "audit_controles_sinteticos.json")
        with open(out, "w") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        print("written:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
