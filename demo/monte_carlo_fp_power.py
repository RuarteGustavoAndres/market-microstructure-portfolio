#!/usr/bin/env python3
"""Monte Carlo de los DETECTORES del auditor — tasa de falsos positivos y potencia.

No es el test nulo del P&L del modelo (eso es `monte_carlo_audit.py`): acá se
mide la calidad del auditor mismo, es decir, los cinco detectores de
`audit_demo.run_audit` como instrumentos de medición:

  FP rate (falsos positivos) — N estrategias NULAS: mercado sin edge
      (random walk) + señal sin poder predictivo en SIETE niveles
      (ruido puro iid, momentum causal, ruido + micro-drift, ruido AR(1),
      stop-loss mecánico declarado, take-profit asimétrico mecánico,
      alpha rápido real chico), plumbing HONESTO de punta a punta (tuning
      walk-forward, primera entrada ejecutable, spread pagado, lista
      completa de trades — el mismo cliente "clean" del demo). Por detector:
      % de estrategias nulas que cada uno marca (FAIL o WARN) con intervalo
      de Wilson 95% + % marcada por al menos uno.

  Potencia (sensibilidad) — N estrategias TRAMPA con intensidad variable:
        lookahead_completo — la señal ve el retorno del hold COMPLETO
                             (peek de 30 barras); plumbing roto de demo.
        lookahead_1bar     — la trampa exacta del demo (peek de 1 barra).
        lookahead_50pct    — peek de 1 barra en 50% de los trades; en el
                             otro 50% la señal es ruido puro (mismos
                             marginales: el tuning no puede distinguirlas).
        edge_realo_pequeno — NO es trampa: edge real genuino de +10% del
                             half-spread por trade (componente predictable
                             del retorno, alpha observable antes de entrar),
                             plumbing honesto. Caso duro de "no falsear":
                             la detección baja ahí es comportamiento
                             CORRECTO, no falencia (los detectores cazan
                             sesgos de plumbing, no la existencia de edge).
      El plumbing de las tres trampas mirror a la trampa del demo: tuning
      in-sample, fills a mid, peor decil descartado. Por intensidad: %
      detectadas por cada detector y por el conjunto.

  Los 3 nulos nuevos (P2 24-10-2026):
      stop_loss           — exits mecánicos: sale antes si el drawdown
                            adverso supera 2 x sigma_path; DECLARADO
                            (stop_loss=True → check 5 SKIP por diseño,
                            la cola izquierda está truncada a propósito).
      pagos_asimetricos   — take-profit mecánico a +3 x half_spread bruto,
                            pérdidas corren hasta el hold completo; NO
                            declarado (exits mecánicos, nada filtrado:
                            prueba si check 5 confunde payoff mecánico
                            con survivorship — la cola truncada acá es
                            la DERECHA, no la izquierda).
      alpha_rapido_real   — edge real chico (+10% del half-spread) por el
                            pipeline honesto; es el caso nulo del detector:
                            un cliente genuinamente rentable NO debería
                            marcarse (idéntico a la intensidad
                            edge_realo_pequeno, contado en el lado FP).

Honestidad: si un detector tiene FP alto o potencia baja se reporta tal cual
— es exactamente lo que hay que saber antes de ofrecer el servicio.

Determinista: semillas fijas, todas reportadas en el JSON. Stdlib only.

Paralelo + checkpoints (P2): --jobs N reparte las corridas entre N procesos
(solo CPU, stdlib); cada corrida terminada se agrega a un checkpoint JSONL
(flush + fsync), así un corte se retoma con --resume sin repetir trabajo.

Intervalo de Wilson 95% (P2b): para k marcadas en n corridas con z=1.959964:
    centro = (p + z^2/2n) / (1 + z^2/n)
    half   = z*sqrt(p(1-p)/n + z^2/4n^2) / (1 + z^2/n)
Con k=0, n=50 el límite superior es ~7.1%; con n=1000, ~1.1%. El intervalo
se calcula en este script y viaja al JSON junto a cada tasa.

CLI:
    python3 demo/monte_carlo_fp_power.py [--n-perm 999] [--per-type 50]
                                         [--per-intensity 100] [--no-write]
                                         [--jobs N] [--checkpoint PATH]
                                         [--resume]

Escribe results/monte_carlo_fp_power.json y una tabla resumen a stdout.
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

# Cinco detectores, en el orden en que run_audit los emite.
DETS = ["capturabilidad", "fills_al_book", "tuning_in_sample", "kfold_cv",
        "supervivencia"]
FLAG = ("FAIL", "WARN")   # un detector "marca" si devuelve FAIL o WARN

# Semillas base (bloques contiguos; ver "seeds" del JSON).
NULL_SEED_BASE = 2_000_000
TRAP_SEED_BASE = 3_000_000
AUDIT_RNG_BASE = 10_000_000

NULL_TYPES = ["ruido_puro", "momento_causal", "ruido_micro_drift", "ruido_ar1",
              "stop_loss", "pagos_asimetricos", "alpha_rapido_real"]
INTENSITIES = ["lookahead_completo", "lookahead_1bar", "lookahead_50pct",
               "edge_realo_pequeno"]

WILSON_Z = 1.959963984540054   # z para IC 95% (dos colas)
WILSON_FORMULA = ("centro = (p + z^2/(2n)) / (1 + z^2/n); "
                  "half = z*sqrt(p(1-p)/n + z^2/(4n^2)) / (1 + z^2/n); "
                  "z = 1.959963984540054 (95% dos colas)")

# Parametrización de los exits mecánicos de los nulos nuevos.
STOP_MULT = 2.0            # stop-loss a 2 x sigma_path (sigma * sqrt(hold))
TP_MULT_HALF_SPREAD = 3.0  # take-profit a +3 x half_spread bruto


# ──────────────────────────────────────────────────────────────────────
# Señales
# ──────────────────────────────────────────────────────────────────────
def null_signal(kind, px, rng):
    """Señal sin poder predictivo sobre el random walk, 4 niveles base."""
    if kind == "ruido_puro":
        return [rng.gauss(0.0, 1.0) for _ in px]
    if kind == "momento_causal":
        # momentum de 5 barras pasadas: honesto, y sin poder en random walk
        return ad.causal_signal(px)
    if kind == "ruido_micro_drift":
        # ruido + sesgo constante en la señal (no hay edge: la dirección
        # sesgada no predice nada en un random walk, solo cobra spread)
        return [rng.gauss(0.0, 1.0) + 0.25 for _ in px]
    if kind == "ruido_ar1":
        # ruido persistente AR(1) estacionario N(0,1): realista, sin poder
        phi, out, x = 0.5, [], 0.0
        for _ in px:
            x = phi * x + rng.gauss(0.0, math.sqrt(1.0 - phi * phi))
            out.append(x)
        return out
    raise ValueError(kind)


def trap_signal(kind, px, rng):
    """Señal trampa según la intensidad."""
    if kind == "lookahead_completo":
        # peek del hold COMPLETO: la "indicación" en t es el retorno de
        # t a t+hold, normalizado — información que no existe al decidir
        H, s, n = ad.HOLD, ad.SIGMA, len(px)
        return [(px[i + H] / px[i] - 1.0) / (s * math.sqrt(H))
                if i + H < n else 0.0 for i in range(n)]
    if kind == "lookahead_1bar":
        return ad.look_ahead_signal(px)
    if kind == "lookahead_50pct":
        # peek de 1 barra en 50% de las barras; el resto, ruido puro.
        # Mismos marginales N(0,1): estadísticamente indistinguible del
        # nivel sin leak para cualquier tuning in-sample.
        base = ad.look_ahead_signal(px)
        return [b if rng.random() < 0.5 else rng.gauss(0.0, 1.0) for b in base]
    raise ValueError(kind)


def build_market_alpha(n, p0, s_eps, sigma_z, seed):
    """Random walk + componente predictable eps[b] ~ N(0, s_eps^2) en cada
    log-retorno. eps[b] se modela como observable con 2 barras de lead (el
    alpha que se conoce antes de la barra que paga — ej. order flow).
    Devuelve (px, eps)."""
    rng = random.Random(seed)
    px, eps = [p0], [0.0]
    for _ in range(n - 1):
        e = rng.gauss(0.0, s_eps)
        px.append(px[-1] * math.exp(e + rng.gauss(0.0, sigma_z)))
        eps.append(e)
    return px, eps


# ──────────────────────────────────────────────────────────────────────
# Backtests con exits mecánicos (nulos stop_loss / pagos_asimetricos)
# ──────────────────────────────────────────────────────────────────────
def backtest_stop(px, sig, thr, stop_level, entry_lag=1, half_spread=ad.HALF_SPREAD,
                  indices=None, hold=ad.HOLD):
    """Espejo de ad.run_backtest con stop-loss mecánico: la posición sale en
    la PRIMERA barra en que el drawdown adverso alcanza -stop_level (bruto);
    si nunca lo toca, mantiene hasta el hold completo. Spread pagado."""
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
    """Espejo de ad.run_backtest con take-profit mecánico: sale en la PRIMERA
    barra con ganancia bruta >= +tp_level; si nunca la toca, mantiene hasta
    el hold completo (las pérdidas corren). Spread pagado. Nada se filtra."""
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
    """Espejo de ad.tune_threshold para un backtest con exits mecánicos:
    misma grilla (lo..hi, 18 pasos), mismo criterio (Sharpe no-solapado),
    tuning honesto sobre la ventana dada (walk-forward)."""
    best_thr, best_sh = lo, -9e9
    for i in range(steps):
        thr = lo + i * (hi - lo) / (steps - 1)
        sh = ad.metrics(bt(px, sig, thr, indices=indices))["sharpe"]
        if sh > best_sh:
            best_sh, best_thr = sh, thr
    return best_thr


# ──────────────────────────────────────────────────────────────────────
# Pipelines de cliente (nulo honesto / trampa rota)
# ──────────────────────────────────────────────────────────────────────
def build_null_client(px, sig):
    """Cliente NULO honesto: tuning walk-forward (primera mitad), primera
    entrada ejecutable, spread pagado, reporta la segunda mitad completa.
    Mismo pipeline que el control 'clean' del demo."""
    half = len(px) // 2
    thr = ad.tune_threshold(px, sig, indices=(0, half),
                            half_spread=ad.HALF_SPREAD)
    booked = ad.run_backtest(px, sig, thr, entry_lag=1, price_book=True,
                             half_spread=ad.HALF_SPREAD,
                             indices=(half, len(px)))
    return booked, sig, thr


def build_trap_client(px, sig):
    """Cliente TRAMPA (mirror de la trampa del demo): tuning in-sample en
    la ventana completa, fills a mid, peor decil descartado."""
    thr = ad.tune_threshold(px, sig, entry_lag=0, price_book=False,
                            half_spread=0.0)
    booked = ad.drop_worst_decile(
        ad.run_backtest(px, sig, thr, entry_lag=0, price_book=False))
    return booked, sig, thr


def audit_flags(booked, px, sig, rng, n_perm, stop_loss=False):
    """Corre run_audit y devuelve (flags por detector 0/1, verdicts F/W/P/S,
    metrics reportados)."""
    f, m = ad.run_audit(booked, px, sig, half_spread=ad.HALF_SPREAD,
                        rng=rng, n_perm=n_perm, stop_loss=stop_loss)
    return ([1 if x["verdict"] in FLAG else 0 for x in f],
            "".join(x["verdict"][0] for x in f), m)


# ──────────────────────────────────────────────────────────────────────
# Jobs y workers (compartido por el camino serial y el paralelo)
# ──────────────────────────────────────────────────────────────────────
def build_jobs(per_type, per_int):
    """Lista determinista de jobs (group, kind, i, seed) en orden de bloques:
    nulos primero (por tipo), trampas después (por intensidad)."""
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
    """Ejecuta UNA auditoría Monte Carlo. Devuelve dict serializable con
    group/kind/seed/flags/verdicts/mean/tiempo. Determinista por semilla."""
    group, kind, i, seed = job
    t0 = time.time()
    n = ad.N_DAYS * 60 * 24
    if group == "null" and kind in ("stop_loss", "pagos_asimetricos"):
        # exits mecánicos sobre mercado nulo + señal sin poder
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
        # edge real chico por el pipeline HONESTO (mismo motor que la
        # intensidad edge_realo_pequeno, contado en el lado FP)
        s_eps = 0.1 * ad.HALF_SPREAD * math.sqrt(math.pi / 2.0)
        sigma_z = math.sqrt(ad.SIGMA ** 2 - s_eps ** 2)
        px, eps = build_market_alpha(n, ad.P0, s_eps, sigma_z, seed)
        sig = [eps[t + 2] / ad.SIGMA if t + 2 < n else 0.0
               for t in range(n)]
        booked, sig, _thr = build_null_client(px, sig)
        flags, verdicts, m = audit_flags(booked, px, sig,
                                         random.Random(AUDIT_RNG_BASE + i),
                                         n_perm)
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
            "t": round(time.time() - t0, 2)}


def _pool_job(args):
    group, kind, i, seed, n_perm = args
    return run_one((group, kind, i, seed), n_perm)


# ──────────────────────────────────────────────────────────────────────
# Partes FP y potencia (serial, camino legacy)
# ──────────────────────────────────────────────────────────────────────
def run_fp(n_perm, per_type):
    """per_type x len(NULL_TYPES) estrategias nulas honestas por el pipeline
    de auditoría. Devuelve lista de (tipo, seed, flags, verdicts, mean)."""
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
    """4 intensidades x per_int estrategias trampa. Devuelve lista de
    (intensidad, seed, flags, verdicts, mean)."""
    out = []
    jobs = build_jobs(0, per_int)
    for k, (group, kind, i, seed) in enumerate(jobs):
        t0 = time.time()
        r = run_one((group, kind, i, seed), n_perm)
        out.append((r["kind"], r["seed"], r["flags"], r["verdicts"], r["mean"]))
        if k % 25 == 0:
            print("  [POT %3d/%d] %s seed=%d flags=%s %.1fs"
                  % (k + 1, len(jobs), kind, seed, r["verdicts"],
                     time.time() - t0), flush=True)
    return out


# ──────────────────────────────────────────────────────────────────────
# Paralelo + checkpoints
# ──────────────────────────────────────────────────────────────────────
def run_jobs_parallel(jobs, n_perm, jobs_n, ckpt_path, resume):
    """Corre jobs en un Pool de jobs_n procesos; cada resultado se agrega al
    checkpoint JSONL (flush + fsync) para poder retomar tras un corte.
    Devuelve la lista completa de dicts de resultados (resumidos + nuevos)."""
    done = {}
    if resume and os.path.exists(ckpt_path):
        with open(ckpt_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                done[(r["group"], r["kind"], r["i"])] = r
        print("  resume: %d corridas ya en checkpoint" % len(done), flush=True)
    todo = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    print("  jobs: %d por correr, %d ya hechas" % (len(todo), len(done)),
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
                    print("  [%4d/%4d] %.1f corridas/s, ETA %.0f min"
                          % (k, len(todo), k / el,
                             (len(todo) - k) * el / k / 60.0), flush=True)
    return results


def _pool_job(args):
    group, kind, i, seed, n_perm = args
    return run_one((group, kind, i, seed), n_perm)


# ──────────────────────────────────────────────────────────────────────
# Agregación y reporte
# ──────────────────────────────────────────────────────────────────────
def wilson(k, n, z=WILSON_Z):
    """Intervalo de Wilson 95% para una proporción k/n. Devuelve
    (lo, hi, p). Con k=0: el límite superior acota FP < ~3/n."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half), p)


def _block(rows):
    """rows: lista de dicts con 'flags' y 'verdicts'. Devuelve (n, flags por
    detector, union, distribución de verdicts)."""
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
    """Agrega por detector con desglose por grupo (tipo/intensidad) y IC
    de Wilson 95% en cada tasa."""
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
        description="Monte Carlo FP/potencia de los detectores del auditor.")
    ap.add_argument("--n-perm", type=int, default=ad.N_PERM,
                    help="permutaciones por test (default %d)" % ad.N_PERM)
    ap.add_argument("--per-type", type=int, default=50,
                    help="estrategias nulas por tipo de señal (default 50)")
    ap.add_argument("--per-intensity", type=int, default=100,
                    help="estrategias trampa por intensidad (default 100)")
    ap.add_argument("--no-write", dest="write", action="store_false",
                    default=True, help="no escribir el JSON")
    ap.add_argument("--jobs", type=int, default=1,
                    help="procesos en paralelo (default 1 = serial legacy)")
    ap.add_argument("--checkpoint", default=None,
                    help="JSONL con corridas terminadas (default: "
                         "results/mc_fp_power_checkpoint.jsonl)")
    ap.add_argument("--resume", action="store_true",
                    help="retomar desde el checkpoint (--jobs > 1)")
    args = ap.parse_args(argv)

    ckpt = args.checkpoint or os.path.join(RESULTS_DIR,
                                           "mc_fp_power_checkpoint.jsonl")
    t_start = time.time()
    jobs = build_jobs(args.per_type, args.per_intensity)
    n_nulls = args.per_type * len(NULL_TYPES)
    n_traps = args.per_intensity * len(INTENSITIES)
    print("FP: %d estrategias nulas (%d x %d tipos) + POT: %d trampas "
          "(%d x %d) = %d auditorias, jobs=%d"
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
                print("  [POT %3d/%d] %s seed=%d flags=%s %.1fs"
                      % (k - n_nulls + 1, n_traps, kind, seed, r["verdicts"],
                         time.time() - t0), flush=True)
    elapsed = time.time() - t_start

    fp_rows = [r for r in rows if r["group"] == "null"]
    pot_rows = [r for r in rows if r["group"] == "trap"]
    assert len(fp_rows) == n_nulls and len(pot_rows) == n_traps, \
        "faltan corridas: %d nulos + %d trampas de %d + %d esperados" % (
            len(fp_rows), len(pot_rows), n_nulls, n_traps)
    fp = summarize(fp_rows, NULL_TYPES)
    pot = {k: summarize([r for r in pot_rows if r["kind"] == k], [k])
           for k in INTENSITIES}

    # ── tablas resumen ──────────────────────────────────────────────
    L = []
    L.append("=" * 78)
    L.append("MONTE CARLO DE DETECTORES — FP y potencia  (%.0fs de CPU)"
             % elapsed)
    L.append("alpha=%.2f, %d permutaciones/test, mercado %d días sin edge, "
             "spread 2x%d bps" % (ad.ALPHA, args.n_perm, ad.N_DAYS,
                                  ad.HALF_SPREAD * 1e4))
    L.append("IC de Wilson 95%%: %s" % WILSON_FORMULA)
    L.append("=" * 78)
    L.append("")
    L.append("FALSOS POSITIVOS — %d estrategias nulas honestas (sin edge, "
             "plumbing limpio), %d x %d tipos" % (len(fp_rows), args.per_type,
                                                  len(NULL_TYPES)))
    L.append("  %-18s %10s %7s %-22s por tipo de señal (n=%d c/u)"
             % ("detector", "marc.", "FP % [IC95]", "", args.per_type))
    for d, name in enumerate(DETS):
        por_tipo = "  ".join(
            "%s=%.0f%%" % (g[:6], 100 * fp["_por_grupo"][g][name]["rate"])
            for g in NULL_TYPES)
        L.append("  %-18s %6d %5.1f%% [%4.1f,%4.1f]  %s"
                 % (name, fp[name]["marcadas"], 100 * fp[name]["rate"],
                    100 * fp[name]["ci95"][0], 100 * fp[name]["ci95"][1],
                    por_tipo))
    L.append("  %-18s %6d %5.1f%% [%4.1f,%4.1f]  <- marcada por al menos un "
             "detector" % ("UNION", fp["_union"]["marcadas"],
                           100 * fp["_union"]["rate"],
                           100 * fp["_union"]["ci95"][0],
                           100 * fp["_union"]["ci95"][1]))
    L.append("")
    L.append("POTENCIA — estrategias trampa por intensidad (n=%d c/u)"
             % args.per_intensity)
    L.append("  %-20s " % "intensidad" +
             " ".join("%13s" % d for d in DETS) + " %10s" % "UNION")
    for kind in INTENSITIES:
        L.append("  %-20s " % kind +
                 " ".join("%5d/%-4d" % (pot[kind][d]["marcadas"],
                                        pot[kind][d]["n"]) for d in DETS) +
                 " %5d/%-4d" % (pot[kind]["_union"]["marcadas"],
                                pot[kind]["_union"]["n"]))
    L.append("")

    # ── veredicto (construido a partir de los datos, no escrito a mano) ──
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
        notas.append("c1(capturabilidad) no pilla look-ahead de hold completo "
                     "(el retraso de 1 barra solo quita 1/30 del peek; "
                     "detecta el peek de 1 barra: %d%%)"
                     % int(100 * pot["lookahead_1bar"]["capturabilidad"]["rate"]))
    if c2_full < 0.5:
        notas.append("c2(fills al book) no dispara cuando el edge supera el "
                     "costo del round trip (edge completo > 2x spread)")
    if pot_edge <= 0.10:
        notas.append("edge real pequeño no detectado (%.0f%%) — correcto: "
                     "no es trampa y los detectores cazan sesgos de plumbing, "
                     "no la existencia de edge" % (100 * pot_edge))
    veredicto = ("AUDIT OK: FP <= 10%% por detector (max %.1f%%) y potencia >= "
                 "80%% en trampas evidentes (min %.0f%%); union FP %.1f%%, "
                 "union potencia trampas %.0f%%"
                 % (100 * fp_max, 100 * min(pot_obvios.values()),
                    100 * fp["_union"]["rate"], 100 * min(pot_obvios.values()))
                 if (fp_ok and pot_ok)
                 else "REVISAR: %s" % (
                     "FP alto en %s" % [d for d in DETS if fp[d]["rate"] > 0.10]
                     if not fp_ok else
                     "potencia baja en %s" % [k for k, v in pot_obvios.items()
                                              if v < 0.80]))
    if notas:
        veredicto += "; limitaciones honestas: " + " | ".join(notas)

    # ── JSON ────────────────────────────────────────────────────────
    result = {
        "params": {
            "alpha": ad.ALPHA, "n_perm": args.n_perm, "n_days": ad.N_DAYS,
            "bars": ad.N_DAYS * 60 * 24, "hold": ad.HOLD,
            "half_spread": ad.HALF_SPREAD, "sigma": ad.SIGMA,
            "null_types": NULL_TYPES, "intensities": INTENSITIES,
            "n_null_per_type": args.per_type,
            "n_trap_per_intensity": args.per_intensity,
            "flag_definition": "verdict FAIL o WARN",
            "null_client": "tuning walk-forward, entry_lag=1, spread pagado, "
                           "reporta segunda mitad completa (control honesto)",
            "trap_client": "tuning in-sample, entry_lag=0, fills a mid, "
                           "drop_worst_decile (mirror de la trampa del demo)",
            "stop_loss_null": "exits mecánicos: stop a %.1f x sigma_path "
                              "(sigma*sqrt(hold) = %.5f bruto); DECLARADO "
                              "(stop_loss=True → check 5 SKIP por diseño)"
                              % (STOP_MULT, STOP_MULT * ad.SIGMA
                                 * math.sqrt(ad.HOLD)),
            "pagos_asimetricos_null": "take-profit mecánico a +%.1f x "
                                      "half_spread bruto (+%.4f), pérdidas "
                                      "corren hasta el hold; NO declarado "
                                      "(cola truncada = DERECHA, nada "
                                      "filtrado: prueba si check 5 confunde "
                                      "payoff mecánico con survivorship)"
                                      % (TP_MULT_HALF_SPREAD,
                                         TP_MULT_HALF_SPREAD * ad.HALF_SPREAD),
            "alpha_rapido_real_null": "edge genuino = 0.1 x half_spread/trade "
                                      "bruto (E|eps| = sqrt(2/pi) s_eps), "
                                      "plumbing honesto; mismo motor que la "
                                      "intensidad edge_realo_pequeno, "
                                      "contado en el lado FP",
            "edge_realo_pequeno": "edge genuino = 0.1 x half_spread/trade "
                                  "bruto (E|eps| = sqrt(2/pi) s_eps), "
                                  "plumbing honesto; NO es trampa",
            "wilson_ci": WILSON_FORMULA,
            "elapsed_s": round(elapsed, 1),
            "jobs": args.jobs,
            "cpu": "solo CPU, stdlib",
        },
        "seeds": {
            "null_market": {"start": NULL_SEED_BASE,
                            "count": len(fp_rows),
                            "note": "seed i-ésima = start + i, por tipo "
                                    "(bloques de n_null_per_type)"},
            "trap_market": {"start": TRAP_SEED_BASE,
                            "blocks": {k: {"start": TRAP_SEED_BASE + t * 1000,
                                           "count": args.per_intensity}
                                       for t, k in enumerate(INTENSITIES)},
                            "note": "cada intensidad tiene un bloque de 1000"},
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
    print("FP_RATE_POR_DETECTORES: " + "; ".join(
        "%s=%.1f%% (%d/%d) IC95=[%.1f%%, %.1f%%]"
        % (d, 100 * fp[d]["rate"], fp[d]["marcadas"], fp[d]["n"],
           100 * fp[d]["ci95"][0], 100 * fp[d]["ci95"][1]) for d in DETS)
        + "; union=%.1f%% (%d/%d) IC95=[%.1f%%, %.1f%%]"
        % (100 * fp["_union"]["rate"], fp["_union"]["marcadas"],
           fp["_union"]["n"], 100 * fp["_union"]["ci95"][0],
           100 * fp["_union"]["ci95"][1]))
    print("POTENCIA_POR_INTENSIDAD: " + "; ".join(
        "%s: union=%.1f%% [%s]" % (
            k, 100 * pot[k]["_union"]["rate"],
            "+".join("%s=%d%%" % (d[:4], 100 * pot[k][d]["rate"])
                     for d in DETS))
        for k in INTENSITIES))
    print("VEREDICTO: " + veredicto)

    if args.write:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = os.path.join(RESULTS_DIR, "monte_carlo_fp_power.json")
        with open(out, "w") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        print("written:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
