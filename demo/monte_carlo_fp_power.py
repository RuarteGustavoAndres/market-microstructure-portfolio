#!/usr/bin/env python3
"""Monte Carlo de los DETECTORES del auditor — tasa de falsos positivos y potencia.

No es el test nulo del P&L del modelo (eso es `monte_carlo_audit.py`): acá se
mide la calidad del auditor mismo, es decir, los cinco detectores de
`audit_demo.run_audit` como instrumentos de medición:

  FP rate (falsos positivos) — N=200 estrategias NULAS: mercado sin edge
      (random walk) + señal sin poder predictivo en cuatro niveles
      (ruido puro iid, momentum causal, ruido + micro-drift, ruido AR(1)),
      plumbing HONESTO de punta a punta (tuning walk-forward, primera entrada
      ejecutable, spread pagado, lista completa de trades — el mismo cliente
      "clean" del demo). Por detector: % de estrategias nulas que cada uno
      marca (FAIL o WARN) + % marcada por al menos uno.

  Potencia (sensibilidad) — N=400 estrategias TRAMPA con intensidad variable
      (100 por intensidad):
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

Honestidad: si un detector tiene FP alto o potencia baja se reporta tal cual
— es exactamente lo que hay que saber antes de ofrecer el servicio.

Determinista: semillas fijas, todas reportadas en el JSON. Stdlib only.

CLI:
    python3 demo/monte_carlo_fp_power.py [--n-perm 999] [--per-type 50]
                                         [--per-intensity 100] [--no-write]

Escribe results/monte_carlo_fp_power.json y una tabla resumen a stdout.
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

# Cinco detectores, en el orden en que run_audit los emite.
DETS = ["capturabilidad", "fills_al_book", "tuning_in_sample", "kfold_cv",
        "supervivencia"]
FLAG = ("FAIL", "WARN")   # un detector "marca" si devuelve FAIL o WARN

# Semillas base (bloques contiguos; ver "seeds" del JSON).
NULL_SEED_BASE = 2_000_000
TRAP_SEED_BASE = 3_000_000
AUDIT_RNG_BASE = 10_000_000

NULL_TYPES = ["ruido_puro", "momento_causal", "ruido_micro_drift", "ruido_ar1"]
INTENSITIES = ["lookahead_completo", "lookahead_1bar", "lookahead_50pct",
               "edge_realo_pequeno"]


# ──────────────────────────────────────────────────────────────────────
# Señales
# ──────────────────────────────────────────────────────────────────────
def null_signal(kind, px, rng):
    """Señal sin poder predictivo sobre el random walk, 4 niveles."""
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


def audit_flags(booked, px, sig, rng, n_perm):
    """Corre run_audit y devuelve (flags por detector 0/1, verdicts F/W/P/S,
    metrics reportados)."""
    f, m = ad.run_audit(booked, px, sig, half_spread=ad.HALF_SPREAD,
                        rng=rng, n_perm=n_perm)
    return ([1 if x["verdict"] in FLAG else 0 for x in f],
            "".join(x["verdict"][0] for x in f), m)


# ──────────────────────────────────────────────────────────────────────
# Partes FP y potencia
# ──────────────────────────────────────────────────────────────────────
def run_fp(n_perm, per_type):
    """200 (o 4*per_type) estrategias nulas honestas por el pipeline de
    auditoría. Devuelve lista de (tipo, seed, flags, verdicts, mean)."""
    out = []
    for ti, kind in enumerate(NULL_TYPES):
        for j in range(per_type):
            i = ti * per_type + j
            seed = NULL_SEED_BASE + i
            t0 = time.time()
            px = ad.build_market(seed=seed)
            sig = null_signal(kind, px, random.Random(seed + 500_000))
            booked, sig, _thr = build_null_client(px, sig)
            flags, verdicts, m = audit_flags(booked, px, sig,
                                             random.Random(AUDIT_RNG_BASE + i),
                                             n_perm)
            out.append((kind, seed, flags, verdicts, m["mean"]))
            if i % 20 == 0:
                print("  [FP %3d/%d] %s seed=%d flags=%s %.1fs"
                      % (i + 1, per_type * len(NULL_TYPES), kind, seed,
                         verdicts, time.time() - t0), flush=True)
    return out


def run_power(n_perm, per_int):
    """4 intensidades x per_int estrategias trampa. Devuelve lista de
    (intensidad, seed, flags, verdicts, mean)."""
    out = []
    n = ad.N_DAYS * 60 * 24
    # edge real pequeño: componente predictable s_eps tal que
    # E[|eps|] = sqrt(2/pi)*s_eps = 0.1 * half_spread por trade (bruto)
    s_eps = 0.1 * ad.HALF_SPREAD * math.sqrt(math.pi / 2.0)
    sigma_z = math.sqrt(ad.SIGMA ** 2 - s_eps ** 2)
    for it, kind in enumerate(INTENSITIES):
        for j in range(per_int):
            i = it * per_int + j
            seed = TRAP_SEED_BASE + it * 1000 + j
            t0 = time.time()
            if kind == "edge_realo_pequeno":
                # NO es trampa: edge real +10% del half-spread, plumbing
                # honesto (walk-forward, lag 1, spread, lista completa).
                # eps[t+2] es observable al decidir en t (2 barras de lead).
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
            out.append((kind, seed, flags, verdicts, m["mean"]))
            if j % 25 == 0:
                print("  [POT %3d/%d] %s seed=%d flags=%s %.1fs"
                      % (j + 1, per_int, kind, seed, verdicts,
                         time.time() - t0), flush=True)
    return out


# ──────────────────────────────────────────────────────────────────────
# Agregación y reporte
# ──────────────────────────────────────────────────────────────────────
def _block(rows):
    """rows: lista de (grupo, flags, verdicts). Devuelve (n, flags por
    detector, union, distribución de verdicts)."""
    n = len(rows)
    per = [sum(1 for _g, _s, fl, _v, _m in rows if fl[d]) for d in range(5)]
    union = sum(1 for _g, _s, fl, _v, _m in rows if any(fl))
    dist = {}
    for _g, _s, _fl, v, _m in rows:
        dist[v] = dist.get(v, 0) + 1
    return n, per, union, dist


def summarize(rows, groups):
    """Agrega por detector con desglose por grupo (tipo/intensidad)."""
    n, per, union, dist = _block(rows)
    out = {}
    for d, name in enumerate(DETS):
        out[name] = {"n": n, "marcadas": per[d], "rate": per[d] / n}
    out["_union"] = {"n": n, "marcadas": union, "rate": union / n}
    out["_verdicts_dist"] = dist
    out["_por_grupo"] = {}
    for g in groups:
        sub = [r for r in rows if r[0] == g]
        gn, gper, gunion, _ = _block(sub)
        out["_por_grupo"][g] = {
            name: {"n": gn, "marcadas": gper[d], "rate": gper[d] / gn}
            for d, name in enumerate(DETS)}
        out["_por_grupo"][g]["_union"] = {"n": gn, "marcadas": gunion,
                                          "rate": gunion / gn}
    return out


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(
        description="Monte Carlo FP/potencia de los detectores del auditor.")
    ap.add_argument("--n-perm", type=int, default=ad.N_PERM,
                    help="permutaciones por test (default %d)" % ad.N_PERM)
    ap.add_argument("--per-type", type=int, default=50,
                    help="estrategias nulas por tipo de señal (default 50, "
                         "total 200)")
    ap.add_argument("--per-intensity", type=int, default=100,
                    help="estrategias trampa por intensidad (default 100, "
                         "total 400)")
    ap.add_argument("--no-write", dest="write", action="store_false",
                    default=True, help="no escribir el JSON")
    args = ap.parse_args(argv)

    t_start = time.time()
    print("FP: %d estrategias nulas (%d x %d tipos)"
          % (args.per_type * len(NULL_TYPES), args.per_type,
             len(NULL_TYPES)), flush=True)
    fp_rows = run_fp(args.n_perm, args.per_type)
    print("Potencia: %d estrategias trampa (%d x %d intensidades)"
          % (args.per_intensity * len(INTENSITIES), args.per_intensity,
             len(INTENSITIES)), flush=True)
    pot_rows = run_power(args.n_perm, args.per_intensity)
    elapsed = time.time() - t_start

    fp = summarize(fp_rows, NULL_TYPES)
    pot = {k: summarize([r for r in pot_rows if r[0] == k], [k])
           for k in INTENSITIES}

    # ── tablas resumen ──────────────────────────────────────────────
    L = []
    L.append("=" * 78)
    L.append("MONTE CARLO DE DETECTORES — FP y potencia  (%.0fs de CPU)" % elapsed)
    L.append("alpha=%.2f, %d permutaciones/test, mercado %d días sin edge, "
             "spread 2x%d bps" % (ad.ALPHA, args.n_perm, ad.N_DAYS,
                                  ad.HALF_SPREAD * 1e4))
    L.append("=" * 78)
    L.append("")
    L.append("FALSOS POSITIVOS — %d estrategias nulas honestas (sin edge, "
             "plumbing limpio)" % len(fp_rows))
    L.append("  %-18s %6s %6s   por tipo de señal (n=%d c/u)"
             % ("detector", "marc.", "FP %", args.per_type))
    for d, name in enumerate(DETS):
        por_tipo = "  ".join("%s=%.0f%%" % (g[:6],
                                            100 * fp["_por_grupo"][g][name]["rate"])
                             for g in NULL_TYPES)
        L.append("  %-18s %6d %5.1f%%   %s"
                 % (name, fp[name]["marcadas"], 100 * fp[name]["rate"], por_tipo))
    L.append("  %-18s %6d %5.1f%%   <- marcada por al menos un detector"
             % ("UNION", fp["_union"]["marcadas"], 100 * fp["_union"]["rate"]))
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
            "edge_realo_pequeno": "edge genuino = 0.1 x half_spread/trade "
                                  "bruto (E|eps| = sqrt(2/pi) s_eps), "
                                  "plumbing honesto; NO es trampa",
            "elapsed_s": round(elapsed, 1),
            "cpu": "solo CPU, stdlib",
        },
        "seeds": {
            "null_market": {"start": NULL_SEED_BASE,
                            "count": len(fp_rows),
                            "note": "seed i-ésima = start + i, por tipo"},
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
        "%s=%.1f%% (%d/%d)" % (d, 100 * fp[d]["rate"], fp[d]["marcadas"],
                               fp[d]["n"]) for d in DETS)
        + "; union=%.1f%% (%d/%d)" % (100 * fp["_union"]["rate"],
                                      fp["_union"]["marcadas"],
                                      fp["_union"]["n"]))
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
