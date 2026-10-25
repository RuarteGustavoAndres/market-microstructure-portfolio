#!/usr/bin/env python3
"""Tareas 2-4: colas del sniper honesto a 2 escalas + controles + pnl relativo.

Evidencia del FALSO POSITIVO del detector "supervivencia"
(tail_symmetry_test) sobre spread_sniper, documentada en
results/audit_controles_sinteticos.json (limitaciones_conocidas)
y en el README: 29.2% = 7/24 FAIL sobre clientes HONESTOS. Y
validación del detector CORREGIDO (audit forense 2026-10-24):
  - Causa raíz real: tamaño muestral. El sniper selectivo de nivel
    entra en ráfaga (entradas solapadas que comparten el mismo
    camino realizado); el test binomial trataba la LISTA REPORTADA
    como dibujos independientes.
  - El costo fijo NO es la causa: pnl_net = pnl_gross - costo es
    una traslación pura y el test es rank-based (invariante por
    traslación) — (nl, nr, p) idénticos net vs gross 48/48; FP
    idéntico 7/24 con costo relativo 200x distinto.
  - Fix: el check 5 ahora corre sobre los trades NO solapados
    (ad.nonoverlapping_pnls, mismas unidades que el Sharpe honesto).
Correr desde demo/ o cualquier cwd: importa los
módulos del propio directorio. Stdlib only, solo CPU.
Salida: resumen impreso + JSON en results/estudio_limitacion_supervivencia.json.

  Tarea 2: 24 seeds de mercado x p0=0.5/100, pipeline honesto exacto de
           aea.build_honest_client, tail_symmetry_test(pnls) absolutos
           (detector ANTIGUO, registro "abs") y sobre trades no
           solapados (detector CORREGIDO, registro "abs_noov").
  Tarea 3: controles momentum_chaser y ruido_puro (señal 203 del 4h) en
           8 seeds a p0=0.5, mismo test.
  Tarea 4: variante relative-pnl del sniper:
           pnl_rel = pos*(px[x_t]/px[e_t]-1) - 2*half_spread_rel,
           half_spread_rel = HALF_SPREAD/0.5 = 0.008 (equivalente en bps
           a la escala base).
"""
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
sys.path.insert(0, HERE)
import audit_demo as ad               # noqa: E402
import audit_controles_sinteticos as aea  # noqa: E402

# 24 semillas de mercado (primas 7..103; las 6 del probe son un subconjunto).
SEEDS = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
         53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103]
P0S = [0.5, 100.0]
SEEDS_CTRL = SEEDS[:8]               # controles: 8 seeds a p0=0.5
HALF_SPREAD_REL = ad.HALF_SPREAD / 0.5   # 0.008
ALPHA = ad.ALPHA


def tail_record(pnls):
    """tail_symmetry_test + verdict replicando la logica del check 5."""
    n = len(pnls)
    ts = ad.tail_symmetry_test(pnls)
    if ts is None:
        return {"n": n, "nl": None, "nr": None, "p": None,
                "verdict": "SKIP" if n < 200 else "SKIP(d<=0)",
                "nr_minus_nl": None}
    nl, nr, p = ts
    verdict = "FAIL" if (p <= ALPHA and nl < nr) else "PASS"
    return {"n": n, "nl": nl, "nr": nr, "p": p, "verdict": verdict,
            "nr_minus_nl": nr - nl}


def noov_record(booked):
    """Registro del check 5 CORREGIDO: el test corre sobre los trades NO
    solapados (unidades independientes; mismas de las que sale el Sharpe
    honesto), no sobre la lista reportada."""
    n_rep = len(booked)
    noov = [p for p in ad.nonoverlapping_pnls(booked, ad.HOLD) if p != 0]
    n_noov = len(noov)
    ts = ad.tail_symmetry_test(noov)
    if ts is None:
        return {"n_rep": n_rep, "n_noov": n_noov, "nl": None, "nr": None,
                "p": None,
                "verdict": "PASS" if n_rep < 200 else "SKIP",
                "nr_minus_nl": None}
    nl, nr, p = ts
    verdict = "FAIL" if (p <= ALPHA and nl < nr) else "PASS"
    return {"n_rep": n_rep, "n_noov": n_noov, "nl": nl, "nr": nr, "p": p,
            "verdict": verdict, "nr_minus_nl": nr - nl}


def run_sniper(seed, p0):
    px = ad.build_market(seed=seed, p0=p0)
    sig = aea.sig_spread_sniper(px)
    # Pipeline EXACTO del 4h original: hs constante 0.004 en ambas escalas
    # (el repo actual ya escala hs; se pinta explicitamente por especificacion).
    booked, thr = aea.build_honest_client(px, sig, ad.HALF_SPREAD)   # pipeline exacto del 4h
    pnls = [p for _t, _pos, p in booked]
    rec = {"seed": seed, "p0": p0, "thr": round(thr, 4)}
    rec["abs"] = tail_record(pnls)
    rec["abs_noov"] = noov_record(booked)
    # Variante relative-pnl (tarea 4): entry_lag=1, hold=HOLD.
    rel_booked = []
    for t, pos, _p in booked:
        e_t = t + 1
        x_t = e_t + ad.HOLD
        rel_booked.append((t, pos,
                           pos * (px[x_t] / px[e_t] - 1.0) - 2.0 * HALF_SPREAD_REL))
    rec["rel"] = tail_record([b[2] for b in rel_booked])
    rec["rel_noov"] = noov_record(rel_booked)
    return rec


def run_control(name, seed_mkt, p0):
    px = ad.build_market(seed=seed_mkt, p0=p0)
    if name == "momentum_chaser":
        sig = aea.sig_momentum_chaser(px)           # sin seed (determinista)
    else:
        sig = aea.sig_ruido_puro(px, 203)           # seed de señal del 4h
    booked, thr = aea.build_honest_client(px, sig, ad.HALF_SPREAD)
    pnls = [p for _t, _pos, p in booked]
    rec = {"name": name, "seed": seed_mkt, "p0": p0, "thr": round(thr, 4)}
    rec.update(tail_record(pnls))
    rec["noov"] = noov_record(booked)
    return rec


def summarize(rows, key):
    """rows: lista de recs; key: 'abs'/'rel' o None (campos de primer nivel)."""
    def row_of(r):
        return r[key] if key is not None else r
    tested = [row_of(r) for r in rows
              if row_of(r)["verdict"] in ("FAIL", "PASS")]
    n_test = len(tested)
    n_fail = sum(1 for t in tested if t["verdict"] == "FAIL")
    diffs = [t["nr_minus_nl"] for t in tested]
    return {
        "n_rows": len(rows),
        "n_tested": n_test,
        "n_fail": n_fail,
        "fail_rate": (n_fail / n_test) if n_test else None,
        "median_nr_minus_nl": (statistics.median(diffs) if diffs else None),
        "mean_nl": (sum(t["nl"] for t in tested) / n_test) if n_test else None,
        "mean_nr": (sum(t["nr"] for t in tested) / n_test) if n_test else None,
    }


def main():
    t0 = time.time()
    print("seeds mercado: %d (7..%d), escalas %s, half_spread_rel=%.3f, "
          "alpha=%.2f" % (len(SEEDS), SEEDS[-1], P0S, HALF_SPREAD_REL, ALPHA),
          flush=True)

    sniper = []
    for seed in SEEDS:
        for p0 in P0S:
            r = run_sniper(seed, p0)
            sniper.append(r)
            a, b = r["abs"], r["rel"]
            print("sniper seed=%3d p0=%-6g thr=%.2f | ABS n=%5d nl=%3d nr=%3d "
                  "p=%.4f %s | REL n=%5d nl=%3d nr=%3d p=%.4f %s"
                  % (seed, p0, r["thr"], a["n"], a["nl"], a["nr"], a["p"],
                     a["verdict"], b["n"], b["nl"], b["nr"], b["p"], b["verdict"]),
                  flush=True)

    ctrl = []
    for name in ("momentum_chaser", "ruido_puro"):
        for seed in SEEDS_CTRL:
            c = run_control(name, seed, 0.5)
            ctrl.append(c)
            print("CTRL %-16s seed=%3d p0=%-6g thr=%.2f n=%5d nl=%3d nr=%3d "
                  "p=%.4f %s"
                  % (name, seed, c["p0"], c["thr"], c["n"], c["nl"], c["nr"],
                     c["p"], c["verdict"]), flush=True)

    # ── resúmenes ────────────────────────────────────────────────────
    per_scale = {p0: summarize([r for r in sniper if r["p0"] == p0], "abs")
                 for p0 in P0S}
    per_scale_noov = {p0: summarize([r for r in sniper if r["p0"] == p0],
                                    "abs_noov")
                      for p0 in P0S}
    per_scale_rel = {p0: summarize([r for r in sniper if r["p0"] == p0], "rel")
                     for p0 in P0S}
    per_scale_rel_noov = {p0: summarize([r for r in sniper if r["p0"] == p0],
                                        "rel_noov")
                          for p0 in P0S}

    # Consistencia del signo de (nr-nl) entre escalas (debe ser idéntica:
    # el test es rank-based e invariante de escala).
    mism = 0
    for i in range(len(SEEDS)):
        r05 = sniper[2 * i]["abs"]["nr_minus_nl"]
        r100 = sniper[2 * i + 1]["abs"]["nr_minus_nl"]
        if r05 is None or r100 is None or r05 != r100:
            mism += 1
    # y el trío (nl, nr, p) exacto
    triples_diff = 0
    for i in range(len(SEEDS)):
        a = sniper[2 * i]["abs"]
        b = sniper[2 * i + 1]["abs"]
        if (a["nl"], a["nr"], round(a["p"], 12)) != \
           (b["nl"], b["nr"], round(b["p"], 12)):
            triples_diff += 1

    ctrl_sum = {name: summarize([c for c in ctrl if c["name"] == name], None)
                for name in ("momentum_chaser", "ruido_puro")}
    ctrl_noov_sum = {name: summarize([c["noov"] for c in ctrl
                                      if c["name"] == name], None)
                     for name in ("momentum_chaser", "ruido_puro")}

    print("\n=== RESUMEN ===", flush=True)
    for p0 in P0S:
        s = per_scale[p0]
        sr = per_scale_rel[p0]
        sn = per_scale_noov[p0]
        srn = per_scale_rel_noov[p0]
        print("sniper ABS p0=%-6g: testados=%d FAIL=%d rate=%.1f%% med(nr-nl)=%s "
              "media nl=%.1f nr=%.1f"
              % (p0, s["n_tested"], s["n_fail"], 100 * s["fail_rate"],
                 s["median_nr_minus_nl"], s["mean_nl"], s["mean_nr"]))
        print("sniper NOOV p0=%-6g: testados=%d FAIL=%d rate=%.1f%% med(nr-nl)=%s "
              "media nl=%.1f nr=%.1f   <- detector CORREGIDO"
              % (p0, sn["n_tested"], sn["n_fail"], 100 * sn["fail_rate"],
                 sn["median_nr_minus_nl"], sn["mean_nl"], sn["mean_nr"]))
        print("sniper REL  p0=%-6g: testados=%d FAIL=%d rate=%.1f%% med(nr-nl)=%s "
              "media nl=%.1f nr=%.1f"
              % (p0, sr["n_tested"], sr["n_fail"], 100 * sr["fail_rate"],
                 sr["median_nr_minus_nl"], sr["mean_nl"], sr["mean_nr"]))
        print("sniper RNVL p0=%-6g: testados=%d FAIL=%d rate=%.1f%% med(nr-nl)=%s "
              "media nl=%.1f nr=%.1f   <- rel con detector CORREGIDO"
              % (p0, srn["n_tested"], srn["n_fail"], 100 * srn["fail_rate"],
                 srn["median_nr_minus_nl"], srn["mean_nl"], srn["mean_nr"]))
    print("consistencia entre escalas: seeds con (nr-nl) distinto = %d/%d; "
          "tríos (nl,nr,p) distintos = %d/%d"
          % (mism, len(SEEDS), triples_diff, len(SEEDS)))
    for name in ("momentum_chaser", "ruido_puro"):
        s = ctrl_sum[name]
        print("CTRL %-16s: testados=%d FAIL=%d rate=%.1f%% med(nr-nl)=%s "
              "media nl=%.1f nr=%.1f"
              % (name, s["n_tested"], s["n_fail"], 100 * (s["fail_rate"] or 0),
                 s["median_nr_minus_nl"], s["mean_nl"], s["mean_nr"]))
    print("elapsed %.1fs" % (time.time() - t0))

    out = {
        "params": {
            "seeds": SEEDS, "p0s": P0S, "seeds_ctrl": SEEDS_CTRL,
            "half_spread": ad.HALF_SPREAD,
            "half_spread_rel": HALF_SPREAD_REL, "alpha": ALPHA,
            "pipeline": "aea.build_honest_client (walk-forward 1a mitad, "
                        "entry_lag=1, price_book=True, half_spread=HALF_SPREAD, "
                        "reporta 2a mitad)",
            "n_days": ad.N_DAYS, "sigma": ad.SIGMA, "hold": ad.HOLD,
            "elapsed_s": round(time.time() - t0, 1),
        },
        "sniper": sniper,
        "controles": ctrl,
        "resumen": {
            "sniper_abs_por_escala": per_scale,
            "sniper_noov_por_escala": per_scale_noov,
            "sniper_rel_por_escala": per_scale_rel,
            "sniper_rel_noov_por_escala": per_scale_rel_noov,
            "consistencia_escalas": {
                "seeds_nr_minus_nl_diferente": mism,
                "triples_nl_nr_p_diferentes": triples_diff,
                "n_seeds": len(SEEDS),
            },
            "controles_p0_05": ctrl_sum,
            "controles_p0_05_noov": ctrl_noov_sum,
        },
    }
    with open(os.path.join(RESULTS, "estudio_limitacion_supervivencia.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("written: %s" % os.path.join(RESULTS, "estudio_limitacion_supervivencia.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
