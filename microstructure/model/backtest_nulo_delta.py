#!/usr/bin/env python3
"""TEST NULO del DELTA del filtro de toxicidad — datos REALES, h=300s.

Diseño (nota interna del vault, 2026-10-22,
sección "Regla de decisión FIJADA 23/10", escrita ANTES de correr — no se modifica):

  - Horizonte único a priori: h=300s (1 fill = 1 muestra)
  - Umbral fijo a priori: 0.7
  - Estadístico: DELTA = P&L(filtrado@0.7) − P&L(sin filtro), convención MAKER
    (pnl por fill = −(P&L del agresor), USD a $1/share nominal).

Nulos:
  A) Barato, SIN reentrenar: filtro ALEATORIO con la MISMA retención por fold que
     el @0.7 real (retención = n_filtrado/n_total del fold), 1.000 draws con seed
     fija → distribución del delta pooled → percentil del delta observado.
  B) Bootstrap por bloques de MERCADO (token_id) dentro del test set de cada fold
     (re-muestreo de mercados con reemplazo, 1.000 draws) → IC (2.5%/97.5%) del
     delta por fold y del pooled.
  C) Control: 50 reentrenos con labels PERMUTADAS dentro de cada mercado
     (permutar el target dentro de cada token_id, en TRAIN) → mismo walk-forward
     → delta esperado bajo la nula de señal. Si el delta con labels permutadas
     no es ~0, el pipeline fabrica señal: se reporta TAL CUAL.
  Secundario: umbral re-elegido por draw del nulo A (grid 0.5–0.85, step 0.05).
     Reporte secundario; NO reemplaza la regla primaria.

Regla de decisión (FIJADA antes de correr): rechazo al 5% si delta pooled ≥ p95
del nulo A Y delta > 0 en ≥4 de 5 folds. Se reporta lo que salga, aunque el nulo
NO rechace (honestidad primero).

Uso:  LGB_DEVICE=gpu ./venv-ml/bin/python backtest_nulo_delta.py
Salida: dataset/nulo_delta_h300.json  (+ JSON parcial intermedio si aborta en C)
"""

import csv
import json
import os
import time

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dataset", "ml_features.csv")
OUT = os.path.join(HERE, "dataset", "nulo_delta_h300.json")
OUT_PARTIAL = OUT + ".parcial.json"

# Mismo pipeline exacto que backtest_walkforward.py (features excluyen
# fill_size/size_rel_depth — fuga maker; params, embargo y bloques idénticos).
FEATURES = ["spread_rel", "imbalance", "depth_total", "n_total",
            "momentum_60s", "momentum_300s", "ofi", "microprice_bias",
            "side_buy", "horizonte_s"]
EMBARGO_S = 300
K_BLOQUES = 6
UMBRAL_FIJO = 0.7
HORIZONTE = 300
N_DRAWS = 1000
N_PERM = 50
SEED = 42
GRILLA_SEC = [round(0.5 + 0.05 * i, 2) for i in range(8)]   # 0.5 .. 0.85
GRILLA_REF = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]  # ref: mejor umbral c/fold

# Decisión del dueño (23/10): GPU para LightGBM. No cambiar.
DEVICE = os.environ.get("LGB_DEVICE", "gpu")

DECISION_RULE = ("Regla de decisión FIJADA 23/10 (antes de correr): horizonte único a priori h=300s; "
                 "umbral fijo a priori 0.7; estadístico = DELTA (filtrado@0.7 − sin filtro). "
                 "Rechazo al 5% si: delta pooled ≥ p95 del nulo A Y delta > 0 en ≥4 de 5 folds.")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load():
    """Filas etiquetadas h=300, convención MAKER (idéntico a backtest_walkforward.py)."""
    rows = []
    with open(DATA) as f:
        for r in csv.DictReader(f):
            if r["target"] == "" or r["mid_futuro"] == "":
                continue
            if int(float(r["horizonte_s"])) != HORIZONTE:
                continue
            mf, fp = float(r["mid_futuro"]), float(r["fill_price"])
            side = 1.0 if float(r["side_buy"]) > 0.5 else -1.0
            pnl_agresor = side * (mf - fp) * 100.0  # centavos/share
            rows.append({
                "ts": float(r["ts_epoch"]),
                "token": r["token_id"],
                "pnl": -pnl_agresor,  # MAKER: P&L del LP = −(P&L del agresor)
                "x": [float(r[k]) if r[k] not in ("", None) else 0.0 for k in FEATURES],
                "y": 1 if float(r["target"]) > 0.5 else 0,
            })
    rows.sort(key=lambda r: r["ts"])
    return rows


def build_folds(rows):
    """Mismos splits temporales + embargo que backtest_walkforward.py (K=6 bloques, folds 1..5)."""
    ts_all = [r["ts"] for r in rows]
    bordes = np.linspace(ts_all[0], ts_all[-1], K_BLOQUES + 1)
    bloques = []
    for k in range(K_BLOQUES):
        if k == K_BLOQUES - 1:
            bloques.append([r for r in rows if r["ts"] >= bordes[k]])
        else:
            bloques.append([r for r in rows if bordes[k] <= r["ts"] < bordes[k + 1]])
    folds = []
    for k in range(1, K_BLOQUES):  # fold k: test = bloque k+1
        test = bloques[k]
        if not test:
            continue
        test_start = test[0]["ts"]
        train = [r for r in rows if r["ts"] < test_start - EMBARGO_S]
        if len(train) < 5000 or len(test) < 500:
            log(f"fold {k}: datos insuficientes, se omite")
            continue
        folds.append({"fold": k, "train": train, "test": test})
    return folds


def train_model(Xtr, ytr):
    """Mismos params que backtest_walkforward.py (seed 42, 200 rondas)."""
    n1 = int(ytr.sum())
    n0 = len(ytr) - n1
    return lgb.train(
        {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
         "num_leaves": 31, "min_child_samples": 30,
         "scale_pos_weight": n0 / max(n1, 1), "verbose": -1, "seed": 42,
         "device_type": DEVICE},
        lgb.Dataset(Xtr, label=ytr), num_boost_round=200)


def main():
    t0 = time.time()
    log(f"test nulo del delta | h={HORIZONTE} | umbral fijo {UMBRAL_FIJO} | "
        f"device={DEVICE} | nulos A/B {N_DRAWS} draws seed {SEED} | control C {N_PERM} reentrenos")

    rows = load()
    log(f"filas h={HORIZONTE} con target y mid_futuro: {len(rows)} | "
        f"{len(set(r['token'] for r in rows))} mercados (token_id)")

    folds = build_folds(rows)
    for f in folds:
        log(f"  fold {f['fold']}: train {len(f['train'])} | test {len(f['test'])}")

    # ---------- RUN OBSERVADO (mismo pipeline del walk-forward real) ----------
    obs = []
    for f in folds:
        Xtr = np.array([r["x"] for r in f["train"]])
        ytr = np.array([r["y"] for r in f["train"]])
        Xte = np.array([r["x"] for r in f["test"]])
        yte = np.array([r["y"] for r in f["test"]])
        pnl = np.array([r["pnl"] for r in f["test"]])          # centavos/share
        tokens_te = np.array([r["token"] for r in f["test"]])
        tokens_tr = np.array([r["token"] for r in f["train"]])

        t1 = time.time()
        model = train_model(Xtr, ytr)
        proba = model.predict(Xte)
        auc = float(roc_auc_score(yte, proba))
        log(f"  [obs] fold {f['fold']}: AUC={auc:.4f} ({time.time() - t1:.1f}s)")

        mask = proba < UMBRAL_FIJO
        pnl_s = float(pnl.sum()) / 100.0
        pnl_f = float(pnl[mask].sum()) / 100.0
        f.update({"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "pnl": pnl, "proba": proba,
                  "tokens_te": tokens_te, "tokens_tr": tokens_tr,
                  "n_total": len(pnl), "n_filt": int(mask.sum()),
                  "pnl_s_cents": float(pnl.sum())})
        # counts por umbral (nulo A secundario + referencia mejor-umbral)
        f["n_filt_grid"] = {t: int((proba < t).sum()) for t in GRILLA_SEC}
        ref = {t: float(pnl[proba < t].sum()) / 100.0 for t in GRILLA_REF}
        # viabilidad como en el script base: n >= 5% del test
        viable = {t: v for t, v in ref.items()
                  if int((proba < t).sum()) >= 0.05 * len(pnl)}
        mejor_t = max(viable, key=lambda t: viable[t]) if viable else None
        obs.append({
            "fold": f["fold"], "auc_oos": round(auc, 4),
            "n_total": f["n_total"], "n_filtrado": f["n_filt"],
            "sin_filtro_usd": round(pnl_s, 2), "filtrado_0p7_usd": round(pnl_f, 2),
            "delta_usd": round(pnl_f - pnl_s, 2),
            "mejor_umbral_ref": mejor_t,
            "mejor_umbral_pnl_usd": round(viable[mejor_t], 2) if mejor_t else None,
        })
        log(f"  [obs] fold {f['fold']}: sin filtro {pnl_s:.2f}$ | @0.7 {pnl_f:.2f}$ "
            f"({f['n_filt']}/{f['n_total']}) | delta {pnl_f - pnl_s:+.2f}$")

    delta_obs_por_fold = [o["delta_usd"] for o in obs]
    delta_obs_pooled = round(sum(delta_obs_por_fold), 2)
    n_folds_pos = sum(1 for d in delta_obs_por_fold if d > 0)
    delta_obs_mejor_t = round(sum((o["mejor_umbral_pnl_usd"] or 0) - o["sin_filtro_usd"]
                                  for o in obs), 2)
    log(f"OBSERVADO: delta pooled {delta_obs_pooled:+.2f}$ | por fold {delta_obs_por_fold} | "
        f"delta>0 en {n_folds_pos}/{len(obs)} folds | (ref mejor-umbral c/fold: {delta_obs_mejor_t:+.2f}$)")

    # ---------- NULO A: filtro aleatorio, MISMA retención por fold ----------
    rngA = np.random.default_rng(SEED)
    deltas_A = np.empty(N_DRAWS)
    deltas_A_resel = np.empty(N_DRAWS)  # secundario: umbral re-elegido por draw
    t1 = time.time()
    for d in range(N_DRAWS):
        tot, tot_r = 0.0, 0.0
        t_d = GRILLA_SEC[int(rngA.integers(len(GRILLA_SEC)))]
        for f in folds:
            pnl = f["pnl"]
            k = f["n_filt"]
            tot += float(pnl[rngA.choice(len(pnl), size=k, replace=False)].sum()) - f["pnl_s_cents"]
            k_r = f["n_filt_grid"][t_d]
            if k_r > 0:
                tot_r += float(pnl[rngA.choice(len(pnl), size=k_r, replace=False)].sum()) - f["pnl_s_cents"]
        deltas_A[d] = tot / 100.0
        deltas_A_resel[d] = tot_r / 100.0
    log(f"NULO A: {N_DRAWS} draws en {time.time() - t1:.0f}s | "
        f"p50 {np.percentile(deltas_A, 50):+.2f}$ | p95 {np.percentile(deltas_A, 95):+.2f}$ | "
        f"p99 {np.percentile(deltas_A, 99):+.2f}$ | mean {deltas_A.mean():+.2f}$ | "
        f"std {deltas_A.std():.2f}$")

    n_below = int(np.sum(deltas_A < delta_obs_pooled))
    n_eq = int(np.sum(deltas_A == delta_obs_pooled))
    pctil_obs = 100.0 * (n_below + 0.5 * n_eq) / N_DRAWS
    log(f"NULO A: percentil del delta observado ({delta_obs_pooled:+.2f}$) = {pctil_obs:.2f}% "
        f"(p95 = {np.percentile(deltas_A, 95):+.2f}$)")

    # ---------- NULO B: bootstrap por bloques de mercado (token_id) ----------
    rngB = np.random.default_rng(SEED + 1)
    deltaB_por_fold, deltaB_pooled = {}, np.zeros(N_DRAWS)
    t1 = time.time()
    for f in folds:
        order = np.argsort(f["tokens_te"], kind="stable")
        _, starts, counts = np.unique(f["tokens_te"][order], return_index=True, return_counts=True)
        ends = starts + counts
        pnl_o = f["pnl"][order]
        mkt_pnl = np.add.reduceat(pnl_o, starts)
        mkt_filt = np.add.reduceat(pnl_o * (f["proba"] < UMBRAL_FIJO), starts)
        M = len(starts)
        idx = rngB.integers(0, M, size=(N_DRAWS, M))
        db = (mkt_filt[idx].sum(axis=1) - mkt_pnl[idx].sum(axis=1)) / 100.0
        deltaB_por_fold[f["fold"]] = {
            "n_mercados": M,
            "ic_low_2_5": round(float(np.percentile(db, 2.5)), 2),
            "ic_high_97_5": round(float(np.percentile(db, 97.5)), 2),
            "mean": round(float(db.mean()), 2),
            "delta_obs_usd": None,  # se llena abajo
        }
        deltaB_pooled += db
    for f, o in zip(folds, obs):
        deltaB_por_fold[f["fold"]]["delta_obs_usd"] = o["delta_usd"]
    log(f"NULO B: {N_DRAWS} draws en {time.time() - t1:.0f}s")
    for f in folds:
        s = deltaB_por_fold[f["fold"]]
        log(f"  [B] fold {f['fold']}: IC 2.5/97.5 [{s['ic_low_2_5']:+.2f}, {s['ic_high_97_5']:+.2f}]$ "
            f"(obs {s['delta_obs_usd']:+.2f}$)")
    icB_pooled = [round(float(np.percentile(deltaB_pooled, 2.5)), 2),
                  round(float(np.percentile(deltaB_pooled, 97.5)), 2)]
    log(f"  [B] pooled: IC 2.5/97.5 [{icB_pooled[0]:+.2f}, {icB_pooled[1]:+.2f}]$ "
        f"(obs {delta_obs_pooled:+.2f}$) → excluye 0: {icB_pooled[0] > 0}")

    # ---------- JSON intermedio (por si aborta en C) ----------
    parcial = {
        "fecha": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "estado": "parcial — control C en curso",
        "decision_rule": DECISION_RULE,
        "observado": {"pooled_delta_usd": delta_obs_pooled,
                      "por_fold": obs},
        "nulo_a": {"n_draws": N_DRAWS, "seed": SEED},
    }
    with open(OUT_PARTIAL, "w") as fp:
        json.dump(parcial, fp, indent=1)

    # ---------- CONTROL C: 50 reentrenos, labels PERMUTADAS dentro de cada mercado ----------
    for f in folds:
        order_tr = np.argsort(f["tokens_tr"], kind="stable")
        _, starts_tr, counts_tr = np.unique(f["tokens_tr"][order_tr],
                                            return_index=True, return_counts=True)
        f["perm_starts"] = starts_tr
        f["perm_ends"] = starts_tr + counts_tr
        f["perm_gidx"] = order_tr  # índice original de cada posición ordenada

    rngC = np.random.default_rng(SEED + 2)
    deltas_C = np.empty(N_PERM)
    t1 = time.time()
    for r in range(N_PERM):
        tot = 0.0
        for f in folds:
            ytr = f["ytr"]
            y_perm = ytr.copy()
            gidx = f["perm_gidx"]
            for s, e in zip(f["perm_starts"], f["perm_ends"]):
                g = gidx[s:e]
                y_perm[g] = ytr[g][rngC.permutation(e - s)]
            model = train_model(f["Xtr"], y_perm)
            p = model.predict(f["Xte"])
            mask = p < UMBRAL_FIJO
            tot += float(f["pnl"][mask].sum()) - f["pnl_s_cents"]
        deltas_C[r] = tot / 100.0
        if (r + 1) % 10 == 0 or r == 0:
            log(f"  [C] reentrenamiento {r + 1}/{N_PERM} | delta {deltas_C[r]:+.2f}$ "
                f"(media hasta acá {deltas_C[:r + 1].mean():+.2f}$) | {time.time() - t1:.0f}s")
    log(f"CONTROL C: {N_PERM} reentrenos en {time.time() - t1:.0f}s | "
        f"mean {deltas_C.mean():+.2f}$ | std {deltas_C.std():+.2f}$ | "
        f"min {deltas_C.min():+.2f}$ | max {deltas_C.max():+.2f}$")

    # ---------- VEREDICTO (regla fija, se reporta lo que salga) ----------
    p95_A = float(np.percentile(deltas_A, 95))
    cond1 = delta_obs_pooled >= p95_A
    cond2 = n_folds_pos >= 4
    rechazo = bool(cond1 and cond2)
    c_cero = bool(deltas_C.min() <= 0 <= deltas_C.max())
    log(f"REGLA: delta pooled {delta_obs_pooled:+.2f}$ >= p95 nulo A {p95_A:+.2f}$ → {cond1} | "
        f"delta>0 en {n_folds_pos}/{len(obs)} folds (>=4) → {cond2} | RECHAZO: {rechazo}")
    if not c_cero:
        log("CONTROL C: 0 FUERA del rango [min,max] de los deltas permutados → "
            "el pipeline fabrica señal bajo la nula (se reporta tal cual).")

    veredicto = (
        f"{'RECHAZADO' if rechazo else 'NO RECHAZADO'} al 5%: delta pooled {delta_obs_pooled:+.2f}$ "
        f"vs nulo A p95 {p95_A:+.2f}$ (percentil observado {pctil_obs:.1f}%) | "
        f"delta>0 en {n_folds_pos}/{len(obs)} folds | IC nulo B pooled "
        f"[{icB_pooled[0]:+.2f}, {icB_pooled[1]:+.2f}]$ | control C "
        f"{deltas_C.mean():+.2f}±{deltas_C.std():.2f}$ "
        f"(min {deltas_C.min():+.2f}, max {deltas_C.max():+.2f}). "
        + ("El delta del filtro @0.7 es significativo frente a la nula de filtro aleatorio."
           if rechazo else
           "El delta del filtro @0.7 NO es estadísticamente significativo frente al nulo A "
           "(se reporta tal cual: no se rechaza la nula)."))
    if not c_cero:
        veredicto += " ADVERTENCIA CONTROL C: delta con labels permutadas no pasa por 0 → " \
                     "posible fabricación de señal por el pipeline."

    # ---------- SALIDA ----------
    resumen = {
        "fecha": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "dataset": "bot/dataset/ml_features.csv (datos reales)",
        "horizonte_s": HORIZONTE,
        "umbral_fijo": UMBRAL_FIJO,
        "convencion": "MAKER: pnl por fill = -(P&L del agresor del tape); el filtro retiene p<0.7",
        "decision_rule": DECISION_RULE,
        "filas": len(rows),
        "folds": [f["fold"] for f in folds],
        "observado": {
            "pooled_delta_usd": delta_obs_pooled,
            "deltas_por_fold_usd": delta_obs_por_fold,
            "n_folds_delta_pos": n_folds_pos,
            "delta_mejor_umbral_por_fold_ref_usd": delta_obs_mejor_t,
            "por_fold": obs,
        },
        "nulo_a": {
            "descripcion": "filtro aleatorio con la MISMA retención por fold que el @0.7 real "
                           "(n_filtrado/n_total), subconjunto sin reemplazo",
            "n_draws": N_DRAWS, "seed": SEED,
            "mean_usd": round(float(deltas_A.mean()), 2),
            "std_usd": round(float(deltas_A.std()), 2),
            "p05_usd": round(float(np.percentile(deltas_A, 5)), 2),
            "p50_usd": round(float(np.percentile(deltas_A, 50)), 2),
            "p95_usd": round(p95_A, 2),
            "p99_usd": round(float(np.percentile(deltas_A, 99)), 2),
            "max_usd": round(float(deltas_A.max()), 2),
            "delta_observado_usd": delta_obs_pooled,
            "percentil_observado_pct": round(pctil_obs, 2),
            "condicion_1_obs_ge_p95": cond1,
        },
        "nulo_b": {
            "descripcion": "bootstrap por bloques de mercado (token_id) dentro del test de cada fold, "
                           "re-muestreo de mercados con reemplazo",
            "n_draws": N_DRAWS, "seed": SEED + 1,
            "por_fold": deltaB_por_fold,
            "pooled": {"ic_low_2_5_usd": icB_pooled[0], "ic_high_97_5_usd": icB_pooled[1],
                       "mean_usd": round(float(deltaB_pooled.mean()), 2),
                       "delta_observado_usd": delta_obs_pooled,
                       "ic_excluye_cero": bool(icB_pooled[0] > 0)},
        },
        "control_c": {
            "descripcion": f"{N_PERM} reentrenos del walk-forward con labels PERMUTADAS dentro de cada "
                           "mercado (token_id) en TRAIN (misma retención @0.7 en test)",
            "n_reentrenos": N_PERM, "seed": SEED + 2,
            "device": DEVICE,
            "mean_usd": round(float(deltas_C.mean()), 2),
            "std_usd": round(float(deltas_C.std()), 2),
            "min_usd": round(float(deltas_C.min()), 2),
            "max_usd": round(float(deltas_C.max()), 2),
            "cero_dentro_minmax": c_cero,
            "nota": ("delta permutado ~0: el pipeline no fabrica señal"
                     if c_cero else
                     "ALERTA: 0 fuera de [min,max] de los deltas permutados → el pipeline fabrica "
                     "señal bajo la nula de labels (reportado tal cual)"),
        },
        "secundario_umbral_reelegido": {
            "descripcion": "nulo A variante: por cada draw se re-elige un umbral uniforme de la grid "
                           "0.5-0.85 (step 0.05); retención por fold = fracción real p<t en ese umbral. "
                           "Reporte secundario: NO reemplaza la regla primaria (umbral fijo 0.7 a priori).",
            "grid": GRILLA_SEC,
            "n_draws": N_DRAWS,
            "mean_usd": round(float(deltas_A_resel.mean()), 2),
            "std_usd": round(float(deltas_A_resel.std()), 2),
            "p95_usd": round(float(np.percentile(deltas_A_resel, 95)), 2),
            "p99_usd": round(float(np.percentile(deltas_A_resel, 99)), 2),
            "percentil_delta_observado_fijo07_pct": round(
                100.0 * (np.sum(deltas_A_resel < delta_obs_pooled) +
                         0.5 * np.sum(deltas_A_resel == delta_obs_pooled)) / N_DRAWS, 2),
            "delta_observado_mejor_umbral_por_fold_ref_usd": delta_obs_mejor_t,
        },
        "regla_aplicada": {
            "delta_pooled_ge_p95_nulo_a": cond1,
            "delta_pos_en_al_4_de_5_folds": cond2,
            "rechazo_5pct": rechazo,
        },
        "veredicto": veredicto,
        "duracion_s": round(time.time() - t0, 1),
    }
    with open(OUT, "w") as fp:
        json.dump(resumen, fp, indent=1)
    try:
        os.remove(OUT_PARTIAL)
    except OSError:
        pass
    log(f"resultados -> {OUT}")

    log("=" * 70)
    log(f"OBSERVADO h={HORIZONTE}: sin filtro y @0.7 por fold: "
        + " | ".join(f"fold {o['fold']}: {o['sin_filtro_usd']:+.2f} → {o['filtrado_0p7_usd']:+.2f} "
                     f"(Δ{o['delta_usd']:+.2f}$)" for o in obs))
    log(f"DELTA POOLED OBSERVADO: {delta_obs_pooled:+.2f}$ (delta>0 en {n_folds_pos}/{len(obs)} folds)")
    log(f"NULO A: p50 {np.percentile(deltas_A, 50):+.2f}$ | p95 {p95_A:+.2f}$ | "
        f"p99 {np.percentile(deltas_A, 99):+.2f}$ | percentil obs {pctil_obs:.2f}%")
    log(f"NULO B pooled IC: [{icB_pooled[0]:+.2f}, {icB_pooled[1]:+.2f}]$ "
        f"(excluye 0: {icB_pooled[0] > 0})")
    log(f"CONTROL C (labels permutadas): mean {deltas_C.mean():+.2f}$ | std {deltas_C.std():.2f}$ | "
        f"min {deltas_C.min():+.2f}$ | max {deltas_C.max():+.2f}$ | 0∈[min,max]: {c_cero}")
    log(f"SECUNDARIO (umbral re-elegido por draw): p95 {np.percentile(deltas_A_resel, 95):+.2f}$ | "
        f"percentil del delta obs fijo-0.7 "
        f"{100.0 * (np.sum(deltas_A_resel < delta_obs_pooled) + 0.5 * np.sum(deltas_A_resel == delta_obs_pooled)) / N_DRAWS:.2f}%")
    log(f"VEREDICTO: {veredicto}")


if __name__ == "__main__":
    main()
