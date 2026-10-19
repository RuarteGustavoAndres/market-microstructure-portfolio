"""Validación HONESTA del filtro de toxicidad — PASO 2: folds por MERCADO.

Fixes de fugas de validación sobre el walkforward temporal (PASO 1):
  1. Folds por MERCADO (GroupKFold por token_id, K=5): el modelo se evalúa
     SOLO sobre mercados que jamás vio en train. El walkforward temporal
     comparte mercados entre train y test -> el modelo puede memorizar la
     base-rate y microestructura específica del mercado (fuga de entidad).
  2. Estandarización honesta: media/sd fit SOLO con filas del train; los
     mercados de test (no vistos) se normalizan con stats globales de train
     (lo único computable en producción para un mercado nuevo).
  3. PR-AUC (average precision): con ~39% de clase positiva (favorable al
     agresor) y desbalance fuerte, el ROC-AUC sobreestimaba; el PR-AUC es
     la métrica de titular con el baseline de prevalencia explícito.
  4. Umbral por CUANTIL (sin mirar el futuro): en vez de umbral fijo 0.7 ni
     grilla por fold (que PASO 1 mostró inestable 0.65->0.55), el umbral de
     cada fold es el cuantil-q de las PREDICCIONES DE TRAIN: el bot retiene
     el flujo cuyo score cae por debajo (p<thr, convención MAKER — flujo
     benigno para el LP). Es una POLÍTICA de retención a priori, no un
     hyperparámetro ajustado contra el test.
  5. P&L MAKER exacto: pnl = -side*(mid_futuro - fill_price), mid_futuro
     real del libro (igual que PASO 1, veredicto arbitraje 19-oct).

Uso: ./venv-ml/bin/python backtest_folds_mercado.py
Salida: dataset/folds_mercado_toxicidad.json
"""

import csv
import json
import os
import time
from collections import defaultdict

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dataset", "ml_features.csv")
OUT = os.path.join(HERE, "dataset", "folds_mercado_toxicidad.json")

# fill_size y size_rel_depth EXCLUÍDOS (fuga para maker: el trade entrante no
# se conoce al cotizar — revisión Claude 19-oct). Columnas siguen en el CSV.
FEATURES = ["spread_rel", "imbalance", "depth_total", "n_total",
            "momentum_60s", "momentum_300s", "ofi", "microprice_bias",
            "side_buy", "horizonte_s"]
K_FOLDS = 5
CUANTIL = 0.5               # umbral = cuantil 0.5 de las predicciones de train
CUANTILES_SENS = [0.3, 0.5, 0.7]  # variante de sensibilidad (misma política)
UMBRAL_FIJO_REF = 0.7       # referencia de continuidad con PASO 1
SEED = 42


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load():
    """Filas etiquetadas, convención MAKER: pnl = -side*(mid_futuro-price)."""
    rows = []
    with open(DATA) as f:
        for r in csv.DictReader(f):
            if r["target"] == "" or r["mid_futuro"] == "":
                continue
            mf, fp = float(r["mid_futuro"]), float(r["fill_price"])
            side = 1.0 if float(r["side_buy"]) > 0.5 else -1.0
            pnl_agresor = side * (mf - fp) * 100.0  # centavos/share
            rows.append({
                "ts": float(r["ts_epoch"]),
                "token": r["token_id"],
                "h": int(float(r["horizonte_s"])),
                "pnl": -pnl_agresor,  # convención MAKER
                "x": [float(r[k]) if r[k] not in ("", None) else 0.0
                      for k in FEATURES],
                "y": 1 if float(r["target"]) > 0.5 else 0,
            })
    rows.sort(key=lambda r: r["ts"])
    return rows


def pnl_stats(sub, mask=None):
    """P&L MAKER (USD a $1/share nominal), ops y hit-rate de un subconjunto."""
    if mask is not None:
        sub = [r for r, keep in zip(sub, mask) if keep]
    if not sub:
        return {"n": 0, "pnl": 0.0, "pnl_op": 0.0, "hit": 0.0}
    pnls = [r["pnl"] for r in sub]
    return {
        "n": len(sub),
        "pnl": round(sum(pnls) / 100.0, 2),       # centavos -> USD/share
        "pnl_op": round(float(np.mean(pnls)) / 100.0, 4),
        "hit": round(100.0 * sum(1 for p in pnls if p > 0) / len(sub), 1),
    }


def main():
    t0 = time.time()
    rows = load()
    X = np.array([r["x"] for r in rows], dtype=np.float32)
    y = np.array([r["y"] for r in rows], dtype=np.int32)
    groups = np.array([r["token"] for r in rows])
    mercados = sorted(set(groups.tolist()))
    log(f"filas: {len(rows)} | mercados: {len(mercados)} | "
        f"positivos: {y.mean():.3f}")

    oof = np.full(len(y), np.nan)
    resultados, thresholds_q = [], []
    gkf = GroupKFold(n_splits=K_FOLDS)
    for k, (tr, va) in enumerate(gkf.split(X, y, groups)):
        # 2. estandarización fit SOLO con train
        mu = X[tr].mean(axis=0, keepdims=True)
        sd = X[tr].std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        Xtr = (X[tr] - mu) / sd
        Xva = (X[va] - mu) / sd
        ytr = y[tr]
        n1 = ytr.sum(); n0 = len(ytr) - n1
        model = lgb.train(
            {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
             "num_leaves": 31, "min_child_samples": 30,
             "scale_pos_weight": n0 / max(n1, 1), "verbose": -1, "seed": SEED},
            lgb.Dataset(Xtr, label=ytr), num_boost_round=200)

        proba_tr = model.predict(Xtr)
        proba_va = model.predict(Xva)
        oof[va] = proba_va
        auc = roc_auc_score(y[va], proba_va)
        pr = average_precision_score(y[va], proba_va)

        # 4. umbral por cuantil de TRAIN (política a priori, sin mirar test)
        thr_q = float(np.quantile(proba_tr, CUANTIL))
        thresholds_q.append(thr_q)
        mask_q = proba_va < thr_q
        rows_va = [rows[i] for i in va]
        sin_f = pnl_stats(rows_va)
        f_q = pnl_stats(rows_va, mask=mask_q)
        sens = {}
        for q in CUANTILES_SENS:
            tq = float(np.quantile(proba_tr, q))
            mq = proba_va < tq
            sens[f"cuantil_{q}"] = {
                "umbral": round(tq, 4),
                "retencion": round(float(mq.mean()), 4),
                **pnl_stats(rows_va, mask=mq),
            }
        f_fijo = pnl_stats(rows_va, mask=proba_va < UMBRAL_FIJO_REF)
        resultados.append({
            "fold": k,
            "train_n": int(len(tr)), "test_n": int(len(va)),
            "mercados_test": int(len(set(groups[va].tolist()))),
            "auc_oos": round(float(auc), 4),
            "pr_auc_oos": round(float(pr), 4),
            "umbral_cuantil": round(thr_q, 4),
            "retencion": round(float(mask_q.mean()), 4),
            "sin_filtro": sin_f,
            "cuantil": f_q,
            "sensibilidad_cuantil": sens,
            f"fijo_{UMBRAL_FIJO_REF}": f_fijo,
        })
        log(f"fold {k+1}: AUC={auc:.3f} PR={pr:.3f} | thr_q({CUANTIL})="
            f"{thr_q:.3f} ret={mask_q.mean():.0%} | pnl sin filtro "
            f"{sin_f['pnl']}$ -> cuantil {f_q['pnl']}$ | "
            f"fijo {UMBRAL_FIJO_REF}: {f_fijo['pnl']}$")

    # agregados OOF
    valid = ~np.isnan(oof)
    auc_pooled = roc_auc_score(y[valid], oof[valid])
    pr_pooled = average_precision_score(y[valid], oof[valid])

    # por horizonte (OOF)
    auc_por_h = {}
    hs = np.array([r["h"] for r in rows])
    for h in sorted(set(hs.tolist())):
        idx = valid & (hs == h)
        yy = y[idx]
        if len(set(yy.tolist())) < 2:
            continue
        auc_por_h[f"h{h}s"] = {
            "n": int(idx.sum()),
            "auc": round(float(roc_auc_score(yy, oof[idx])), 4),
            "pr_auc": round(float(average_precision_score(yy, oof[idx])), 4),
        }

    # P&L maker agregado OOS: sin filtro vs cuantil vs fijo (suma de folds)
    agg_sin = pnl_stats([rows[i] for i in np.where(valid)[0]])
    agg_cu = {"n": 0, "pnl": 0.0}
    agg_fijo = {"n": 0, "pnl": 0.0}
    ret_celdas = []
    for r in resultados:
        agg_cu["n"] += r["cuantil"]["n"]
        agg_cu["pnl"] += r["cuantil"]["pnl"]
        agg_fijo["n"] += r[f"fijo_{UMBRAL_FIJO_REF}"]["n"]
        agg_fijo["pnl"] += r[f"fijo_{UMBRAL_FIJO_REF}"]["pnl"]
        ret_celdas.append(r["retencion"])
    retencion = float(np.mean(ret_celdas))

    resumen = {
        "fecha": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "paso": 2,
        "metodologia": (
            "GroupKFold por mercado (K=5), estandarización fit con train, "
            "umbral por cuantil de predicciones de train (política a priori), "
            "PR-AUC con baseline de prevalencia, P&L MAKER exacto "
            "(pnl = -(agresor), mid_futuro real)"),
        "convencion": "MAKER: el filtro retiene p<thr (flujo benigno para LP)",
        "filas": len(rows),
        "mercados": len(mercados),
        "k_folds": K_FOLDS,
        "cuantil": CUANTIL,
        "auc_pooled_honesto": round(float(auc_pooled), 4),
        "pr_auc_oos": round(float(pr_pooled), 4),
        "prevalencia_positiva": round(float(y.mean()), 4),
        "auc_por_horizonte": auc_por_h,
        "umbral_cuantil_por_fold": [round(t, 4) for t in thresholds_q],
        "retencion_cuantil": round(retencion, 4),
        "maker_pnl": {
            "sin_filtro": agg_sin,
            "cuantil": agg_cu,
            f"fijo_{UMBRAL_FIJO_REF}": agg_fijo,
        },
        "folds": resultados,
    }
    with open(OUT, "w") as f:
        json.dump(resumen, f, indent=1)
    log(f"resultados -> {OUT}")

    log("=== RESUMEN PASO 2 (folds por mercado, convención MAKER) ===")
    log(f"AUC pooled honesto (mercados no vistos): {auc_pooled:.4f}")
    log(f"PR-AUC OOS: {pr_pooled:.4f} (prevalencia {y.mean():.3f})")
    log(f"P&L maker OOS sin filtro:        {agg_sin['pnl']}$ "
        f"({agg_sin['n']} fills)")
    log(f"P&L maker OOS cuantil {CUANTIL}:            {agg_cu['pnl']}$ "
        f"({agg_cu['n']} fills, retención {retencion:.0%})")
    log(f"P&L maker OOS umbral fijo {UMBRAL_FIJO_REF}:     {agg_fijo['pnl']}$ "
        f"({agg_fijo['n']} fills)")
    log(f"umbral por fold: {[round(t, 3) for t in thresholds_q]}")
    log(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
