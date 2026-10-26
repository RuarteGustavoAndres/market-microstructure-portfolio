"""Walk-Forward HONESTO del filtro de toxicidad: ¿se mantiene el P&L neto
cuando el modelo solo ve el PASADO y el umbral no se ajusta por fold?

Metodología (López de Prado 2018 ch.7/12; Pardo 1992 walk-forward):
  1. Splits TEMPORALES contiguos (K bloques por tiempo, no aleatorios):
     para cada bloque de test k, se entrena SOLO con datos estrictamente
     anteriores al inicio del bloque, con EMBARGO de 300s (= horizonte máximo
     de etiqueta) entre el último dato de train y el primer dato de test.
     Purga: ninguna etiqueta del train usa información posterior al inicio
     del test (el embargo lo garantiza porque horizonte_max = 300s).
  2. P&L EXACTO por fill, en CONVENCIÓN MAKER (el bot es LP: cota pasiva y
     el fill llega cuando un AGRESOR cruza; el tape registra fills de
     agresores, así que el P&L del maker por fill = -(P&L del agresor)):
     agresor BUY : su P&L = (mid_futuro - fill_price)
     agresor SELL: su P&L = (fill_price - mid_futuro)
     pnl_maker = -side*(mid_futuro - fill_price)   [centavos/share]
     con mid_futuro = mid del libro REAL a t+horizonte (columna del dataset,
     misma fuente que las etiquetas: libro_snapshots).
     El filtro p<thr RETIENE el flujo con baja P(favorable al agresor) =
     flujo no informado, que es el que conviene recibir como LP (el costo
     adverso del maker = el agresor gana post-fill). Veredicto del
     arbitraje 19-oct (Claude): el filtro p<thr es CORRECTO para maker;
     el bug era reportar el P&L del lado del agresor.
  3. Grilla de umbrales por fold: si cada fold elige un umbral distinto ->
     señal de sobreajuste del umbral. Se reporta además el P&L agregado a
     umbral FIJO 0.7 (sin mirar el futuro al elegir) y el Walk-Forward
     Efficiency (OOS/IS): <30% = aviso de sobreajuste (literatura).

Uso: ./venv-ml/bin/python backtest_walkforward.py
Salida: dataset/walkforward_toxicidad.json
"""

import csv
import json
import os
import sys
import time

import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dataset", "ml_features.csv")
OUT = os.path.join(HERE, "dataset", "walkforward_toxicidad.json")

# fill_size y size_rel_depth EXCLUÍDOS: características del trade entrante que
# un maker no conoce al cotizar (fuga para la estrategia maker — revisión
# Claude 19-oct). Las columnas siguen en el CSV para análisis, no como features.
FEATURES = ["spread_rel", "imbalance", "depth_total", "n_total",
            "momentum_60s", "momentum_300s", "ofi", "microprice_bias",
            "side_buy", "horizonte_s"]
EMBARGO_S = 300          # = horizonte máximo de etiqueta
K_BLOQUES = 6            # bloques temporales contiguos
UMBRAL_FIJO = 0.7        # elegido a priori (no se ajusta con el futuro)
UMBRAL_FIJO_CONSERV = 0.8  # variante conservadora
GRILLA = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

# Horizonte único opcional (--horizonte 300): filtra el dataset a UN horizonte.
# Motivo (revisión Claude 22-oct): sumar los 4 horizontes de cada fill cuenta el
# mismo fill 4 veces (los fills no son independientes entre horizontes). Con
# --horizonte H cada fill entra UNA vez; el OUT usa sufijo _h<H>.
HORIZONTE_UNICO = None
if "--horizonte" in sys.argv:
    HORIZONTE_UNICO = int(float(sys.argv[sys.argv.index("--horizonte") + 1]))
    OUT = OUT.replace(".json", f"_h{HORIZONTE_UNICO}.json")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load():
    """Filas etiquetadas, convención MAKER: pnl = -side*(mid_futuro-price)."""
    rows = []
    with open(DATA) as f:
        for r in csv.DictReader(f):
            if r["target"] == "" or r["mid_futuro"] == "":
                continue
            if HORIZONTE_UNICO is not None and int(float(r["horizonte_s"])) != HORIZONTE_UNICO:
                continue
            mf, fp = float(r["mid_futuro"]), float(r["fill_price"])
            side = 1.0 if float(r["side_buy"]) > 0.5 else -1.0
            pnl_agresor = side * (mf - fp) * 100.0  # centavos/share (el tape registra fills de agresores)
            rows.append({
                "ts": float(r["ts_epoch"]),
                "token": r["token_id"],
                "h": int(float(r["horizonte_s"])),
                "pnl": -pnl_agresor,  # convención MAKER: P&L del LP = -(P&L del agresor)
                "x": [float(r[k]) if r[k] not in ("", None) else 0.0
                      for k in FEATURES],
                "y": 1 if float(r["target"]) > 0.5 else 0,
            })
    rows.sort(key=lambda r: r["ts"])
    return rows


def pnl_stats(sub, thr=None, proba=None):
    """P&L MAKER (USD a $1/share nominal), ops y hit-rate de un subconjunto."""
    if thr is not None:
        sub = [r for r, p in zip(sub, proba) if p < thr]
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
    log(f"filas etiquetadas con mid_futuro: {len(rows)} | "
        f"{len(set(r['token'] for r in rows))} tokens")

    # bloques temporales contiguos
    ts_all = [r["ts"] for r in rows]
    tmin, tmax = ts_all[0], ts_all[-1]
    bordes = np.linspace(tmin, tmax, K_BLOQUES + 1)
    bloques = []
    for k in range(K_BLOQUES):
        b = [r for r in rows if bordes[k] <= r["ts"] < bordes[k + 1]]
        if k == K_BLOQUES - 1:
            b = [r for r in rows if r["ts"] >= bordes[k]]
        bloques.append(b)
        log(f"  bloque {k+1}: {len(b)} filas "
            f"({time.strftime('%m-%d %H:%M', time.gmtime(bordes[k]))} -> "
            f"{time.strftime('%m-%d %H:%M', time.gmtime(bordes[k+1]))} UTC)")

    resultados, umbral_optimo_por_fold = [], []
    oos_y, oos_p, oos_tokens = [], [], []  # pooled OOS (provenance 24-oct)
    for k in range(1, K_BLOQUES):  # fold k: test = bloque k+1
        test = bloques[k]
        test_start = test[0]["ts"]
        # train = todo lo anterior al inicio del test MENOS el embargo
        train = [r for r in rows if r["ts"] < test_start - EMBARGO_S]
        if len(train) < 5000 or len(test) < 500:
            log(f"fold {k}: datos insuficientes, se omite")
            continue
        Xtr = np.array([r["x"] for r in train])
        ytr = np.array([r["y"] for r in train])
        Xte = np.array([r["x"] for r in test])
        yte = np.array([r["y"] for r in test])
        n1 = ytr.sum(); n0 = len(ytr) - n1
        # device_type por env var (default cpu): benchmark 22-oct en fold corto
        # (129k filas, 200 rondas, mismos params) dio CPU ~1.0s vs GPU ~1.4s;
        # la GPU (LGB_DEVICE=gpu) queda disponible sin tocar features/params.
        model = lgb.train(
            {"objective": "binary", "metric": "auc", "learning_rate": 0.05,
             "num_leaves": 31, "min_child_samples": 30,
             "scale_pos_weight": n0 / max(n1, 1), "verbose": -1, "seed": 42,
             "device_type": os.environ.get("LGB_DEVICE", "cpu")},
            lgb.Dataset(Xtr, label=ytr), num_boost_round=200)
        proba = model.predict(Xte)
        auc = roc_auc_score(yte, proba)
        oos_y.extend(int(v) for v in yte)
        oos_p.extend(float(v) for v in proba)
        oos_tokens.extend(r["token"] for r in test)

        sin_filtro = pnl_stats(test)
        grid = {t: pnl_stats(test, thr=t, proba=proba) for t in GRILLA}
        viable = {t: s for t, s in grid.items() if s["n"] >= 0.05 * len(test)}
        mejor_t = max(viable, key=lambda t: viable[t]["pnl"]) if viable else None
        umbral_optimo_por_fold.append(mejor_t)
        fijo = pnl_stats(test, thr=UMBRAL_FIJO, proba=proba)
        conserv = pnl_stats(test, thr=UMBRAL_FIJO_CONSERV, proba=proba)

        # in-sample (mismo umbral fijo, sobre train) para WFE
        proba_tr = model.predict(Xtr)
        is_fijo = pnl_stats(train, thr=UMBRAL_FIJO, proba=proba_tr)

        resultados.append({
            "fold": k,
            "train_n": len(train), "test_n": len(test),
            "auc_oos": round(float(auc), 4),
            "sin_filtro": sin_filtro,
            "mejor_umbral": mejor_t,
            "mejor_umbral_pnl": viable[mejor_t]["pnl"] if mejor_t else None,
            f"fijo_{UMBRAL_FIJO}": fijo,
            f"fijo_{UMBRAL_FIJO_CONSERV}": conserv,
            "in_sample_fijo": is_fijo,
            "grilla": grid,
        })
        log(f"fold {k}: AUC={auc:.3f} | sin filtro {sin_filtro['pnl']}$ "
            f"({sin_filtro['n']} ops) | mejor t={mejor_t} "
            f"({viable[mejor_t]['pnl'] if mejor_t else 0}$) | "
            f"fijo {UMBRAL_FIJO}: {fijo['pnl']}$ ({fijo['n']} ops)")

    # agregados
    def suma(d, key):
        return round(sum(r[key]["pnl"] for r in resultados if r[key]["n"] > 0), 2)

    oos_fijo = suma(resultados, f"fijo_{UMBRAL_FIJO}")
    oos_conserv = suma(resultados, f"fijo_{UMBRAL_FIJO_CONSERV}")
    oos_sin = suma(resultados, "sin_filtro")
    oos_opt = sum(r["mejor_umbral_pnl"] or 0 for r in resultados)
    is_fijo_total = suma(resultados, "in_sample_fijo")
    # WFE POR OPERACIÓN: los n de IS y OOS difieren (IS acumula folds);
    # comparar P&L por op, no totales (revisión Claude 19-oct).
    oos_n = sum(r[f"fijo_{UMBRAL_FIJO}"]["n"] for r in resultados)
    is_n_total = sum(r["in_sample_fijo"]["n"] for r in resultados)
    wfe_totals = round(100 * oos_fijo / is_fijo_total, 1) if is_fijo_total > 0 else None
    wfe = None
    if oos_n and is_n_total:
        is_op = is_fijo_total / is_n_total
        oos_op = oos_fijo / oos_n
        wfe = round(100 * oos_op / is_op, 1) if is_op > 0 else None
    thresholds = [t for t in umbral_optimo_por_fold if t is not None]

    resumen = {
        "fecha": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
        "horizonte_unico": HORIZONTE_UNICO,
        "convencion": "MAKER: pnl por fill = -(P&L del agresor del tape); "
                      "el filtro retiene p<thr (flujo benigno para el LP). "
                      "Veredicto arbitraje 19-oct.",
        "metodologia": "walk-forward temporal contiguo, embargo 300s, "
                       "P&L maker exacto vs mid_futuro real, umbral fijo a priori",
        "filas": len(rows), "bloques": K_BLOQUES,
        "umbral_fijo": UMBRAL_FIJO,
        "oos_sin_filtro_usd": oos_sin,
        f"oos_fijo_{UMBRAL_FIJO}_usd": oos_fijo,
        f"oos_fijo_{UMBRAL_FIJO_CONSERV}_usd": oos_conserv,
        "oos_umbral_optimo_por_fold_usd": round(oos_opt, 2),
        "umbral_optimo_por_fold": umbral_optimo_por_fold,
        "umbral_estable": (len(set(thresholds)) == 1 if thresholds else None),
        "in_sample_fijo_usd": is_fijo_total,
        "in_sample_fijo_n": is_n_total,
        "oos_fijo_n": oos_n,
        # métricas pooled de clasificación sobre TODO el OOS (cada fila una vez
        # dentro de su fold; en el esquema fill×horizonte cada fill aparece una
        # vez por horizonte). Agregado 24-oct para provenance del titular README.
        "n_oos": len(oos_y),
        "auc_pooled_oos": round(float(roc_auc_score(oos_y, oos_p)), 4) if oos_y else None,
        "pr_auc_oos": round(float(average_precision_score(oos_y, oos_p)), 4) if oos_y else None,
        "prevalencia_oos": round(sum(oos_y) / len(oos_y), 4) if oos_y else None,
        "fills_distintos_oos": len(set(oos_tokens)),
        "wfe_maker_per_op_pct": wfe,
        "wfe_totals_pct": wfe_totals,
        "nota_wfe": "WFE POR OPERACIÓN (OOS pnl/op vs IS pnl/op a umbral fijo); "
                    "<30% = aviso de sobreajuste (literatura walk-forward)",
        "folds": resultados,
    }
    with open(OUT, "w") as f:
        json.dump(resumen, f, indent=1)
    log(f"resultados -> {OUT}")

    log("=== RESUMEN (convención MAKER: pnl = -(agresor)) ===")
    log(f"P&L OOS maker sin filtro:      {oos_sin}$")
    log(f"P&L OOS maker umbral fijo {UMBRAL_FIJO}:  {oos_fijo}$")
    log(f"P&L OOS maker umbral fijo {UMBRAL_FIJO_CONSERV}:  {oos_conserv}$")
    log(f"P&L OOS maker umbral óptimo c/fold (techo): {round(oos_opt, 2)}$")
    log(f"Umbrales óptimos por fold: {umbral_optimo_por_fold} "
        f"-> estable: {resumen['umbral_estable']}")
    log(f"WFE maker por-operación (OOS/IS a umbral fijo): {wfe}% "
        f"(por totales: {wfe_totals}%)")


if __name__ == "__main__":
    main()
