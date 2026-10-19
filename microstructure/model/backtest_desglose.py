"""DESGLOSE del veredicto walk-forward: ¿DÓNDE falla la traducción AUC->P&L?

Reusa la metodología de backtest_walkforward.py (splits temporales contiguos,
embargo 300s, P&L exacto con mid_futuro) pero guarda TODAS las predicciones
out-of-sample y desglosa el P&L OOS por:
  - horizonte (1s/10s/60s/300s)
  - lado (BUY/SELL)
  - agresividad del fill (pos_en_spread: 0=llena al bid del BUY, 1=al ask;
    ~0.5 = fill "maker-like" pegado al mid)  [solo segmentación de análisis,
    la feature NO entra al modelo]
  - spread del mercado al momento del fill (chico/medio/ancha)
y compara, en cada segmento: sin filtro vs filtro @0.7 fijo.
Además prueba un ajuste "maker-realista": al P&L de un fill maker-like se le
acredita medio spread capturado (el ingreso del maker que la simulación de
taker-fills no tiene) y se reintenta el filtro.

CONVENCIÓN MAKER (veredicto del arbitraje 19-oct): el P&L por fill se
reporta desde el lado del LP = -(P&L del agresor del tape). El filtro
p<UMBRAL retiene el flujo benigno para el maker y NO se invierte.

Uso: ./venv-ml/bin/python backtest_desglose.py
Salida: dataset/desglose_walkforward.json + tabla en stdout.
"""

import csv, json, os, time
from collections import defaultdict
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dataset", "ml_features.csv")
OUT = os.path.join(HERE, "dataset", "desglose_walkforward.json")

FEATURES = ["spread_rel", "imbalance", "depth_total", "n_total",
            "momentum_60s", "momentum_300s", "fill_size",
            "size_rel_depth", "ofi", "microprice_bias",
            "side_buy", "horizonte_s"]
EMBARGO_S, K, UMBRAL = 300, 6, 0.7


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load():
    rows = []
    with open(DATA) as f:
        for r in csv.DictReader(f):
            if r["target"] == "" or r["mid_futuro"] == "":
                continue
            mf, fp = float(r["mid_futuro"]), float(r["fill_price"])
            side = 1.0 if float(r["side_buy"]) > 0.5 else -1.0
            bb, ba = float(r["best_bid"]), float(r["best_ask"])
            spread = ba - bb
            mid = (bb + ba) / 2
            pos = (fp - bb) / spread if spread > 0 else 0.5
            if side < 0:
                pos = 1 - pos
            rows.append({
                "ts": float(r["ts_epoch"]), "h": int(float(r["horizonte_s"])),
                # convención MAKER: P&L del LP = -(P&L del agresor del tape)
                "pnl": -side * (mf - fp) * 100.0,
                "x": [float(r[k]) if r[k] not in ("", None) else 0.0 for k in FEATURES],
                "y": 1 if float(r["target"]) > 0.5 else 0,
                "spread": spread, "pos": pos, "side": side,
            })
    rows.sort(key=lambda r: r["ts"])
    return rows


def seg(sub, proba=None, filtrado=False):
    if filtrado:
        sub = [r for r in sub if r["p"] < UMBRAL]
    if not sub:
        return {"n": 0, "pnl_usd": 0.0}
    return {"n": len(sub), "pnl_usd": round(sum(r["pnl"] for r in sub) / 100.0, 2),
            "pnl_op_c": round(float(np.mean([r["pnl"] for r in sub])), 3)}


def main():
    rows = load()
    log(f"filas: {len(rows)}")
    ts_all = [r["ts"] for r in rows]
    bordes = np.linspace(ts_all[0], ts_all[-1], K + 1)
    oos = []
    for k in range(1, K):
        test = [r for r in rows if bordes[k] <= r["ts"] < bordes[k + 1]] \
            if k < K - 1 else [r for r in rows if r["ts"] >= bordes[k]]
        if len(test) < 500:
            continue
        train = [r for r in rows if r["ts"] < test[0]["ts"] - EMBARGO_S]
        if len(train) < 5000:
            continue
        Xtr = np.array([r["x"] for r in train]); ytr = np.array([r["y"] for r in train])
        n1 = ytr.sum(); n0 = len(ytr) - n1
        m = lgb.train({"objective": "binary", "metric": "auc", "learning_rate": 0.05,
                       "num_leaves": 31, "min_child_samples": 30,
                       "scale_pos_weight": n0 / max(n1, 1), "verbose": -1, "seed": 42},
                      lgb.Dataset(Xtr, label=ytr), num_boost_round=200)
        pr = m.predict(np.array([r["x"] for r in test]))
        for r, p in zip(test, pr):
            r["p"] = float(p)
        oos.extend(test)
        log(f"fold {k}: test={len(test)} auc={roc_auc_score([r['y'] for r in test], pr):.3f}")

    log(f"OOS total: {len(oos)} filas")

    def bucket(r):
        p = r["pos"]
        return "maker_like" if 0.15 <= p <= 0.85 else ("taker_buy" if p < 0.15 else "taker_sell")

    def spread_bucket(s):
        return "spread<2c" if s < 0.02 else ("2-5c" if s < 0.05 else ">=5c")

    desglose = {}
    def desglosar(nombre, keyfn, filtrado):
        d = defaultdict(list)
        for r in oos:
            if filtrado and r["p"] >= UMBRAL:
                continue
            d[keyfn(r)].append(r["pnl"])
        out = {}
        for kk, v in sorted(d.items()):
            out[kk] = {"n": len(v), "pnl_usd": round(sum(v) / 100.0, 2),
                       "pnl_op_c": round(float(np.mean(v)), 3),
                       "hit_pct": round(100 * sum(1 for x in v if x > 0) / len(v), 1)}
        desglose[f"{nombre}{'_FILTRADO' if filtrado else ''}"] = out

    for filtrado in (False, True):
        desglosar("por_horizonte", lambda r: f"h{r['h']}s", filtrado)
        desglosar("por_lado", lambda r: "BUY" if r["side"] > 0 else "SELL", filtrado)
        desglosar("por_agresividad", bucket, filtrado)
        desglosar("por_spread", lambda r: spread_bucket(r["spread"]), filtrado)

    # ajuste maker-realista: acredita medio spread a los fills maker-like
    for label, filtrado in (("maker_ajustado", False), ("maker_ajustado_FILTRADO", True)):
        d = defaultdict(list)
        for r in oos:
            if filtrado and r["p"] >= UMBRAL:
                continue
            b = bucket(r)
            adj = (r["spread"] / 2) * 100 if b == "maker_like" else 0.0  # centavos ganados
            d[b].append(r["pnl"] + adj)
        desglose[label] = {kk: {"n": len(v), "pnl_usd": round(sum(v) / 100.0, 2),
                                "pnl_op_c": round(float(np.mean(v)), 3)}
                           for kk, v in sorted(d.items())}

    tot_sin = seg(oos)
    tot_con = seg(oos, filtrado=True)
    desglose["TOTAL"] = {"sin_filtro": tot_sin, "filtrado_0.7": tot_con}
    desglose["convencion"] = ("MAKER: pnl por fill = -(P&L del agresor del "
                              "tape); el filtro retiene p<0.7 (flujo benigno "
                              "para el LP). Veredicto arbitraje 19-oct.")

    with open(OUT, "w") as f:
        json.dump(desglose, f, indent=1)
    log(f"-> {OUT}")

    for k in ("por_horizonte", "por_horizonte_FILTRADO", "por_agresividad",
              "por_agresividad_FILTRADO", "por_lado", "por_lado_FILTRADO",
              "por_spread", "por_spread_FILTRADO",
              "maker_ajustado", "maker_ajustado_FILTRADO"):
        log(f"\n=== {k} ===")
        for kk, v in desglose[k].items():
            log(f"  {kk:16s} n={v['n']:7d} pnl={v['pnl_usd']:10.2f}$ "
                f"op={v.get('pnl_op_c', 0):+.3f}c hit={v.get('hit_pct', '-')}")


if __name__ == "__main__":
    main()
