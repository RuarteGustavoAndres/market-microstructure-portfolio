#!/usr/bin/env python3
"""Audit of robertmartin8/MachineLearningStocks backtest (reads keystats.csv only; runs none of the repo's code).

Same model/features/label/accounting as the repo's backtesting.py:
  RandomForestClassifier(n_estimators=100, random_state=0), features = the 41 columns after SP500_p_change (Market Cap ... Shares Short (prior month)), label = stock_p_change - SP500_p_change >= 10,
  'trade' = every row predicted 1, return = mean(stock_p_change) vs mean(SP500_p_change) over those rows.
Only the split changes.

E0  reproduce: 100 random 80/20 splits (the repo uses a single unseeded split)
E1  temporal expanding walk-forward, 365-day embargo (label horizon = 1 year), test = following calendar year
E2  random split restricted to the same test period as E1's pooled test rows (apples to apples)
E3  GroupKFold by ticker (is it ticker memorisation?)
usage: audit_mls.py keystats.csv out.json
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import GroupKFold, train_test_split

PATH, OUT = sys.argv[1], sys.argv[2]
df = pd.read_csv(PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.dropna(axis=0, how="any").reset_index(drop=True)  # same as repo
feat = list(df.columns[list(df.columns).index('SP500_p_change') + 1:])
assert len(feat) == 41 and feat[0] == 'Market Cap' and 'SP500_p_change' not in feat, feat  # same inputs as the project's backtesting.py
X = df[feat].values
y = ((df["stock_p_change"] - df["SP500_p_change"]) >= 10).astype(int).values
r_s = df["stock_p_change"].values
r_m = df["SP500_p_change"].values
print(f"rows after dropna {len(df)} | features {len(feat)} | positives {y.mean():.3f} | dates {df.Date.min().date()} -> {df.Date.max().date()}")


def clf():
    return RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=4)


def report(idx_test, pred):
    p = pred == 1
    n = int(p.sum())
    if n == 0:
        return dict(n_test=len(idx_test), n_trades=0)
    yt = y[idx_test]
    return dict(
        n_test=int(len(idx_test)), n_trades=n,
        acc=float((pred == yt).mean()), prec=float(precision_score(yt, pred, zero_division=0)),
        prevalence=float(yt.mean()),
        ret=float(r_s[idx_test][p].mean()), mkt=float(r_m[idx_test][p].mean()),
        out=float(r_s[idx_test][p].mean() - r_m[idx_test][p].mean()),
        ret_all=float(r_s[idx_test].mean()), mkt_all=float(r_m[idx_test].mean()),
    )


res = {}

# E0
rows = []
idx = np.arange(len(df))
for seed in range(100):
    tr, te = train_test_split(idx, test_size=0.2, random_state=seed)
    m = clf().fit(X[tr], y[tr])
    rows.append(report(te, m.predict(X[te])))
E0 = pd.DataFrame(rows)
res["E0_random_split_100seeds"] = {k: dict(mean=float(E0[k].mean()), p5=float(E0[k].quantile(.05)), p95=float(E0[k].quantile(.95)))
                                   for k in ["acc", "prec", "prevalence", "n_trades", "ret", "mkt", "out"]}
print("E0 random split:", {k: round(v["mean"], 3) for k, v in res["E0_random_split_100seeds"].items()})

# E1
yr = df.Date.dt.year.values
test_rows, per_year = [], {}
all_te, all_pred = [], []
for Y in range(2006, 2014):
    te = np.where(yr == Y)[0]
    t0 = pd.Timestamp(f"{Y}-01-01")
    tr = np.where(df.Date.values < (t0 - pd.Timedelta(days=365)).to_datetime64())[0]
    if len(te) < 50 or len(tr) < 500:
        continue
    m = clf().fit(X[tr], y[tr])
    pred = m.predict(X[te])
    per_year[Y] = dict(n_train=int(len(tr)), **report(te, pred))
    all_te.append(te)
    all_pred.append(pred)
te_all = np.concatenate(all_te)
pred_all = np.concatenate(all_pred)
res["E1_temporal_walkforward_embargo365"] = dict(pooled=report(te_all, pred_all), per_year=per_year)
print("E1 temporal pooled:", {k: round(v, 3) for k, v in res["E1_temporal_walkforward_embargo365"]["pooled"].items()})
for Y, d in per_year.items():
    print(f"   {Y}: train {d['n_train']:5d} test {d['n_test']:4d} trades {d['n_trades']:4d} prec {d.get('prec', float('nan')):.2f} (prev {d.get('prevalence', float('nan')):.2f}) out {d.get('out', float('nan')):+6.1f}")

# E2: random split but evaluated only on the SAME rows E1 tests (train on all other rows incl. future & same year)
rows = []
for seed in range(20):
    rng = np.random.RandomState(seed)
    mask = np.ones(len(df), bool)
    te = te_all
    # 5-fold random CV over E1's test rows; train = every other row (past AND future)
    perm = rng.permutation(te)
    folds = np.array_split(perm, 5)
    preds = np.zeros(len(df), int)
    for f in folds:
        trm = np.ones(len(df), bool)
        trm[f] = False
        m = clf().fit(X[trm], y[trm])
        preds[f] = m.predict(X[f])
    rows.append(report(te, preds[te]))
E2 = pd.DataFrame(rows)
res["E2_random_cv_on_same_test_rows_20seeds"] = {k: dict(mean=float(E2[k].mean()), p5=float(E2[k].quantile(.05)), p95=float(E2[k].quantile(.95)))
                                                 for k in ["acc", "prec", "n_trades", "ret", "mkt", "out"]}
print("E2 random CV on E1 test rows:", {k: round(v["mean"], 3) for k, v in res["E2_random_cv_on_same_test_rows_20seeds"].items()})

# E3 GroupKFold by ticker
gkf = GroupKFold(n_splits=5)
preds = np.zeros(len(df), int)
for tr, te in gkf.split(X, y, groups=df["Ticker"].values):
    preds[te] = clf().fit(X[tr], y[tr]).predict(X[te])
res["E3_groupkfold_ticker"] = report(np.arange(len(df)), preds)
print("E3 group-by-ticker:", {k: round(v, 3) for k, v in res["E3_groupkfold_ticker"].items()})

import hashlib, platform, sklearn
res["env"] = dict(python=platform.python_version(), sklearn=sklearn.__version__, pandas=pd.__version__, numpy=np.__version__,
                  keystats_sha256=hashlib.sha256(open(PATH, "rb").read()).hexdigest(),
                  rows_after_dropna=int(len(df)), rows_raw=int(len(pd.read_csv(PATH))),
                  audited_repo="robertmartin8/MachineLearningStocks", audited_commit="1c3712af741a9a030edb678b37ef18d1d2db9147")
json.dump(res, open(OUT, "w"), indent=1)
print("saved", OUT)
