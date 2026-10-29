import sys, json
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv(sys.argv[1]); df["Date"] = pd.to_datetime(df["Date"])
df = df.dropna(axis=0, how="any").reset_index(drop=True)
feat = list(df.columns[list(df.columns).index('SP500_p_change') + 1:])
assert len(feat) == 41 and feat[0] == 'Market Cap' and 'SP500_p_change' not in feat, feat
X = df[feat].values
y = ((df.stock_p_change - df.SP500_p_change) >= 10).astype(int).values
rs, rm = df.stock_p_change.values, df.SP500_p_change.values
yr = df.Date.dt.year.values
rows = []
for Y in (2009, 2011, 2012, 2013):
    te = np.where(yr == Y)[0]
    tr = np.where(df.Date.values < (pd.Timestamp(f"{Y}-01-01") - pd.Timedelta(days=365)).to_datetime64())[0]
    p = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=2).fit(X[tr], y[tr]).predict(X[te])
    d = pd.DataFrame(dict(m=df.Date.iloc[te].dt.to_period("M").astype(str).values, p=p, y=y[te], rs=rs[te], rm=rm[te]))
    rows.append(d)
d = pd.concat(rows, ignore_index=True)
months = d.m.unique()
g = {m: d[d.m == m] for m in months}
def stat(parts):
    z = pd.concat(parts); b = z[z.p == 1]
    if len(b) == 0: return np.nan, np.nan, 0
    return b.rs.mean() - b.rm.mean(), b.y.mean() - z.y.mean(), len(b)
base = stat([g[m] for m in months])
rng = np.random.RandomState(1); out = []
for _ in range(2000):
    s = rng.choice(months, len(months), replace=True)
    out.append(stat([g[m] for m in s])[:2])
o = np.array(out, float)
print("meses (bloques):", len(months), "| operaciones:", base[2])
print(f"outperformance puntos: {base[0]:+.2f}  IC95% [{np.nanpercentile(o[:,0],2.5):+.2f}, {np.nanpercentile(o[:,0],97.5):+.2f}]")
print(f"precision - prevalencia: {base[1]:+.3f}  IC95% [{np.nanpercentile(o[:,1],2.5):+.3f}, {np.nanpercentile(o[:,1],97.5):+.3f}]")
json.dump(dict(
    method="block bootstrap by calendar month (2000 resamples, seed 1) over the pooled temporal walk-forward test rows (2009, 2011-2013)",
    n_month_blocks=int(len(months)), n_trades=int(base[2]),
    outperformance_pts=dict(point=float(base[0]), ci95=[float(np.nanpercentile(o[:, 0], 2.5)), float(np.nanpercentile(o[:, 0], 97.5))]),
    precision_minus_prevalence=dict(point=float(base[1]), ci95=[float(np.nanpercentile(o[:, 1], 2.5)), float(np.nanpercentile(o[:, 1], 97.5))]),
), open(sys.argv[2], "w"), indent=1)
