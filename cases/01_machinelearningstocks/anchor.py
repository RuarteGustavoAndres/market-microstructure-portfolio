"""Anchor: run the AUTHOR's own backtesting.py under np.random.seed(i) and compare, number by number, with our
re-implementation using train_test_split(random_state=i). Same permutation => results must match exactly."""
import contextlib, io, os, re, sys
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
REPO = "/tmp/cand/MachineLearningStocks"
sys.path.insert(0, REPO); os.chdir(REPO)
import backtesting as author   # the author's file, unmodified
df = pd.read_csv("keystats.csv", index_col="Date").dropna(axis=0, how="any")
X = df[df.columns[6:]].values                      # the author's own feature selection
y = list((df["stock_p_change"] - df["SP500_p_change"]) >= 10)
z = np.array(df[["stock_p_change", "SP500_p_change"]])
# OUR feature selection (copied from fix1/audit_mls.py): columns after SP500_p_change on a frame read WITHOUT index_col
d2 = pd.read_csv("keystats.csv"); d2["Date"] = pd.to_datetime(d2["Date"]); d2 = d2.dropna(axis=0, how="any").reset_index(drop=True)
feat = list(d2.columns[list(d2.columns).index("SP500_p_change") + 1:])
Xo = d2[feat].values
print("n features: author", X.shape[1], "| ours", Xo.shape[1], "| identical matrix:", np.array_equal(X, Xo))
ok = True
for i in range(8):
    np.random.seed(i)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): author.backtest()
    t = buf.getvalue()
    A = dict(acc=float(re.search(r"Accuracy score:\s+([\d.]+)", t).group(1)), prec=float(re.search(r"Precision score:\s+([\d.]+)", t).group(1)),
             trades=int(re.search(r"Total Trades: (\d+)", t).group(1)), out=float(re.search(r"earns\s+(-?[\d.]+) percentage", t).group(1)))
    tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=i)
    yy = np.array(y, int); m = RandomForestClassifier(n_estimators=100, random_state=0).fit(Xo[tr], yy[tr]); p = m.predict(Xo[te])
    sel = p == 1
    O = dict(acc=round(float((p == yy[te]).mean()), 2), prec=round(float(precision_score(yy[te], p)), 2), trades=int(sel.sum()),
             out=round(float(z[te][sel, 0].mean() - z[te][sel, 1].mean()), 1))
    same = (A == O); ok &= same
    print(i, "autor", A, "| nuestro", O, "| IGUAL" if same else "| DIFIERE")
print("TODAS IGUALES:", ok)
