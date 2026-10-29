#!/usr/bin/env python3
"""Audit of huseinzol05/Stock-Prediction-Models (agent notebooks) on GOOG-year.csv.

Faithful-extraction audit: the notebook cells are concatenated verbatim and only
mechanically required transformations are applied, each recorded per run in the JSON.
Original runs used notebook-era pandas (0.x); five compat strips are needed:
  - drop `%%time` (IPython magic, SyntaxError in a plain script)
  - drop dead `import tensorflow as tf` in agents 21/22 (no tf.* op is ever used)
  - drop the dead `pkg_resources` metadata block in agent 6 (never executed on its data)
  - agent 2: `signals['signal'][short_window:] = ...` (chained assignment) is a silent
    no-op under pandas 3 Copy-on-Write -> minimal `.loc` equivalent; without it the
    agent makes zero trades and prints nothing.
Modes (usage: audit_sp.py <clone-of-Stock-Prediction-Models> <out_dir> [mode]):
  published    extract the saved "total balance" outputs from the 7 notebooks (no execution)
  repro        re-run the 4 rule-based agents (1,2,3,23); deterministic; 2 needs the .loc fix
  stochastic   agents 6/21/22 with fixed seeds (they ship none); records balance and
               min cash during the final buy() -> evidence for the unbacked-buying gate bug
  lookahead    re-run 2 with execution delayed 1 day and 23 with the signal shifted to the
               day the a-b-c-d pattern first becomes knowable (min d-a = 12 days)
  oos          agents 6/21/22 retrained on the first 70% of days, evaluated on the last 30%;
               with the recorded seeds every stochastic number is deterministic
All subprocesses run with MPLBACKEND=Agg and cwd=<clone>/agent (scripts read
'dataset/GOOG-year.csv' relatively). Nothing in the clone is modified.
"""
import hashlib
import json
import os
import platform
import re
import subprocess
import sys

import numpy as np
import pandas as pd

REPO = os.path.abspath(sys.argv[1])
OUT = os.path.abspath(sys.argv[2])
MODE = sys.argv[3] if len(sys.argv) > 3 else "all"
AGENT_DIR = os.path.join(REPO, "agent")
DATA = os.path.join(REPO, "dataset", "GOOG-year.csv")
SKIP_LINES = {"import tensorflow as tf", "%%time", "import pkg_resources", "import types"}
BLOCK_HEADS = ("def get_imports():", "imports = list(set(get_imports()))",
               "for m in pkg_resources.working_set:", "if m.project_name in imports",
               "requirements.append((m.project_name, m.version))",
               "for r in requirements:", "print('{}=={}'.format(*r))")
AGENTS = {"1": "turtle", "2": "moving-average", "3": "signal-rolling",
          "6": "evolution-strategy", "21": "neuro-evolution",
          "22": "neuro-evolution-novelty-search", "23": "abcd-strategy"}
SEEDS = {"6": [42, 43, 44, 45, 46], "21": [42, 43, 44, 45, 46], "22": [42, 43, 44]}


def nb_src(num):
    path = os.path.join(AGENT_DIR, f"{num}.{AGENTS[num]}-agent.ipynb")
    nb = json.load(open(path))
    return ("\n".join("".join(c.get("source", [])) for c in nb["cells"]
                      if c["cell_type"] == "code"), path)


def nb_out_balance(num):
    """Last 'total balance' line saved in the notebook's own outputs."""
    path = os.path.join(AGENT_DIR, f"{num}.{AGENTS[num]}-agent.ipynb")
    nb = json.load(open(path))
    vals = []
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            tp = o.get("data", {}).get("text/plain", "")
            tp = "".join(tp) if isinstance(tp, list) else (tp or "")
            txt = "".join(o.get("text", [])) or tp
            vals += re.findall(r"total balance (-?[\d.]+)", txt)
    return float(vals[-1]) if vals else None


def strip(src):
    out, skip = [], False
    for line in src.split("\n"):
        s = line.strip()
        if s in SKIP_LINES:
            continue
        if any(s.startswith(b) for b in BLOCK_HEADS):
            skip = (s == "def get_imports():")
            continue
        if skip:
            if s.startswith("yield name"):
                skip = False
            continue
        out.append(line)
    return "\n".join(out)


INSTRUMENT = ("initial_money -= self.trend[t]; _MIN_CASH.append(initial_money)")
CUT_SNIPPET = ("close = df.Close.values.tolist()\n"
               "_CUT = int(len(close) * 0.7)\n"
               "close_test = close[_CUT-1:]\n"
               "close = close[:_CUT]")


def instrument(src, seeds, oos=False):
    code = strip(src)
    code = code.replace("initial_money -= self.trend[t]", INSTRUMENT)
    code = "_MIN_CASH = []\n" + code
    if oos:
        code = code.replace("close = df.Close.values.tolist()", CUT_SNIPPET)
        code = re.sub(r"(\n(states_buy, states_sell, total_gains, invest = )((?:agent|neural_evolve)\.buy[^\n]*))",
                      r"\n_MIN_CASH.clear()\n(agent if 'agent' in dir() else neural_evolve).trend = close_test\n\2\3",
                      code)
        code = re.sub(r"_MIN_CASH\.clear\(\)\n",
                      "_MIN_CASH.clear()\nprint('MIN_CASH', (min(_MIN_CASH) if _MIN_CASH else None))\n",
                      code, count=1)
    else:
        code = re.sub(r"(\n(states_buy, states_sell, total_gains, invest = )((?:agent|neural_evolve)\.buy[^\n]*))",
                      r"\n_MIN_CASH.clear()\n\2\3\nprint('MIN_CASH', (min(_MIN_CASH) if _MIN_CASH else None))",
                      code)
    return code


def run(code, tag):
    path = os.path.join(OUT, f"_run_{tag}.py")
    open(path, "w").write(code)
    env = dict(os.environ, MPLBACKEND="Agg")
    r = subprocess.run([sys.executable, path], capture_output=True, text=True,
                       cwd=AGENT_DIR, env=env, timeout=600)
    return r


def balances(stdout):
    return [float(x) for x in re.findall(r"total balance (-?[\d.]+)", stdout)]


def loc_fix(src):
    return re.sub(r"signals\['signal'\]\[short_window:\]",
                  "signals.loc[short_window:, 'signal']", src)


def buy_hold():
    df = pd.read_csv(DATA)
    c = df["Close"].values
    bh = 10_000 / c[0] * c[-1]
    cut = int(len(c) * 0.7)
    return {"full": round(10_000 / c[0] * c[-1], 2),
            "train_70": round(10_000 / c[0] * c[cut - 1], 2),
            "test_30": round(10_000 / c[cut - 1] * c[-1], 2),
            "days": int(len(c)), "cut_index": cut,
            "first": str(pd.read_csv(DATA)["Date"].iloc[0]),
            "last": str(pd.read_csv(DATA)["Date"].iloc[-1])}


def env_block():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "repo_commit": subprocess.run(
                ["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True
            ).stdout.strip() or None,
            "goog_year_csv_sha256": hashlib.sha256(open(DATA, "rb").read()).hexdigest()}


def result():
    return {"env": env_block(), "buy_and_hold": buy_hold(), "agents": {}}


out = {}

# ---- published ------------------------------------------------------------
if MODE in ("published", "all"):
    r = result()
    for num in AGENTS:
        r["agents"][num + "-" + AGENTS[num]] = {"published_total_balance": nb_out_balance(num)}
    json.dump(r, open(os.path.join(OUT, "sp_published.json"), "w"), indent=1)
    out["published"] = True

# ---- repro (rule-based, deterministic) -------------------------------------
if MODE in ("repro", "all"):
    r = result()
    plans = {"1": (None, None), "2": (loc_fix, ".loc chained-assignment fix (pandas 3 CoW)"),
             "3": (None, None), "23": (None, None)}
    for num, (fx, note) in plans.items():
        src, _ = nb_src(num)
        code = fx(strip(src)) if fx else strip(src)
        rr = run(code, f"repro_{num}")
        b = balances(rr.stdout)
        pub = nb_out_balance(num)
        r["agents"][num + "-" + AGENTS[num]] = {
            "published_total_balance": pub,
            "reproduced_total_balance": b[-1] if b else None,
            "exact": bool(b and abs(b[-1] - pub) < 0.01),
            "transform": note or "none"}
    json.dump(r, open(os.path.join(OUT, "sp_repro.json"), "w"), indent=1)
    out["repro"] = True

# ---- stochastic (6/21/22, fixed seeds) --------------------------------------
if MODE in ("stochastic", "all"):
    r = result()
    for num, seeds in SEEDS.items():
        src, _ = nb_src(num)
        base = instrument(src, seeds)
        runs = []
        for s in seeds:
            rr = run("import numpy as np\nnp.random.seed(%d)\n" % s + base, f"sto_{num}_{s}")
            b = balances(rr.stdout)
            m = re.search(r"MIN_CASH (-?[\d.e]+|None)", rr.stdout)
            runs.append({"seed": s,
                         "final_balance": b[-1] if b else None,
                         "min_cash_during_buy": float(m.group(1)) if m and m.group(1) != "None" else None,
                         "rc": rr.returncode})
        r["agents"][num + "-" + AGENTS[num]] = {
            "published_total_balance": nb_out_balance(num),
            "runs": runs,
            "min_cash_negative_runs": sum(1 for x in runs if (x["min_cash_during_buy"] or 0) < 0),
            "transform": "dead imports/%%time stripped; _MIN_CASH instrumented in buy(); np.random.seed(s) prepended"}
    json.dump(r, open(os.path.join(OUT, "sp_stochastic.json"), "w"), indent=1)
    out["stochastic"] = True

# ---- lookahead --------------------------------------------------------------
if MODE in ("lookahead", "all"):
    r = result()
    # agent 2: signal uses close[i] (its own MA includes day i) but trades at close[i]
    src, _ = nb_src("2")
    code = loc_fix(strip(src)).replace(
        "signals['positions'] = signals['signal'].diff()",
        "signals['positions'] = signals['signal'].diff().shift(1).fillna(0.0)")
    rr = run(code, "look_2")
    r["agents"]["2-moving-average"] = {
        "published_total_balance": nb_out_balance("2"),
        "execution_delay_days": 1,
        "delayed_total_balance": balances(rr.stdout)[-1],
        "why": "buy decision at day i uses the short/long MA that includes close[i]; an "
               "executable version must trade at the next day's close"}
    # agent 23: the a,b,c,d pattern with step 4 is only knowable at d; min d-a = 12 days
    src, _ = nb_src("23")
    code = strip(src).replace("signal = abcd(df['Close'])",
                              "signal = abcd(df['Close'])\n"
                              "signal = np.concatenate([np.zeros(12), signal[:-12]])")
    rr = run(code, "look_23")
    r["agents"]["23-abcd-strategy"] = {
        "published_total_balance": nb_out_balance("23"),
        "signal_delay_days": 12,
        "delayed_total_balance": balances(rr.stdout)[-1],
        "why": "buy day a is signalled from ma[b], ma[c], ma[d] with b>=a, c>=b, d>=c "
               "(step 4), so the pattern is only knowable on day d; min d-a = 12 days"}
    json.dump(r, open(os.path.join(OUT, "sp_lookahead.json"), "w"), indent=1)
    out["lookahead"] = True

# ---- oos (6/21/22 train 70% / test 30%) -------------------------------------
if MODE in ("oos", "all"):
    r = result()
    for num, seeds in SEEDS.items():
        src, _ = nb_src(num)
        base = instrument(src, seeds, oos=True)
        runs = []
        for s in seeds:
            rr = run("import numpy as np\nnp.random.seed(%d)\n" % s + base, f"oos_{num}_{s}")
            b = balances(rr.stdout)
            m = re.search(r"MIN_CASH (-?[\d.e]+|None)", rr.stdout)
            runs.append({"seed": s,
                         "oos_balance": b[-1] if b else None,
                         "min_cash_during_buy": float(m.group(1)) if m and m.group(1) != "None" else None,
                         "rc": rr.returncode})
        r["agents"][num + "-" + AGENTS[num]] = {
            "published_total_balance": nb_out_balance(num),
            "runs": runs,
            "transform": "dead imports/%%time stripped; _MIN_CASH instrumented; np.random.seed(s); "
                         "close split 70/30, fit on train[:176], agent.trend = close[175:] for the final buy()"}
    json.dump(r, open(os.path.join(OUT, "sp_oos.json"), "w"), indent=1)
    out["oos"] = True

print(json.dumps(out))
