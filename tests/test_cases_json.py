"""Case studies: JSON artifacts must parse, and the case page must match them.

Two gates:
1. every cases/**/*.json parses with json.load — regression for a shell
   wrapper line (EXIT=0) appended to a shipped JSON that no test caught;
2. every number the case page prints is derivable from the JSON it cites
   (same rounding). The page is never edited by hand away from the JSON:
   a number that changes in the JSON must change in the page, or CI fails.

stdlib + pytest only.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES_DIR = ROOT / "cases"
CASE_MD = ROOT / "docs" / "cases" / "CASE_01_MachineLearningStocks.md"

CASE_DIR = CASES_DIR / "01_machinelearningstocks"


def load(name):
    return json.loads((CASE_DIR / name).read_text(encoding="utf-8"))


def md_text():
    text = CASE_MD.read_text(encoding="utf-8")
    # normalize: unicode minus AND en-dash range -> ASCII hyphen, bold markers away
    return text.replace("−", "-").replace("–", "-").replace("**", "")


def _f1(x):
    return f"{x:.1f}"


def _f2(x):
    return f"{x:.2f}"


def test_every_cases_json_parses():
    files = sorted(CASES_DIR.glob("**/*.json"))
    assert files, "no JSON artifacts found under cases/ — check the path"
    bad = []
    for path in files:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - report the actual parse error
            bad.append(f"{path}: {exc}")
    assert not bad, "invalid JSON artifacts under cases/: " + "; ".join(bad)


def test_case_files_exist():
    for name in ("audit_mls.py", "boot_mls.py", "mls_audit.json",
                 "mls_bootstrap.json", "anchor.py", "anchor_stdout.txt"):
        assert (CASE_DIR / name).is_file(), f"missing {name}"
    assert CASE_MD.is_file(), "case page missing under docs/cases/"


def test_case_page_numbers_match_json():
    """Every printed number on the page is the JSON number, same rounding."""
    audit = load("mls_audit.json")
    boot = load("mls_bootstrap.json")
    md = md_text()

    e0 = audit["E0_random_split_100seeds"]
    e1 = audit["E1_temporal_walkforward_embargo365"]["pooled"]
    e2 = audit["E2_random_cv_on_same_test_rows_20seeds"]
    e3 = audit["E3_groupkfold_ticker"]

    required = [
        # row 1 — random split as in the project (E0)
        f"{_f2(e0['prec']['mean'])} ({_f2(e0['prevalence']['mean'])})",
        f"{_f1(e0['ret']['mean'])}%",
        f"{_f1(e0['mkt']['mean'])}%",
        "+" + _f1(e0["out"]["mean"]),
        f"{_f1(e0['out']['p5'])}-{_f1(e0['out']['p95'])}",
        # row 2 — random CV on the same test rows (E2; base rate = E1 pooled)
        f"{_f2(e2['prec']['mean'])} ({_f2(e1['prevalence'])})",
        f"{_f1(e2['ret']['mean'])}%",
        f"{_f1(e2['mkt']['mean'])}%",
        "+" + _f1(e2["out"]["mean"]),
        # row 3 — held-out companies (E3)
        f"{_f2(e3['prec'])} ({_f2(e3['prevalence'])})",
        f"{_f1(e3['ret'])}%",
        f"{_f1(e3['mkt'])}%",
        "+" + _f1(e3["out"]),
        # row 4 — time-ordered walk-forward (E1 pooled)
        f"{_f2(e1['prec'])} ({_f2(e1['prevalence'])})",
        f"{_f1(e1['ret'])}%",
        f"{_f1(e1['mkt'])}%",
        "+" + _f2(e1["out"]),
        # bootstrap uncertainty
        "+" + _f2(boot["outperformance_pts"]["point"]),
        _f1(boot["outperformance_pts"]["ci95"][0]),
        "+" + _f1(boot["outperformance_pts"]["ci95"][1]),
        f"{boot['precision_minus_prevalence']['point']:.3f}",
        f"{boot['precision_minus_prevalence']['ci95'][0]:.3f}",
        "+" + f"{boot['precision_minus_prevalence']['ci95'][1]:.3f}",
        # counts the page states
        f"{audit['env']['rows_raw']:,} rows",
        f"{audit['env']['rows_after_dropna']:,} after",
        str(e1["n_trades"]),
        str(boot["n_month_blocks"]),
        "2,000 resamples",
        # environment the JSON itself declares
        audit["env"]["sklearn"],
        audit["env"]["pandas"],
        audit["env"]["numpy"],
        audit["env"]["keystats_sha256"][:8],
        audit["env"]["audited_commit"][:7],
    ]
    missing = [tok for tok in required if tok not in md]
    assert not missing, "case page numbers not derivable from JSON: " + str(missing)


def test_scripts_use_the_projects_41_inputs():
    """Regression: the first published version selected df.columns[6:] on a frame read
    without index_col, which silently added SP500_p_change (the index's own forward
    return) as a 42nd input. Both scripts must select inputs by name and assert the
    project's exact 41 inputs."""
    for name in ("audit_mls.py", "boot_mls.py"):
        src = (CASE_DIR / name).read_text(encoding="utf-8")
        assert "len(feat) == 41" in src, f"{name}: missing 41-input assertion"
        assert "'SP500_p_change' not in feat" in src, f"{name}: leaking column not excluded"
        assert "columns[6:]" not in src, f"{name}: positional column selection"


def test_anchor_run_matches_the_projects_own_backtest():
    """anchor_stdout.txt records the author's own backtesting.py run under the same
    random splits as our re-implementation: same input matrix, identical results."""
    out = (CASE_DIR / "anchor_stdout.txt").read_text(encoding="utf-8")
    assert "author 41 | ours 41 | identical matrix: True" in out
    assert "TODAS IGUALES: True" in out
    assert "DIFIERE" not in out


# ---------------------------------------------------------------- case 02
CASE02_DIR = CASES_DIR / "02_stockpredictionmodels"
CASE02_MD = ROOT / "docs" / "cases" / "CASE_02_StockPredictionModels.md"


def load02(name):
    return json.loads((CASE02_DIR / name).read_text(encoding="utf-8"))


def md02_text():
    text = CASE02_MD.read_text(encoding="utf-8")
    return (text.replace("−", "-").replace("–", "-")
            .replace("→", "->").replace("**", ""))


def _usd(x):
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


def _pct(x):
    return f"{x:+.1f}%".replace("+0.0", "-0.0")


def test_case02_files_exist():
    for name in ("audit_sp.py", "sp_published.json", "sp_repro.json",
                 "sp_stochastic.json", "sp_lookahead.json", "sp_oos.json"):
        assert (CASE02_DIR / name).is_file(), f"missing {name}"
    assert CASE02_MD.is_file(), "case page missing under docs/cases/"


def test_case02_repro_all_exact_and_fully_valued():
    """Rule agents reproduce exactly AND every run records mark-to-market."""
    rep = load02("sp_repro.json")
    bad = [k for k, v in rep["agents"].items() if not v["exact"]]
    assert not bad, f"rule agents not reproduced exactly: {bad}"
    for k, v in rep["agents"].items():
        assert v["unsold_units"] is not None, f"{k}: MTM not recorded"
        assert v["total_value"] == v["reproduced_total_balance"] + v["unsold_units"] * v["last_price"]


def test_case02_stochastic_mtm_recorded():
    """Every stochastic/oos run must carry unsold_units + last_price, and
    total_value must equal cash + units * last close."""
    for f in ("sp_stochastic.json", "sp_oos.json"):
        data = load02(f)
        for k, v in data["agents"].items():
            for r in v["runs"]:
                key = "final_balance" if f == "sp_stochastic.json" else "oos_cash"
                assert r["unsold_units"] is not None and r["last_price"] is not None, (f, k, r["seed"])
                assert abs(r["total_value"] - (r[key] + r["unsold_units"] * r["last_price"])) < 1e-6


def test_case02_stochastic_counts_consistent():
    """'cash dips below zero in 3 of 5 seeds' (21) must equal the JSON counter."""
    sto = load02("sp_stochastic.json")
    n21 = sto["agents"]["21-neuro-evolution"]["min_cash_negative_runs"]
    assert n21 == 3, f"page says 3 of 5, JSON says {n21}"
    n22 = sto["agents"]["22-neuro-evolution-novelty-search"]["min_cash_negative_runs"]
    assert n22 == len(sto["agents"]["22-neuro-evolution-novelty-search"]["runs"]), (
        f"page says agent 22 negative in every seed, JSON says {n22}")


def test_case02_page_numbers_match_json():
    pub = load02("sp_published.json")
    rep = load02("sp_repro.json")
    sto = load02("sp_stochastic.json")
    look = load02("sp_lookahead.json")
    oos = load02("sp_oos.json")
    md = md02_text()
    bh = pub["buy_and_hold"]["full"]
    t30 = oos["buy_and_hold"]["test_30"]

    required = [
        # commit + env
        pub["env"]["repo_commit"][:8],
        # buy-and-hold baselines
        _usd(bh),
        _usd(oos["buy_and_hold"]["train_70"]),
        _usd(t30),
        # section 1 — reproduction (cash == MTM: zero unsold shares)
        _usd(pub["agents"]["1-turtle"]["published_total_balance"]),
        _usd(pub["agents"]["2-moving-average"]["published_total_balance"]),
        _usd(pub["agents"]["3-signal-rolling"]["published_total_balance"]),
        _usd(pub["agents"]["23-abcd-strategy"]["published_total_balance"]),
        _usd(pub["agents"]["21-neuro-evolution"]["published_total_balance"]),
        _usd(pub["agents"]["22-neuro-evolution-novelty-search"]["published_total_balance"]),
        _usd(rep["agents"]["23-abcd-strategy"]["reproduced_total_balance"]),
        _pct(100 * (rep["agents"]["1-turtle"]["total_value"] / bh - 1)),
        _pct(100 * (rep["agents"]["2-moving-average"]["total_value"] / bh - 1)),
        _pct(100 * (rep["agents"]["3-signal-rolling"]["total_value"] / bh - 1)),
        _pct(100 * (rep["agents"]["23-abcd-strategy"]["total_value"] / bh - 1)),
        # section 2 — stochastic: cash -> total value
        *(_usd(r["final_balance"]) + "->" + _usd(r["total_value"])
          for r in sto["agents"]["6-evolution-strategy"]["runs"]),
        *(_usd(r["final_balance"]) + "->" + _usd(r["total_value"])
          for r in sto["agents"]["21-neuro-evolution"]["runs"]),
        *(_usd(r["final_balance"]) + "->" + _usd(r["total_value"])
          for r in sto["agents"]["22-neuro-evolution-novelty-search"]["runs"]),
        # in-sample min/max total value (section 4 row labels)
        _usd(min(r["total_value"] for r in sto["agents"]["6-evolution-strategy"]["runs"])),
        _usd(max(r["total_value"] for r in sto["agents"]["6-evolution-strategy"]["runs"])),
        _usd(min(r["total_value"] for r in sto["agents"]["21-neuro-evolution"]["runs"])),
        _usd(max(r["total_value"] for r in sto["agents"]["21-neuro-evolution"]["runs"])),
        _usd(min(r["total_value"] for r in sto["agents"]["22-neuro-evolution-novelty-search"]["runs"])),
        _usd(max(r["total_value"] for r in sto["agents"]["22-neuro-evolution-novelty-search"]["runs"])),
        # section 2 — the leverage bug evidence
        _usd(min(r["min_cash_during_buy"] for r in sto["agents"]["21-neuro-evolution"]["runs"]
                 if r["min_cash_during_buy"] is not None)),
        # section 3 — lookahead (total value, not cash)
        _usd(look["agents"]["2-moving-average"]["total_value"]),
        _usd(look["agents"]["23-abcd-strategy"]["total_value"]),
        _pct(100 * (look["agents"]["2-moving-average"]["total_value"] / bh - 1)),
        _pct(100 * (look["agents"]["23-abcd-strategy"]["total_value"] / bh - 1)),
        # section 4 — oos total-value ranges vs test B&H
        _usd(min(r["total_value"] for r in oos["agents"]["6-evolution-strategy"]["runs"])),
        _usd(max(r["total_value"] for r in oos["agents"]["6-evolution-strategy"]["runs"])),
        _usd(min(r["total_value"] for r in oos["agents"]["21-neuro-evolution"]["runs"])),
        _usd(max(r["total_value"] for r in oos["agents"]["21-neuro-evolution"]["runs"])),
        _usd(min(r["total_value"] for r in oos["agents"]["22-neuro-evolution-novelty-search"]["runs"])),
        _usd(max(r["total_value"] for r in oos["agents"]["22-neuro-evolution-novelty-search"]["runs"])),
        _pct(100 * (min(r["total_value"] for r in oos["agents"]["6-evolution-strategy"]["runs"]) / t30 - 1)),
        _pct(100 * (max(r["total_value"] for r in oos["agents"]["6-evolution-strategy"]["runs"]) / t30 - 1)),
        _pct(100 * (min(r["total_value"] for r in oos["agents"]["21-neuro-evolution"]["runs"]) / t30 - 1)),
        _pct(100 * (max(r["total_value"] for r in oos["agents"]["21-neuro-evolution"]["runs"]) / t30 - 1)),
        _pct(100 * (min(r["total_value"] for r in oos["agents"]["22-neuro-evolution-novelty-search"]["runs"]) / t30 - 1)),
        _pct(100 * (max(r["total_value"] for r in oos["agents"]["22-neuro-evolution-novelty-search"]["runs"]) / t30 - 1)),
        # oos cash ranges quoted in the leverage caveat
        _usd(min(r["oos_cash"] for r in oos["agents"]["22-neuro-evolution-novelty-search"]["runs"])),
        _usd(max(r["oos_cash"] for r in oos["agents"]["22-neuro-evolution-novelty-search"]["runs"])),
    ]
    # M5 (external audit): a bare min/max token check let the wrong in-sample range pass
    # (the true min $13,490.13 appeared elsewhere on the page). Require the full
    # section-4 table row per agent.
    for ag in ("6-evolution-strategy", "21-neuro-evolution", "22-neuro-evolution-novelty-search"):
        tv = [r["total_value"] for r in sto["agents"][ag]["runs"]]
        required.append(f"| {ag.split('-')[0]} | {_usd(min(tv))}…{_usd(max(tv))} |")
    missing = [tok for tok in required if tok not in md]
    assert not missing, "case-02 page numbers not derivable from JSON: " + str(missing)


def test_case02_no_stale_cash_only_claims():
    """The retracted cash-only headlines must not reappear on the page."""
    md = md02_text()
    for banned in ("−54%", "-54%", "−86%", "-86%", "−63.2%", "-63.2%",
                   "−625.9%", "beats the market", "last commit 2024-06"):
        assert banned not in md, f"stale claim still on page: {banned}"
    # the contradiction Claude flagged must not come back either
    for gone in ("none of them beats buy-and-hold", "never beat buy-and-hold to begin with"):
        md = md.replace(gone, "")
    assert "beat buy-and-hold" not in md
