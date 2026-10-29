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
                 "mls_bootstrap.json"):
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


# ---------------------------------------------------------------- case 02
CASE02_DIR = CASES_DIR / "02_stockpredictionmodels"
CASE02_MD = ROOT / "docs" / "cases" / "CASE_02_StockPredictionModels.md"


def load02(name):
    return json.loads((CASE02_DIR / name).read_text(encoding="utf-8"))


def md02_text():
    text = CASE02_MD.read_text(encoding="utf-8")
    return text.replace("−", "-").replace("–", "-").replace("**", "")


def _usd(x):
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


def test_case02_files_exist():
    for name in ("audit_sp.py", "sp_published.json", "sp_repro.json",
                 "sp_stochastic.json", "sp_lookahead.json", "sp_oos.json"):
        assert (CASE02_DIR / name).is_file(), f"missing {name}"
    assert CASE02_MD.is_file(), "case page missing under docs/cases/"


def test_case02_page_numbers_match_json():
    pub = load02("sp_published.json")
    rep = load02("sp_repro.json")
    sto = load02("sp_stochastic.json")
    look = load02("sp_lookahead.json")
    oos = load02("sp_oos.json")
    md = md02_text()

    required = [
        # commit + env
        pub["env"]["repo_commit"][:8],
        # buy-and-hold baselines
        _usd(pub["buy_and_hold"]["full"]),
        _usd(oos["buy_and_hold"]["train_70"]),
        _usd(oos["buy_and_hold"]["test_30"]),
        # section 1 — published + reproduced (rules)
        _usd(pub["agents"]["1-turtle"]["published_total_balance"]),
        _usd(pub["agents"]["2-moving-average"]["published_total_balance"]),
        _usd(pub["agents"]["3-signal-rolling"]["published_total_balance"]),
        _usd(pub["agents"]["21-neuro-evolution"]["published_total_balance"]),
        _usd(pub["agents"]["22-neuro-evolution-novelty-search"]["published_total_balance"]),
        _usd(pub["agents"]["23-abcd-strategy"]["published_total_balance"]),
        _usd(rep["agents"]["2-moving-average"]["reproduced_total_balance"]),
        _usd(rep["agents"]["23-abcd-strategy"]["reproduced_total_balance"]),
        # section 2 — stochastic runs (seed order 42, 43, ...)
        _usd(sto["agents"]["6-evolution-strategy"]["runs"][0]["final_balance"]),
        _usd(sto["agents"]["6-evolution-strategy"]["runs"][1]["final_balance"]),
        _usd(sto["agents"]["21-neuro-evolution"]["runs"][1]["final_balance"]),
        _usd(sto["agents"]["21-neuro-evolution"]["runs"][2]["final_balance"]),
        _usd(sto["agents"]["21-neuro-evolution"]["runs"][0]["final_balance"]),
        _usd(sto["agents"]["21-neuro-evolution"]["runs"][3]["final_balance"]),
        _usd(sto["agents"]["21-neuro-evolution"]["runs"][4]["final_balance"]),
        _usd(sto["agents"]["22-neuro-evolution-novelty-search"]["runs"][0]["final_balance"]),
        _usd(sto["agents"]["22-neuro-evolution-novelty-search"]["runs"][1]["final_balance"]),
        _usd(sto["agents"]["22-neuro-evolution-novelty-search"]["runs"][2]["final_balance"]),
        _usd(sto["agents"]["21-neuro-evolution"]["runs"][1]["min_cash_during_buy"]),
        f"3 of 5 seeds",  # cross-checked below against min_cash_negative_runs
        # section 3 — lookahead
        _usd(look["agents"]["2-moving-average"]["delayed_total_balance"]),
        _usd(look["agents"]["23-abcd-strategy"]["delayed_total_balance"]),
        # section 4 — oos min/max across runs
        _usd(min(r["oos_balance"] for r in oos["agents"]["6-evolution-strategy"]["runs"])),
        _usd(max(r["oos_balance"] for r in oos["agents"]["6-evolution-strategy"]["runs"])),
        _usd(min(r["oos_balance"] for r in oos["agents"]["21-neuro-evolution"]["runs"])),
        _usd(max(r["oos_balance"] for r in oos["agents"]["21-neuro-evolution"]["runs"])),
        _usd(min(r["oos_balance"] for r in oos["agents"]["22-neuro-evolution-novelty-search"]["runs"])),
        _usd(max(r["oos_balance"] for r in oos["agents"]["22-neuro-evolution-novelty-search"]["runs"])),
    ]
    missing = [tok for tok in required if tok not in md]
    assert not missing, "case-02 page numbers not derivable from JSON: " + str(missing)


def test_case02_stochastic_counts_consistent():
    """'cash-negative in 3 of 5 seeds' (21) must equal the JSON counter."""
    sto = load02("sp_stochastic.json")
    n21 = sto["agents"]["21-neuro-evolution"]["min_cash_negative_runs"]
    assert n21 == 3, f"page says 3 of 5, JSON says {n21}"
    n22 = sto["agents"]["22-neuro-evolution-novelty-search"]["min_cash_negative_runs"]
    assert n22 == len(sto["agents"]["22-neuro-evolution-novelty-search"]["runs"]), (
        f"page says agent 22 negative in every seed, JSON says {n22}")


def test_case02_repro_all_exact():
    rep = load02("sp_repro.json")
    bad = [k for k, v in rep["agents"].items() if not v["exact"]]
    assert not bad, f"rule agents not reproduced exactly: {bad}"

