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
