"""Guard for the pre-registered reserved validation window (2026-11-08).

Audit point 26-10: the final evaluation must ALWAYS report the market-block
bootstrap CI of the pooled delta next to the headline delta — the current CI
[-20.45, +121.74] includes 0, so a positive delta alone would overstate the
edge. Three stdlib-only guards:

1. the dated addendum in docs/VALIDATION_WINDOW.md must keep both bullets
   (no re-training on window data; block-level CI reported at evaluation) and
   must NOT touch the original 2026-10-23 rejection rule;
2. the script that produces the evaluation JSON must emit the block CI
   unconditionally — no flag or env var can skip it;
3. the shipped null-test JSON carries the CI in its verdict line, and the
   future evaluation JSON (results/validacion_reservada_2026-11-08.json),
   once it exists, must carry the same block CI and README line.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WINDOW = ROOT / "docs" / "VALIDATION_WINDOW.md"
SCRIPT = ROOT / "microstructure" / "model" / "backtest_nulo_delta.py"
RESULTS = ROOT / "results"

READMES = {
    "README.md": (ROOT / "README.md").read_text(),
    "README.es.md": (ROOT / "README.es.md").read_text(),
}


def _present(text: str, value: float, nd: int) -> bool:
    s = f"{abs(value):.{nd}f}"
    return any(v in text for v in (s, s.replace(".", ",")))


# --------------------------------------------------- 1. pre-registered text
def test_addendum_keeps_both_bullets():
    doc = WINDOW.read_text()
    assert "Addendum 2026-10-26" in doc, "dated addendum missing from the window doc"
    assert "No re-training on window data" in doc, "no-retrain bullet missing"
    assert "Block-level CI reported at evaluation" in doc, "CI bullet missing"
    assert "descriptive, not part of the rejection rule" in doc, \
        "CI must stay descriptive, outside the rejection rule"


def test_original_rejection_rule_unchanged():
    doc = WINDOW.read_text()
    for phrase in (
        "reject the \"no edge\" null at the 5% level iff",
        "delta pooled ≥ p95 of the random-filter null (null A)",
        "delta > 0 in ≥ 4 of 5 folds",
        "fixed at **0.7**",
    ):
        assert phrase in doc, f"original rule phrase altered or missing: {phrase}"


# --------------------------------------------------- 2. script emits CI always
def test_nulo_b_is_unconditional_in_script():
    src = SCRIPT.read_text()
    assert "nulo_b" in src, "nulo B block missing from the null-test script"
    assert "ic_low_2_5" in src, "block CI low bound missing from the script"
    # the CI must not be skippable: no flag/env can disable it
    for forbidden in ("--skip", "SKIP_NULL", "NULO_B", "RUN_NULL_B"):
        assert forbidden not in src, f"skippable CI guard found: {forbidden}"


def test_verdict_line_carries_ci():
    nulo = json.loads((RESULTS / "nulo_delta_h300.json").read_text())
    ic_low = nulo["nulo_b"]["pooled"]["ic_low_2_5_usd"]
    ic_high = nulo["nulo_b"]["pooled"]["ic_high_97_5_usd"]
    verdict = nulo["veredicto"]
    assert "IC nulo B pooled" in verdict, "verdict must quote the block CI"
    assert _present(verdict, ic_low, 2) and _present(verdict, ic_high, 2), \
        f"verdict CI [{ic_low}, {ic_high}] does not match the JSON's own bounds"


# --------------------------------------------------- 3. evaluation JSON guard
def test_evaluation_json_must_carry_block_ci():
    """When the 2026-11-08 evaluation runs, its published JSON must contain
    the block CI of the pooled delta — and the READMEs must quote it.
    Skipped until the evaluation date; fails CI the moment the file exists
    without the CI."""
    path = RESULTS / "validacion_reservada_2026-11-08.json"
    if not path.exists():
        import pytest
        pytest.skip("evaluation not run yet (2026-11-08)")
    ev = json.loads(path.read_text())
    ic_low = ev["nulo_b"]["pooled"]["ic_low_2_5_usd"]
    ic_high = ev["nulo_b"]["pooled"]["ic_high_97_5_usd"]
    assert "IC nulo B pooled" in ev["veredicto"], \
        "evaluation verdict must quote the block CI"
    for text in READMES.values():
        assert _present(text, ic_low, 2), f"IC low {ic_low} missing from a README"
        assert _present(text, ic_high, 2), f"IC high {ic_high} missing from a README"
