"""Anti-stale guard: every headline number in the READMEs must come from a
published JSON in results/, and the retired numbers must stay retired.

The README had two desync incidents (0.688-vs-0.765, 0.7384-without-artifact).
This test makes a third one fail CI instead of shipping. Stdlib only.

Value policy: the README displays 2-4 decimals and uses unicode minus and,
in the Spanish mirror, comma decimals in prose (and comma thousands in
"382,582"). We therefore search for dot and comma variants of each expected
magnitude, taken from the JSON at test time — no numbers are duplicated here.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def _load(name):
    return json.loads((RESULTS / name).read_text())


READMES = {
    "README.md": (ROOT / "README.md").read_text(),
    "README.es.md": (ROOT / "README.es.md").read_text(),
}


def _variants(x: float, nd: int):
    s = f"{abs(x):.{nd}f}"
    return {s, s.replace(".", ",")}


def _present(text: str, value: float, nd: int) -> bool:
    return any(v in text for v in _variants(value, nd))


# ---------------------------------------------------------------- present
def test_auc_pooled_oos_in_headline():
    v = _load("walkforward_toxicidad.json")["auc_pooled_oos"]
    for text in READMES.values():
        assert _present(text, v, 4), f"auc_pooled_oos {v} missing from README"


def test_pr_auc_oos_in_headline():
    v = _load("walkforward_toxicidad.json")["pr_auc_oos"]
    for text in READMES.values():
        assert _present(text, v, 4), f"pr_auc_oos {v} missing from README"


def test_prevalencia_in_headline():
    v = _load("walkforward_toxicidad.json")["prevalencia_oos"]
    for text in READMES.values():
        assert _present(text, v, 3), f"prevalencia {v} missing from README"


def test_h300_walkforward_levels():
    wf = _load("walkforward_toxicidad_h300.json")
    for text in READMES.values():
        for key in ("oos_sin_filtro_usd", "oos_fijo_0.7_usd"):
            assert _present(text, wf[key], 2), f"{key} ({wf[key]}) missing from README"
        delta = wf["oos_fijo_0.7_usd"] - wf["oos_sin_filtro_usd"]
        assert _present(text, delta, 2), f"h300 delta {delta:.2f} missing from README"


def test_nulo_delta_numbers():
    nulo = _load("nulo_delta_h300.json")
    obs = nulo["observado"]["pooled_delta_usd"]
    p95 = nulo["nulo_a"]["p95_usd"]
    ic_low = nulo["nulo_b"]["pooled"]["ic_low_2_5_usd"]
    ic_high = nulo["nulo_b"]["pooled"]["ic_high_97_5_usd"]
    ctrl = nulo["control_c"]["mean_usd"]
    for text in READMES.values():
        for value, nd, label in (
            (obs, 2, "observed delta"),
            (p95, 2, "null p95"),
            (ic_low, 2, "IC low"),
            (ic_high, 2, "IC high"),
            (ctrl, 2, "permuted-label control"),
        ):
            assert _present(text, value, nd), f"{label} ({value}) missing from README"


def test_walkforward_row_counts():
    wf = _load("walkforward_toxicidad.json")
    n_rows, n_oos, n_fills = wf["filas"], wf["n_oos"], wf["fills_distintos_oos"]
    for text in READMES.values():
        # thousands separator: "382,582" / "268,446"
        assert f"{n_rows:,}" in text or f"{n_rows:,}".replace(",", ".") in text, "total rows missing"
        assert f"{n_oos:,}" in text or f"{n_oos:,}".replace(",", ".") in text, "OOS rows missing"
        assert str(n_fills) in text, "distinct fills missing"


# ---------------------------------------------------------------- absent
def test_retired_numbers_stay_retired():
    for text in READMES.values():
        for bad in ("0.7384", "0.506", "0.5057", "0.688", "361", "9,900"):
            assert bad not in text, f"retired number {bad} reappeared in a README"
