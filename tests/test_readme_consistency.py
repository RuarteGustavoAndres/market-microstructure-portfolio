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


# ---------------------------------------------------------------- about
def _about(text: str) -> str:
    """The About section = everything before the first '## ' heading."""
    return text.split("\n## ", 1)[0]


def test_about_carries_headline_numbers():
    """P4d: the About must show the ranking answer (AUC) AND the economic
    answer (delta with its CI) — sourced from the same JSONs."""
    wf = _load("walkforward_toxicidad_h300.json")
    delta = wf["oos_fijo_0.7_usd"] - wf["oos_sin_filtro_usd"]
    nulo = _load("nulo_delta_h300.json")
    ic_low = nulo["nulo_b"]["pooled"]["ic_low_2_5_usd"]
    ic_high = nulo["nulo_b"]["pooled"]["ic_high_97_5_usd"]
    auc = _load("walkforward_toxicidad.json")["auc_pooled_oos"]
    for text in READMES.values():
        about = _about(text)
        for value, nd, label in (
            (auc, 4, "pooled AUC"),
            (delta, 2, "walk-forward delta"),
            (ic_low, 2, "IC low"),
            (ic_high, 2, "IC high"),
        ):
            assert _present(about, value, nd), f"{label} ({value}) missing from the About"


# ---------------------------------------------------------------- persist
def test_persist_row_numbers():
    """P3c: the queue-sim persist row must carry the JSON's real numbers."""
    pf = _load("walkforward_toxicidad_h300_persist.json")
    delta = pf["oos_fijo_0.7_usd"] - pf["oos_sin_filtro_usd"]
    for text in READMES.values():
        for value, nd, label in (
            (pf["oos_sin_filtro_usd"], 2, "persist unfiltered level"),
            (pf["oos_fijo_0.7_usd"], 2, "persist filtered level"),
            (delta, 2, "persist delta"),
            (pf["auc_oos_pool"], 4, "persist pooled AUC"),
            (pf["wfe_maker_per_op_pct"], 0, "persist WFE per op"),
        ):
            assert _present(text, value, nd), f"{label} ({value}) missing from README"


# ------------------------------------------------------- monte carlo row
def _pct(v: float) -> float:
    """JSON stores rates as fractions; the README row shows percentages."""
    return round(v * 100, 2)


def test_monte_carlo_row_matches_json():
    """P2d: the MC row's FP rates, Wilson CIs and new-null FPs must come
    straight from results/monte_carlo_fp_power.json (rates shown as %)."""
    mc = _load("monte_carlo_fp_power.json")
    pd = mc["fp_rate_por_detector"]
    fu = mc["fp_union"]
    checks = [(_pct(pd[k]["rate"]), 2, f"{k} FP") for k in pd]
    checks += [(_pct(pd[k]["ci95"][0]), 2, f"{k} ci low") for k in ("kfold_cv", "supervivencia")]
    checks += [(_pct(pd[k]["ci95"][1]), 2, f"{k} ci high") for k in ("kfold_cv", "supervivencia")]
    checks += [(_pct(fu["rate"]), 2, "union FP"),
               (_pct(fu["ci95"][0]), 2, "union ci low"),
               (_pct(fu["ci95"][1]), 2, "union ci high")]
    for text in READMES.values():
        assert "7,000" in text or "7.000" in text, "total nulls missing from README"
        for value, nd, label in checks:
            assert _present(text, value, nd), f"MC {label} ({value}) missing from README"


def test_monte_carlo_new_nulls_match_json():
    """P2c: the 3 added nulls (stop-loss declared, asymmetric payoffs NOT
    declared, small real fast alpha) must show the JSON's union FP + CI."""
    mc = _load("monte_carlo_fp_power.json")
    pt = mc["fp_rate_por_tipo"]
    for text in READMES.values():
        for tipo in ("stop_loss", "pagos_asimetricos", "alpha_rapido_real"):
            u = pt[tipo]["union"]
            assert _present(text, _pct(u["rate"]), 2), f"{tipo} union FP missing"
            assert _present(text, _pct(u["ci95"][0]), 2), f"{tipo} CI low missing"
            assert _present(text, _pct(u["ci95"][1]), 2), f"{tipo} CI high missing"
        sv = pt["pagos_asimetricos"]["supervivencia"]
        assert _present(text, _pct(sv["rate"]), 2), "asymmetric-payoff survivorship FP missing"


# ---------------------------------------------------------------- markout
def test_markout_table_matches_json():
    """B1: the markout section (window, trade count, per-symbol bps, BTC
    drift/entry decomposition, staleness) must come straight from
    results/markout_report.json — this section went stale once already."""
    mk = _load("markout_report.json")
    until_date = mk["window"]["until"][:10]  # "2026-10-19"
    total = mk["trades_input_totales"]
    for text in READMES.values():
        assert until_date in text, "markout window end missing from README"
        assert f"{total:,}" in text, "markout total trades missing from README"
        for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
            h5 = mk["resumen_por_simbolo"][sym]["por_horizonte"]["5"]
            h300 = mk["resumen_por_simbolo"][sym]["por_horizonte"]["300"]
            assert _present(text, h5["mean_bps"], 2), f"{sym} mean@5s missing"
            assert _present(text, h5["median_bps"], 2), f"{sym} median@5s missing"
            assert f"{round(h5['toxic_frac'] * 100)}%" in text, f"{sym} toxic frac missing"
            assert _present(text, h300["mean_bps"], 2), f"{sym} mean@300s missing"
        btc5 = mk["resumen_por_simbolo"]["BTCUSDT"]["por_horizonte"]["5"]
        assert _present(text, btc5["drift_bps"], 2), "BTC drift bps missing"
        assert _present(text, btc5["entry_bps"], 2), "BTC entry bps missing"


def test_markout_strict_freshness_matches_json():
    """Claude-audit item 3: the drift/entry headline decomposition was removed
    and replaced by the strict-freshness paragraph (0.5s cap) — every number
    in that paragraph must come straight from
    results/markout_strict_freshness.json (review caught 0.39/0.87 written
    before checking the JSON; artifact says 0.38/0.68)."""
    sf = _load("markout_strict_freshness.json")
    h = {s: sf["resumen_por_simbolo"][s]["por_horizonte"]
         for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT")}
    for text in READMES.values():
        for sym in h:
            assert _present(text, h[sym]["5"]["mean_bps"], 2), \
                f"{sym} strict-freshness mean@5s missing"
            assert _present(text, h[sym]["5"]["drift_bps"], 2), \
                f"{sym} strict-freshness drift@5s missing"
            assert _present(text, h[sym]["5"]["entry_bps"], 2), \
                f"{sym} strict-freshness entry@5s missing"
        assert _present(text, h["BTCUSDT"]["300"]["drift_bps"], 2), \
            "BTC strict-freshness drift@300s missing"
        assert _present(text, sf["frescura_mid_mediana_s"], 2), \
            "strict-freshness median staleness missing"
        # "drops ~78% of fills" — cross-check the share against the JSON
        for sym in h:
            d = h[sym]["5"]["n_descartados_por_frescura"]
            used = h[sym]["5"]["n_usados"]
            share = 100 * d / (d + used)
            assert 77 <= share <= 79, \
                f"{sym} strict-freshness drop is {share:.1f}%, README says ~78%"
        assert "78%" in text, "strict-freshness drop % missing"


# ---------------------------------------------------------------- absent
def test_retired_numbers_stay_retired():
    for text in READMES.values():
        for bad in ("0.7384", "0.506", "0.5057", "0.688", "361", "9,900",
                    "3,111,633", "Sharpe 34"):
            assert bad not in text, f"retired number {bad} reappeared in a README"
