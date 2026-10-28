"""Every JSON artifact shipped in results/ must be valid JSON.

Regression for 2026-10-28: a shell-wrapper line (EXIT=0) was appended to
results/markout_entry_lag_probe.json by a mis-directed redirect and no test
caught it — any consumer doing json.load() on a published artifact would
crash. This test parses every results/*.json so wrapper noise can never ship
again.
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def test_every_results_json_parses():
    files = sorted(RESULTS_DIR.glob("*.json"))
    assert files, "no JSON artifacts found in results/ — check the path"
    bad = []
    for path in files:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - report the actual parse error
            bad.append(f"{path.name}: {exc}")
    assert not bad, "invalid JSON artifacts in results/: " + "; ".join(bad)


def test_lag_probe_grid_encloses_minimum():
    """The +2s grid extension must keep the entry-floor minimum enclosed."""
    data = json.loads(
        (RESULTS_DIR / "markout_entry_lag_probe.json").read_text(encoding="utf-8")
    )
    assert data["offsets_s"] == [-0.5, -0.3, -0.1, 0.0, 0.1, 0.3, 0.5, 1.0, 2.0]
    btc = data["resumen_por_simbolo"]["BTCUSDT"]["por_offset_s"]
    medians = {k: v["all"]["median_bps"] for k, v in btc.items()}
    # median collapses to ~half-spread (0.006) by +0.5s and turns positive at +2s
    assert abs(medians["0.0"] - (-0.28020673294942255)) < 1e-9
    assert abs(medians["0.5"] - (-0.006260960594504553)) < 1e-9
    assert medians["2.0"] > 0.0
    # mean at +1s is still negative (zero-crossing is between +1s and +2s)
    assert btc["1.0"]["all"]["mean_bps"] < 0.0 < btc["2.0"]["all"]["mean_bps"]
