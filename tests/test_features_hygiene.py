"""Guard: model FEATURES must NOT contain incoming-trade features (maker leak).

fill_size and size_rel_depth describe the trade that lands at t — a resting
maker does not know them when quoting. A past review round claimed they were
removed while the scripts still trained on them; this test makes that
regression impossible (review Claude 2026-10-19).
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE.parent / "microstructure" / "model"
FORBIDDEN = {"fill_size", "size_rel_depth"}


def features_of(path: Path) -> set[str]:
    src = path.read_text(encoding="utf-8")
    m = re.search(r"^FEATURES\s*=\s*\[(.*?)\]", src, re.S | re.M)
    assert m, f"FEATURES list not found in {path.name}"
    return set(re.findall(r'"([a-z_0-9]+)"', m.group(1)))


def test_no_incoming_trade_features_in_any_model_script():
    scripts = sorted(p for p in MODEL_DIR.glob("*.py")
                     if "backtest" in p.name or "entrenar" in p.name)
    assert scripts, f"no model scripts found in {MODEL_DIR}"
    for path in scripts:
        feats = features_of(path)
        leaked = FORBIDDEN & feats
        assert not leaked, (
            f"{path.name}: incoming-trade features {sorted(leaked)} back in "
            "FEATURES — a resting maker cannot know them when quoting"
        )


def test_all_model_scripts_have_features_list():
    scripts = sorted(MODEL_DIR.glob("*.py"))
    trainable = [p for p in scripts if "backtest" in p.name or "entrenar" in p.name]
    assert trainable, "no trainable scripts found"
    for path in trainable:
        feats = features_of(path)
        assert feats, f"{path.name} has an empty FEATURES list"


def test_generator_script_exempt_or_clean():
    """features_ml.py GENERATES the dataset (may emit the columns for
    analysis); if it declares a FEATURES list at all, it must be clean."""
    p = MODEL_DIR / "features_ml.py"
    if not p.exists():
        return
    src = p.read_text(encoding="utf-8")
    if not re.search(r"^FEATURES\s*=\s*\[", src, re.M):
        return
    assert not (FORBIDDEN & features_of(p))
