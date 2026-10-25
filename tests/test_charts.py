"""Tests de charts/generar_graficos.py — SVG deterministas y fieles a los JSON.

Verifica:
  1. determinismo: dos corridas del generador producen archivos byte-idénticos;
  2. los dos SVG son XML parseable (xml.etree);
  3. los valores anotados en cada SVG coinciden con los artefactos de results/.

Corre con: python3 -m pytest tests/ -q  (stdlib only, sin dependencias)
"""
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARTS = os.path.join(ROOT, "charts")
RESULTS = os.path.join(ROOT, "results")
GENERATOR = os.path.join(CHARTS, "generar_graficos.py")

SVG_DELTA = os.path.join(CHARTS, "delta_por_fold.svg")
SVG_MARKOUT = os.path.join(CHARTS, "markout_por_horizonte.svg")


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _run_generator():
    r = subprocess.run([sys.executable, GENERATOR],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, f"generador falló: {r.stderr}"


def _svg_texts(path):
    root = ET.parse(path).getroot()
    return [el.text or "" for el in root.iter() if el.tag.endswith("text")]


def _fmt(v):
    return f"{v:+.2f}"


def test_generator_determinista():
    _run_generator()
    h1 = (_sha(SVG_DELTA), _sha(SVG_MARKOUT))
    _run_generator()
    h2 = (_sha(SVG_DELTA), _sha(SVG_MARKOUT))
    assert h1 == h2, "los SVG no son byte-idénticos entre corridas"


def test_svgs_parseables_como_xml():
    for p in (SVG_DELTA, SVG_MARKOUT):
        assert os.path.exists(p), f"falta {p} (correr charts/generar_graficos.py)"
        root = ET.parse(p).getroot()
        assert root.tag.endswith("svg"), f"{p}: raíz no es <svg>"
        assert int(root.get("width", "0")) > 0 and int(root.get("height", "0")) > 0


def test_delta_por_fold_valores_coinciden_con_json():
    wf = json.load(open(os.path.join(RESULTS, "walkforward_toxicidad.json")))
    nulo = json.load(open(os.path.join(RESULTS, "nulo_delta_h300.json")))
    texts = " ".join(_svg_texts(SVG_DELTA))

    # delta por fold = fijo_0.7 - sin_filtro
    for f in wf["folds"]:
        delta = f["fijo_0.7"]["pnl"] - f["sin_filtro"]["pnl"]
        assert _fmt(delta) in texts, f"delta del fold {f['fold']} no está en el SVG"

    # delta pooled y P&L OOS de ambos brazos
    pooled = wf["oos_fijo_0.7_usd"] - wf["oos_sin_filtro_usd"]
    assert _fmt(pooled) in texts
    assert f"{wf['oos_sin_filtro_usd']:.2f}" in texts
    assert f"{wf['oos_fijo_0.7_usd']:.2f}" in texts

    # IC del nulo B (esquema h=300s) — debe estar etiquetado como h=300s
    ci_lo = nulo["nulo_b"]["pooled"]["ic_low_2_5_usd"]
    ci_hi = nulo["nulo_b"]["pooled"]["ic_high_97_5_usd"]
    assert f"[{_fmt(ci_lo)}, {_fmt(ci_hi)}]" in texts
    assert "h=300s" in texts, "el IC debe estar etiquetado como esquema h=300s"


def test_markout_por_horizonte_valores_coinciden_con_json():
    data = json.load(open(os.path.join(RESULTS, "markout_report.json")))
    texts = " ".join(_svg_texts(SVG_MARKOUT))

    rps = data["resumen_por_simbolo"]
    for simbolo in data["symbols_usados"]:
        assert simbolo in texts, f"el símbolo {simbolo} no está en el SVG"
        for h, row in rps[simbolo]["por_horizonte"].items():
            assert f"{row['mean_bps']:+.2f}" in texts, \
                f"mean_bps de {simbolo} h={h}s no está en el SVG"

    # la ventana de captura declarada en el JSON debe figurar en el pie
    win = data["window"]
    assert win["since"][:10] in texts and win["until"][:10] in texts
