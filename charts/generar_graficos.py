#!/usr/bin/env python3
"""Deterministic SVG charts for the public portfolio (stdlib only).

Reads the JSON artifacts under results/ and renders two charts under
charts/:

  * delta_por_fold.svg — per-fold P&L delta of the toxicity filter
    (fill x horizon scheme, fixed threshold 0.7), plus the pooled delta
    and the block-bootstrap 95% CI of the h=300s single-horizon scheme,
    clearly labeled as coming from a different scheme.
  * markout_por_horizonte.svg — mean markout (bps of mid) by horizon and
    symbol (Binance USDT-M perps). The per-fill model score is NOT in the
    public data (results/maker_fills_sim_persist.csv has no score column),
    so a score-decile chart is not available; this is the honest
    alternative, labeled as such.

Guarantees:
  * stdlib only (json, pathlib) — no numpy/matplotlib;
  * deterministic: no timestamps, no randomness, fixed iteration order —
    two runs produce byte-identical files (verified by tests/test_charts.py);
  * every number rendered comes from the JSON artifacts; nothing is
    invented (annotations are the artifact values at 2 decimals).
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "results"
CHARTS = REPO / "charts"

INK = "#2f3b45"
GRID = "#d8dde2"
AXIS = "#8a97a3"
BAR_POS = "#5f7d95"
BAR_NEG = "#a06a5f"
CI_FILL = "#9aa5ad"
SYMBOL_COLORS = {
    "BTCUSDT": "#5f7d95",
    "ETHUSDT": "#6b8f71",
    "SOLUSDT": "#a06a5f",
}
FONT = 'font-family="Helvetica, Arial, sans-serif"'


def esc(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


def fmt(x):
    return f"{x:+.2f}"


def text(x, y, s, size=13, anchor="middle", fill=INK, weight="normal",
         transform=None):
    t = f' transform="{transform}"' if transform else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
            f'text-anchor="{anchor}" fill="{fill}" font-weight="{weight}" '
            f'{FONT}{t}>{esc(s)}</text>')


def rect(x, y, w, h, fill, opacity=None, stroke=None):
    o = f' fill-opacity="{opacity}"' if opacity is not None else ""
    st = f' stroke="{stroke}"' if stroke else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'fill="{fill}"{o}{st}/>')


def line(x1, y1, x2, y2, stroke, width=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{width}"{d}/>')


def svg_header(w, h):
    return [
        (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
         f'viewBox="0 0 {w} {h}">'),
        f'<rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff"/>',
    ]


def svg_close():
    return ["</svg>"]


def chart_delta_por_fold():
    """Per-fold P&L delta (fill x horizon scheme) + h=300s block-bootstrap CI."""
    wf = json.loads((RESULTS / "walkforward_toxicidad.json").read_text())
    nulo = json.loads((RESULTS / "nulo_delta_h300.json").read_text())

    deltas = [f["fijo_0.7"]["pnl"] - f["sin_filtro"]["pnl"] for f in wf["folds"]]
    pooled = wf["oos_fijo_0.7_usd"] - wf["oos_sin_filtro_usd"]
    ci_lo = nulo["nulo_b"]["pooled"]["ic_low_2_5_usd"]
    ci_hi = nulo["nulo_b"]["pooled"]["ic_high_97_5_usd"]

    W, H = 880, 520
    x0, x1, ytop, ybot = 80, 840, 80, 400
    ymin, ymax = -100, 420

    def y(v):
        return ybot - (v - ymin) / (ymax - ymin) * (ybot - ytop)

    out = svg_header(W, H)
    out.append(text(W / 2, 30,
                    "Toxicity filter: P&L delta per temporal fold (fixed threshold 0.7)",
                    size=17, weight="bold"))
    out.append(text(W / 2, 52,
                    f"fill x horizon scheme — {wf['filas']:,} rows, "
                    f"{wf['n_oos']:,} OOS (each fill counted once per horizon)",
                    size=12, fill=AXIS))
    # legend row (CI band), above the plot
    out.append(rect(92, 57, 16, 12, CI_FILL, opacity=0.4))
    out.append(text(114, 68,
                    f"block-bootstrap 95% CI (h=300s): [{fmt(ci_lo)}, {fmt(ci_hi)}]",
                    size=12, anchor="start"))

    # grid + y ticks
    for tv in range(ymin, ymax + 1, 100):
        yy = y(tv)
        out.append(line(x0, yy, x1, yy, AXIS if tv == 0 else GRID, 1))
        out.append(text(x0 - 8, yy + 4, str(tv), size=12, anchor="end", fill=AXIS))
    out.append(text(24, (ytop + ybot) / 2, "P&L delta vs no filter (USD)", size=13,
                    transform=f"rotate(-90 24 {(ytop + ybot) / 2:.1f})"))

    # block-bootstrap CI band (h=300s single-horizon scheme — labeled in legend)
    band_top = y(ci_hi)
    band_h = y(ci_lo) - y(ci_hi)
    out.append(rect(x0, band_top, x1 - x0, band_h, CI_FILL, opacity=0.22))
    out.append(line(x0, y(ci_hi), x1, y(ci_hi), CI_FILL, 1.5))
    out.append(line(x0, y(ci_lo), x1, y(ci_lo), CI_FILL, 1.5))

    # pooled delta line (fill x horizon scheme) with OOS P&L of both arms
    py = y(pooled)
    out.append(line(x0, py, x1, py, INK, 1.5, dash="6 4"))
    out.append(text(x0 + 12, py - 8,
                    f"pooled delta (fill x horizon): {fmt(pooled)} — OOS P&L: "
                    f"{wf['oos_sin_filtro_usd']:.2f} (no filter) → "
                    f"{wf['oos_fijo_0.7_usd']:.2f} (fixed 0.7)",
                    size=12, anchor="start"))

    # bars
    n = len(deltas)
    slot = (x1 - x0) / n
    bw = 84
    for i, d in enumerate(deltas):
        cx = x0 + (i + 0.5) * slot
        bx = cx - bw / 2
        if d >= 0:
            by, bh = y(d), y(0) - y(d)
            lab_y = by - 8
        else:
            by, bh = y(0), y(d) - y(0)
            lab_y = by + bh + 16
        out.append(rect(bx, by, bw, max(bh, 0.5), BAR_POS if d >= 0 else BAR_NEG))
        out.append(text(cx, lab_y, fmt(d), size=13, weight="bold"))
        out.append(text(cx, ybot + 20, f"fold {i + 1}", size=12))
    out.append(text((x0 + x1) / 2, ybot + 44, "temporal fold (chronological)",
                    size=13, fill=AXIS))

    # source caption
    out.append(text(16, 470,
                    "Source: results/walkforward_toxicidad.json (per-fold and pooled deltas, "
                    "fill x horizon scheme, fixed threshold 0.7);",
                    size=11, anchor="start", fill=AXIS))
    out.append(text(16, 488,
                    "results/nulo_delta_h300.json (block-bootstrap CI, h=300s single-horizon scheme).",
                    size=11, anchor="start", fill=AXIS))
    out.append(text(16, 506,
                    "Generated by charts/generar_graficos.py — deterministic, stdlib only.",
                    size=11, anchor="start", fill=AXIS))
    out.extend(svg_close())
    return "\n".join(out) + "\n"


def chart_markout_por_horizonte():
    """Mean markout (bps of mid) by horizon and symbol, from markout_report.json."""
    data = json.loads((RESULTS / "markout_report.json").read_text())
    order = data["symbols_usados"]
    rps = data["resumen_por_simbolo"]
    horizons = sorted(int(h) for h in rps[order[0]]["por_horizonte"])
    vals = {s: {int(h): rps[s]["por_horizonte"][h]["mean_bps"]
                for h in rps[s]["por_horizonte"]} for s in order}
    win = data["window"]
    desde = win["since"].replace("T", " ").replace("Z", " UTC")
    hasta = win["until"].replace("T", " ").replace("Z", " UTC")
    n_trades = data["trades_input_totales"]

    W, H = 880, 540
    x0, x1, ytop, ybot = 80, 840, 90, 400
    ymin, ymax = -0.4, 0.6

    def y(v):
        return ybot - (v - ymin) / (ymax - ymin) * (ybot - ytop)

    out = svg_header(W, H)
    out.append(text(W / 2, 30,
                    "Mean markout by horizon and symbol (Binance USDT-M perps)",
                    size=17, weight="bold"))
    out.append(text(W / 2, 52,
                    "markout in bps of mid; positive = taker flow profitable at that horizon, "
                    "negative = adverse",
                    size=12, fill=AXIS))

    # legend
    lx = x1 - 360
    for i, s in enumerate(order):
        out.append(rect(lx + i * 130, 62, 14, 14, SYMBOL_COLORS[s]))
        out.append(text(lx + i * 130 + 20, 74, s, size=12, anchor="start"))

    # grid + y ticks
    for tv in (-0.4, -0.2, 0.0, 0.2, 0.4, 0.6):
        yy = y(tv)
        out.append(line(x0, yy, x1, yy, AXIS if tv == 0 else GRID, 1))
        lab = "0" if tv == 0 else f"{tv:+.1f}"
        out.append(text(x0 - 8, yy + 4, lab, size=12, anchor="end", fill=AXIS))
    out.append(text(24, (ytop + ybot) / 2, "mean markout (bps of mid)", size=13,
                    transform=f"rotate(-90 24 {(ytop + ybot) / 2:.1f})"))

    # grouped bars
    n = len(horizons)
    slot = (x1 - x0) / n
    bw, gap = 48, 8
    total_w = len(order) * bw + (len(order) - 1) * gap
    for gi, hs in enumerate(horizons):
        gcx = x0 + (gi + 0.5) * slot
        start = gcx - total_w / 2
        for si, s in enumerate(order):
            v = vals[s][hs]
            bx = start + si * (bw + gap)
            if v >= 0:
                by, bh = y(v), y(0) - y(v)
                lab_y = by - 6
            else:
                by, bh = y(0), y(v) - y(0)
                lab_y = by + bh + 14
            out.append(rect(bx, by, bw, max(bh, 0.5), SYMBOL_COLORS[s]))
            out.append(text(bx + bw / 2, lab_y, fmt(v), size=11))
        out.append(text(gcx, ybot + 20, f"{hs}s", size=12))
    out.append(text((x0 + x1) / 2, ybot + 44, "horizon", size=13, fill=AXIS))

    # source caption
    out.append(text(16, 470,
                    f"Source: results/markout_report.json — capture window {desde} → {hasta}, "
                    f"{n_trades:,} input trades.",
                    size=11, anchor="start", fill=AXIS))
    out.append(text(16, 490,
                    "Per-fill model scores are not in the public data, so a score-decile chart is "
                    "not available;",
                    size=11, anchor="start", fill=AXIS))
    out.append(text(16, 508,
                    "This chart shows markout by horizon and symbol. Generated by "
                    "charts/generar_graficos.py — deterministic, stdlib only.",
                    size=11, anchor="start", fill=AXIS))
    out.extend(svg_close())
    return "\n".join(out) + "\n"


def main():
    CHARTS.mkdir(exist_ok=True)
    (CHARTS / "delta_por_fold.svg").write_text(chart_delta_por_fold(),
                                               encoding="utf-8")
    (CHARTS / "markout_por_horizonte.svg").write_text(
        chart_markout_por_horizonte(), encoding="utf-8")
    print("wrote charts/delta_por_fold.svg and charts/markout_por_horizonte.svg")


if __name__ == "__main__":
    main()
