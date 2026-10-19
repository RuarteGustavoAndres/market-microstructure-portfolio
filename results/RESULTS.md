# Honest results — toxicity filter validation

## CURRENT: leak-free re-run (2026-10-19)

All numbers out-of-sample, **maker perspective** (P&L per fill = −(aggressor
markout)), on the leak-free dataset: **285,764 labeled rows across 215
markets** (freshness-capped snapshots: 136,294 rows dropped; incoming-trade
size features removed; causal standardization fit on train only). Artifacts:
`walkforward_toxicidad.json`, `desglose_walkforward.json`,
`folds_mercado_toxicidad.json`.

| Scheme | Unfiltered | Filtered | Notes |
|---|---|---|---|
| Temporal walk-forward (6 blocks, 300s embargo) | **−$208.84** | @0.7: **+$111.08** | WFE **2.1%** — in-sample P&L decays ~98% OOS; per-fold thresholds [0.6, 0.5, 0.5, 0.5, 0.55] not stable |
| Per-market GroupKFold K=5 (upper bound) | −$117.00 | quantile-0.5: **+$512.34** | **5/5 folds positive**; retention 48.8%; lets future data of other markets into train — read as ceiling, not production estimate |

| Metric | Value |
|---|---|
| AUC pooled OOS | **0.739** (was 0.765 with leaks — horizon-inflated) |
| AUC per horizon | 0.719 − 0.731 (flat: pooled was inflated by horizon) |
| PR-AUC OOS / prevalence | **0.514 / 0.314** |
| Filtered edge by horizon | h≤60s positive; **h300s −$120 (negative)** |
| Filtered edge by segment | taker_sell **+$231** (+0.16¢/fill); maker-like **−$61.5** (only positive with the ½-spread credit: +$76.8) |

**Verdict: the old "no edge" verdict does not survive the leak-free re-run —
the filter reduces tape adverse selection in maker perspective and OOS maker
P&L is positive in both schemes. But WFE 2.1% (severe in-sample→OOS decay),
unstable per-fold thresholds, and missing queue-position/cost models keep the
trading gate CLOSED. The gate closes for real with the queue-aware maker
simulation (PLAN_SIM_COLA.md).**

## History: 2026-10-16 first validation (aggressor perspective, leaky features)

> Superseded by the section above, kept on record. Its P&L numbers measure the
> AGGRESSOR side of each fill and were computed on leaky features (incoming
> trade size, stale snapshots, non-causal scaling) — the 2026-10-19 re-run
> corrects both. We keep it verbatim because publishing our own corrections is
> the product.

All numbers below are out-of-sample, computed on **224,590 labeled fills across
119 markets** captured live 2026-10-14 → 2026-10-16 (Polymarket CLOB, own
24/7 capture).

### Ranking quality (real, improves with data)

| Model | AUC out-of-fold | Samples / markets |
|---|---|---|
| MLP baseline (original 42k dataset) | 0.6425 | 41,828 / 45 |
| LightGBM (original) | 0.688 | 41,828 / 45 |
| **LightGBM (re-trained 2026-10-16)** | **0.765** | 224,520 / 119 |
| MLP (re-trained 2026-10-16) | 0.751 | 224,520 / 119 |

Folds are consistent (0.75–0.79). **But ranking quality does not
translate into net P&L. That is the honest headline.**

### The arbiter: temporal walk-forward with exact markout

Method: contiguous time blocks (6), train only on the past, embargo 300 s
(= max label horizon), P&L per fill = side × (real future mid − fill price),
threshold fixed a-priori (never tuned on the future).

| Aggregated OOS | P&L |
|---|---|
| No filter (always trade) | **-$225** |
| Filter @0.7 | **-$514** (worse) |
| Filter @0.8 | -$337 |
| Per-fold "optimal" threshold (cherry-picked ceiling) | -$199 |
| Optimal threshold per fold | [0.9, 0.9, 0.5, 0.9, 0.5] — unstable |

An earlier backtest (spread-approximated markout, random GroupKFold) showed
the filter "flipping" P&L from -$3,857 to +$361. **That was an artifact** —
exactly the failure mode the walk-forward literature warns about. We keep the
wrong number on record in git history and correct it here.

### Where the loss actually lived (OOS breakdown, aggressor perspective)

| Segment | n | P&L | P&L/fill | Hit |
|---|---|---|---|---|
| taker-like SELL fills (hit the bid) | 165,930 | **-$436** | -0.26¢ | 22% |
| maker-like fills (near mid) | 13,303 | **+$113** | +0.85¢ | 55% |
| taker-like BUY fills | 8,086 | +$97 | +1.20¢ | 73% |
| books with spread < 2¢ | 166,480 | -$256 | -0.15¢ | 24% |
| books with spread 2-5¢ | 18,202 | **+$19** | +0.10¢ | 43% |
| books with spread ≥ 5¢ | 2,637 | **+$12** | +0.44¢ | 49% |
| horizon +300s | 44,505 | **+$110** | +0.25¢ | 36% |
| horizon ≤ 60s | all negative | — | — | — |

Findings (of that run):
1. **The loss was concentrated in taker-side SELL fills in ultra-tight books.**
2. **The filter @0.7 hurt every profitable segment** — later identified as a
   perspective error (maker vs aggressor), not a filter failure.
3. **A fill-based P&L ignores the maker's spread income** — the queue-aware
   maker simulation is the next validation step, not a better classifier.

### Verdict of that run
- Toxicity **ranking**: real signal (AUC 0.765 OOF — later corrected to 0.739 leak-free).
- Toxicity **filter as a trading rule**: **no demonstrated edge yet** — gate
  stays closed. No real order will be placed on it.
- We publish the failure because an honest audit that reports its own negative
  results is the product.
