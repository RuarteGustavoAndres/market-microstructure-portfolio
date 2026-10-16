# Honest results — toxicity filter validation (2026-10-16)

All numbers below are out-of-sample, computed on **224,590 labeled fills across
119 markets** captured live 2026-10-14 → 2026-10-16 (Polymarket CLOB, own
24/7 capture). Artifacts: `walkforward_toxicidad.json`, `desglose_walkforward.json`.

## Ranking quality (real, improves with data)

| Model | AUC out-of-fold | Samples / markets |
|---|---|---|
| MLP baseline (original 42k dataset) | 0.6425 | 41,828 / 45 |
| LightGBM (original) | 0.688 | 41,828 / 45 |
| **LightGBM (re-trained 2026-10-16)** | **0.765** | 224,520 / 119 |
| MLP (re-trained 2026-10-16) | 0.751 | 224,520 / 119 |

Folds are consistent (0.75–0.79) — no leakage. **But ranking quality does not
translate into net P&L. That is the honest headline.**

## The arbiter: temporal walk-forward with exact markout

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

## Where the loss actually lives (OOS breakdown)

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

Findings:
1. **The loss is concentrated in taker-side SELL fills in ultra-tight books.**
   The profitable pockets (maker-like fills, wider books, 5-min horizon) are
   real but small.
2. **The filter @0.7 hurts every profitable segment** (e.g. BUY +$98 → -$292;
   +300s +$110 → -$77): a ranker trained on all fills mixed does not know
   which fills a maker actually profits from.
3. **A fill-based P&L ignores the maker's spread income.** Adding the
   half-spread credit to maker-like fills moves the aggregate from -$225 to
   positive (maker-like: +$292) — but the filter still reduces it (+$156).
   The next validation step is a queue-position-aware maker simulation, not a
   better classifier.

## Verdict
- Toxicity **ranking**: real signal (AUC 0.765 OOF, scales with data).
- Toxicity **filter as a trading rule**: **no demonstrated edge yet** — gate
  stays closed. No real order will be placed on it.
- We publish the failure because an honest audit that reports its own negative
  results is the product.
