# Plan — Queue-position simulation of maker fills (closing the AUC→P&L gate)

Status: **plan** (2026-10-18). Results will be appended below as separate
dated sections; nothing in this document will be retro-edited.

## Background (measured, not claimed)

The fill-toxicity classifier ranks well but does not yet produce positive net
P&L out-of-sample:

- LightGBM OOF AUC **0.765** (224k fills, 119 unseen markets, re-trained
  2026-10-16). Fresh walk-forward (2026-10-18, 395k labeled fills, 5 temporal
  folds, embargo 300s): OOS AUC 0.71–0.84, threshold-stability improved
  (optimal threshold 0.9 in 4/5 folds).
- OOS P&L (exact, real `mid_futuro`): **−$225 unfiltered** / **−$514 filtered
  @0.7** (2026-10-16 dataset). On the fresh 2026-10-18 dataset: ~breakeven
  (−$1.75 unfiltered, per-fold optimal −$8.57 ceiling) — still no edge.

The segment breakdown (`backtest_desglose.py`) shows where the P&L lives:

| Segment | P&L per op | Verdict |
|---|---|---|
| maker-like fills (raw markout) | +0.85c (54% hit) | positive |
| maker-like + maker credit (½ spread) | **+2.20c** | strongly positive |
| horizon 300s | +0.48c | positive |
| horizons 1s / 10s | −0.31 / −0.18c | negative |
| taker SELL fills | **−0.62c** | the toxic core |
| spread ≥ 2c | +0.29 / +0.78c | positive |
| spread < 2c | −0.23c | toxic |

**Hypothesis:** the current markout-by-fill P&L penalizes fills that a real
maker would only receive if its order was actually at the front of the queue.
A fill-probability model (queue ahead + trade-through volume) is the missing
calibration between "the ranking is good" and "the P&L is good".

## Method — approved assumptions (2026-10-18)

1. **FIFO queue-ahead**: estimated from the depth already at our price level
   when the order is posted (`bid_sz[0]` / `ask_sz[0]` from `libro_top5`).
   No priority information is exposed by Polymarket — FIFO is the standard
   conservative assumption.
2. **Fill = trade-through with sufficient volume**: a maker order fills when
   the trade tape crosses its price and cumulative traded volume at that
   level exceeds the queue ahead. Fill is proportional (volume-scaled), not
   all-or-nothing.
3. **Implicit re-quote**: when the mid moves through our price, the resting
   order is treated as cancelled (conservative; no adverse pick-up).
4. **Costs = adverse inventory cost**: Polymarket CLOB has no trading fee in
   the relevant range; the real cost is inventory developed against informed
   flow. Existing rule from the rewards work: break-even adverse ≥ 2c per
   share, and `income − adverse_cost ≥ 0`.

## Data available (all local, read-only)

- `trades_full` — 382k+ fills with aggressor side and tx_hash.
- `libro_top5` — top-5 book snapshots (~2s cadence) with per-level price/size
  arrays.
- `libro_eventos` — 85M+ order-book level events with best bid/ask.

## Deliverables

- **T1** — `bot/sim_cola.py`: pure, stdlib-only queue-fill simulator
  (same pattern as `binance_book.py`: logic separated from I/O, unit tests
  that use the real protocol, not a friendly variant).
- **T2** — walk-forward of the toxicity filter **combined with the queue
  model** → new gate verdict with realistic maker P&L.
- **T3** — cost calibration → final net P&L per market / horizon.

**Gate rule (unchanged):** the trading gate opens only if the OOS walk-forward
P&L under the queue model is positive AND the threshold is stable across
folds. An AUC improvement alone never reopens it.

## What is intentionally out of scope

- Signals for sale (the portfolio explicitly declares "no edge yet").
- GPU/NN re-training (CPU LightGBM + event-driven simulation is sufficient).
- Binance venue 72h markout rerun (scheduled separately, 2026-10-20).

## Results

_(to be appended after T2/T3 — dated sections, no retro-editing)_
