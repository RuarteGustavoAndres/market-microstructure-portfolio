# Plan — Queue-position simulation of maker fills (closing the AUC→P&L gate)

Status: **plan** (2026-10-18). Results will be appended below as separate
dated sections; nothing in this document will be retro-edited.

> **Status update 2026-10-22:** the Background below predates the
> maker-convention fix and the leak-free re-run — its P&L numbers are
> aggressor-perspective and superseded (current arbiter numbers: README,
> 2026-10-22 refresh). The T2/T3 simulation results below remain the best
> queue-aware estimate to date: AUC on simulated fills ~0.50-0.56
> (approximated features, out-of-distribution) → inconclusive, not negative.
> Dated sections are never retro-edited.

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

## Results — T2/T3 executed 2026-10-18

**T2 — maker simulation on real capture** (`sim_maker_dataset.py`: 41 of the
top-50 tokens by labeled fills; 7 skipped; quotes at best bid/ask, size 20;
FIFO queue-ahead from `libro_top5` depth):

- 15,564 rows (fill × horizon): **7,686 filled / 9,380 cancelled / 31
  unfilled** — fill rate 44.6%.
- Raw maker P&L (entry at own quote price; no extra spread credit):
  **+$1,919 @60s / +$2,952 @300s** on 7,782 fill-horizon pairs — positive
  unfiltered, vs ~breakeven-to-negative for the raw taker-fill markout.

**T3 — toxicity filter applied to simulated maker fills**
(`backtest_maker_queue.py`: walk-forward, trained on REAL fills, tested on
simulated maker fills):

- AUC OOS on sim fills: **0.50–0.56 (≈ chance)** — the model trained on real
  fills does not rank simulated maker fills. Expected: out-of-distribution
  test (ofi/momentum in the sim are snapshot-level approximations of the
  event-level features).
- Consequence: the filter adds nothing. P&L: **+$3,999 unfiltered** vs
  +$3,317 (@0.7) / +$3,965 (@0.8) / +$3,998 (@0.9); per-fold optimal $4,015.
- Optimal threshold per fold: [0.65, 0.85, 0.8, 0.9, 0.9] — unstable.

### Gate verdict: **CLOSED**

The rule requires positive OOS P&L **and** threshold stability. The filter
contributes nothing under the queue model (its ranking power does not
transfer to the simulated fills), so the AUC→P&L translation via the filter
remains open.

### Honest reading (what actually changed)

- The maker economics under the queue model are positive **on their own**:
  ~+$4.0k over 7,782 fills ≈ +51¢ per 20-share fill ≈ **+2.6¢/share**, with
  no filter.
- Caveats before treating that as an edge: sample is small (41 tokens);
  bid-side P&L (+$2,655) ≫ ask-side (+$1,344), consistent with period drift
  rather than skill; an always-requoting maker earns the spread mechanically
  in trends; no capital/inventory constraint is modeled.
- Next validation (does NOT change the gate): extend the simulation to all
  266 tokens and the full window, add the capital model, and check bid/ask
  symmetry across sub-periods.
