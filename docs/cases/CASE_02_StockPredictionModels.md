# Case 02 — Re-auditing a public deep-learning trading-agent zoo

**Audited:** [`huseinzol05/Stock-Prediction-Models`](https://github.com/huseinzol05/Stock-Prediction-Models)
at commit `33266732b0b16188b565e0aeb6b24efa71161f6a` (Apache-2.0 licence). A public
collection of machine-learning stock-forecasting and trading-bot notebooks by Husein Zolkepli;
last commit 2021-01-05.
**Question:** 23 notebook trading agents report results on 252 days of GOOG (2016-11-02 →
2017-11-01). Do the published numbers reproduce, and do they survive executable signal timing
and an honest train/test split?
**Note on scope:** the notebooks ship no seeds, no tests and no "how to read this" — this case
measures what the saved outputs claim, without accusing anything the notebooks do not claim.

## Accounting first: the notebooks' "final balance" is cash only

Every notebook ends with `total_gains = initial_money - starting_money`, where `initial_money`
only moves on buys and sells. **Unsold inventory is never valued.** Buy-and-hold, by
construction, is valued at the final price. Comparing the notebooks' cash-only balance against
buy-and-hold is an apples-to-oranges comparison, and it is the single biggest effect in this
case: every run below therefore also records unsold shares and the last close, and every
comparison uses **total value = cash + unsold shares × last close** (mark-to-market, MTM).
Cash per run is identical to the notebooks' own accounting to the cent.

## What the notebooks report
Final balances saved in the notebooks' own outputs (all agents start with $10,000 — cash only,
unsold shares unvalued; buy-and-hold over the same file is **$13,340.70, +33.4%**):

| Agent | Saved final balance (cash only) |
|---|---|
| 1 turtle | $10,361.98 |
| 2 moving-average | $10,062.00 |
| 3 signal-rolling | $10,144.97 |
| 6 evolution-strategy | $13,740.72 |
| 21 neuro-evolution | $12,537.79 |
| 22 neuro-evolution-novelty-search | −$70,156.30 |
| 23 abcd-strategy | $13,054.58 |

## What I did
7 of the 23 agents run in modern Python (numpy 2.5.4, pandas 3.1.0): the code cells are
concatenated verbatim, only dead `%%time`/dead imports are stripped (each transform recorded
per run in the JSONs), and the scripts execute with the notebook's own data file
`dataset/GOOG-year.csv`. Each executed run additionally prints unsold inventory and the last
close at the end of the final `buy()`. The other 16 agents need TensorFlow 1.x and are out of
scope here. Nothing in the clone was modified.

### 1. Reproduction (deterministic rule-based agents)

| Agent | Saved | Reproduced | Unsold shares | Total value | vs buy-and-hold |
|---|---|---|---|---|---|
| 1 turtle | $10,361.98 | $10,361.98 | 0 | $10,361.98 | −22.3% |
| 2 moving-average | $10,062.00 | $10,062.00 | 0 | $10,062.00 | −24.6% |
| 3 signal-rolling | $10,144.97 | $10,144.97 | 0 | $10,144.97 | −24.0% |
| 23 abcd-strategy | $13,054.58 | $13,054.58 | 0 | $13,054.58 | −2.1% |

\* with a one-line compatibility fix: under pandas 3 (copy-on-write) the notebook's chained
assignment `signals['signal'][short_window:] = ...` silently does nothing — the agent makes
**zero trades with no error**. The `.loc` equivalent restores the original behaviour. All four
agents end with zero unsold shares, so cash and total value coincide, and **none of them beats
buy-and-hold** — agent 23 included, which lands 2.1% below it.

### 2. The stochastic agents ship no seeds (6, 21, 22)
Re-runs with pinned seeds (5 runs for 6/21, 3 for 22; `np.random.seed` prepended):

| Agent | Saved (cash) | Our runs: cash → total value | Total value vs B&H |
|---|---|---|---|
| 6 | $13,740.72 | $12,539.73→$13,565.23 / $13,841.62→$13,841.62 / $13,540.14→$13,540.14 / $13,840.41→$13,840.41 / $13,490.13→$13,490.13 | +1.1%…+3.8% |
| 21 | $12,537.79 | $11,815.60→$11,815.60 / **−$10,483.90→$16,179.10** / $12,032.05→$12,032.05 / $11,661.11→$11,661.11 / $11,552.89→$11,552.89 | −13.4%…+21.3% |
| 22 | −$70,156.30 | **−$181,312.99→$41,220.51 / −$71,147.36→$23,198.64 / −$79,002.02→$25,598.98** | +73.9%…+209.0% |

The saved cash figures are single draws from a wide distribution. Agent 22's published
−$70,156 is almost certainly cash spent into unsold shares rather than a loss — in every
seed we ran the broken gate keeps buying while broke and ends cash-negative — but the saved
run's own inventory is not recoverable, so its mark-to-market value cannot be checked. As
inference from our seeded runs: the same cash behavior valued at the last close lands
between +73.9% and +209.0% over buy-and-hold. **That outperformance is not tradable skill** — both
agents keep buying after their money is gone (see below), so their positions are financed by
leverage the notebooks never account for. Agent 6, whose cash gate is correct, lands
+1.1%…+3.8% above buy-and-hold in-sample — a small edge that, out of sample (section 4),
shrinks to −0.8%…+2.0%.

During the final `buy()` run, agent 21's cash dips below zero in 3 of 5 seeds (worst
−$22,280.72; only seed 43 also ends cash-negative) and agent 22's in every seed (all 3 also
end cash-negative). Root cause (in the code, both agents): the
`buy()` gate is `if starting_money >= trend[t]` — but `starting_money` is **never decremented**;
the deduction hits a different variable (`initial_money`), so the agent keeps buying while
broke. In a bull year this leverage inflates results rather than sinking them. Agent 6's gate
is correct (min cash $23.04–$110.06, always positive).

### 3. Signal timing (rule agents)

| Agent | Saved | With executable timing (total value) |
|---|---|---|
| 2 moving-average | $10,062.00 | $10,100.26 (execution delayed 1 day; 1 unsold share) |
| 23 abcd-strategy | $13,054.58 | $12,910.54 (signal delayed 12 days; 9 unsold shares) |

Agent 2 **executes on the same close it uses as input**: its buy decision at day `i` reads
moving averages that include `close[i]`, and it trades at `close[i]`. An executable version
must trade at the next day's close. Total value after the delay: $10,100.26, −24.3% vs
buy-and-hold — a same-close execution convention (agents 1 and 3 share it), not look-ahead of
future prices, and the strategy
loses to buy-and-hold either way.

Agent 23's signal for day `a` reads `ma[b], ma[c], ma[d]` with `b ≥ a, c ≥ b, d ≥ c` (step 4):
the a-b-c-d pattern is only knowable on day `d`, at least 12 days later — and `d` varies per
signal. What we ran is a **fixed 12-day shift** ($12,910.54 total value, −3.2% vs
buy-and-hold): a sensitivity check, not a causal version — some trades in the shifted run
may still use moving averages not yet knowable at their execution day, so the ~1% effect is
a lower bound on the peeking cost, not a measurement of it. Either way agent 23 never beat
buy-and-hold to begin with. Agents 1, 2 and 3 decide with data up to and including day `i` and
trade at `close[i]` (agent 1's `shift(1)` only excludes today from its breakout bands; it does
not delay execution). None reads future prices, but all three execute at the close they
observe. The 1-day delay above was run only for agent 2, as a sensitivity check; agents 1 and
3 were not delayed.

### 4. Out-of-sample: train on 70% of the days, evaluate on the last 30%
The evolution agents' fitness **is** the profit on the same series they trade (in-sample by
construction). Holding the trained weights fixed and evaluating on the final 76 days
(train B&H $12,403.02, +24.0%; test B&H $10,756.02, +7.6%). The in-sample column comes from
separate runs trained on the full 252-day series, not from the 70% training segment of these
runs. Total values below:

| Agent | In-sample total value (ours, seeds) | Out-of-sample total value | vs test B&H |
|---|---|---|---|
| 6 | $13,490.13…$13,841.62 | $10,666.11…$10,971.46 | −0.8%…+2.0% |
| 21 | $11,552.89…$16,179.10 | $10,263.64…$12,339.89 | −4.6%…+14.7% |
| 22 | $23,198.64…$41,220.51 | $11,175.75…$14,310.24 | +3.9%…+33.0% |

Honest agent 6 (correct cash gate, causal accounting) lands within about ±2% of buy-and-hold out of
sample: **no edge, but no collapse either**. The out-of-sample "collapse" claimed by the
earlier draft was the same cash-only accounting artifact. Agents 21/22 land above test
buy-and-hold only through the same free leverage as above — agent 22's out-of-sample cash is
−$2,155.75…−$42,092.26 while it holds 13–55 unpaid shares.

## Reading
- The rule-based agents reproduce to the cent (with a 1-line pandas-3 fix); all end with
  zero unsold shares and **none beats buy-and-hold** — agent 23 is −2.1%. Its future
  peeking is a real defect, but the fixed-12-day-shift check only bounds its effect from
  below (~1%); that run is not a fully causal version (section 3).
- The notebooks' saved "final balance" ignores unsold inventory; once inventory is valued in
  our seeded runs, the headline negatives (agent 22 "−$70,156", the OOS "collapse") turn into
  leverage-inflated or near-parity results.
- Two of the three stochastic agents buy while broke because the cash gate watches a variable
  the code never decrements; in a bull year that leverage inflates their results. Of the three
  stochastic agents, only agent 6 respects its own cash limit: +1.1%…+3.8% over buy-and-hold
  in-sample and −0.8%…+2.0% out of sample (5 seeds, no significance test) — at most a small
  edge, not distinguishable from buy-and-hold out of sample.

## Limits (what this does *not* show)
- 16 of 23 agents (TensorFlow 1.x) are out of scope; conclusions cover the 7 that
  run in modern Python.
- One year, one ticker (GOOG, a bull year), notebook-era accounting; nothing here is evidence
  about current markets or the author's production work.
- MTM values unsold shares at the last close, with no bid/ask or market impact — the same
  convention as the buy-and-hold baseline, not a claim of realizable value.
- Reproduction of 6/21/22 is statistical by design (no seeds shipped); we pin seeds to make
  *our* runs deterministic, but other draws will land elsewhere in the same ranges.
- The 70/30 split uses the last train day as context for the first test window (77 evaluation
  days, 76 trades); one boundary day of overlap is unavoidable with a 30-day state window.
- `min_cash` is tracked during the final `buy()` only; the fitness phase is different —
  its accounting *does* deduct cash on each trade, while the `buy()` gate checks a variable
  that is never decremented (section 2), so the two phases do not trade the same way.

## Reproduce
```bash
git clone https://github.com/huseinzol05/Stock-Prediction-Models   # commit 33266732
python3 audit_sp.py /path/to/Stock-Prediction-Models out_dir published   # seconds, no execution
python3 audit_sp.py /path/to/Stock-Prediction-Models out_dir repro       # <10 s, deterministic
python3 audit_sp.py /path/to/Stock-Prediction-Models out_dir lookahead   # <10 s, deterministic
python3 audit_sp.py /path/to/Stock-Prediction-Models out_dir stochastic  # ~8 min CPU
python3 audit_sp.py /path/to/Stock-Prediction-Models out_dir oos         # ~5 min CPU
```
Needs numpy, pandas, matplotlib (imported by the notebooks), scikit-learn (agent 22) and
seaborn (imported by some of the notebook cells — the runs fail without it).
Environment: Python 3.12.13, numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1,
matplotlib 3.11.2. Data file: `dataset/GOOG-year.csv`, sha256
`3667ef08ae4dac9e8048d3942c703319bda86298aed9060c19d9bb3d0259cfe6` (252 rows,
GOOG 2016-11-02 → 2017-11-01).
With the seeds recorded in the JSONs, stochastic/oos results are deterministic. Nothing in
the clone is modified. Files: [`audit_sp.py`](../../cases/02_stockpredictionmodels/audit_sp.py),
[`sp_published.json`](../../cases/02_stockpredictionmodels/sp_published.json),
[`sp_repro.json`](../../cases/02_stockpredictionmodels/sp_repro.json),
[`sp_stochastic.json`](../../cases/02_stockpredictionmodels/sp_stochastic.json),
[`sp_lookahead.json`](../../cases/02_stockpredictionmodels/sp_lookahead.json),
[`sp_oos.json`](../../cases/02_stockpredictionmodels/sp_oos.json).
