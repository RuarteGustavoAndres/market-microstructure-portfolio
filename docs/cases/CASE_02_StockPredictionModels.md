# Case 02 — Re-auditing a public deep-learning trading-agent zoo

**Audited:** [`huseinzol05/Stock-Prediction-Models`](https://github.com/huseinzol05/Stock-Prediction-Models)
at commit `33266732b0b16188b565e0aeb6b24efa71161f6a` (Apache-2.0 licence; the repo's own README
badge says "MIT"). An educational notebook collection by Husein Zolkepli; last commit 2024-06.
**Question:** 23 notebook trading agents report results on 252 days of GOOG (2016-11-02 →
2017-11-01). Do the published numbers reproduce, and do they survive executable signal timing
and an honest train/test split?
**Note on scope:** the notebooks ship no seeds, no tests and no "how to read this" — this case
measures what the saved outputs claim, without accusing anything the notebooks do not claim.

## What the notebooks report
Final balances saved in the notebooks' own outputs (all agents start with $10,000;
buy-and-hold over the same file is **$13,340.70, +33.4%**):

| Agent | Saved final balance | vs buy-and-hold |
|---|---|---|
| 1 turtle | $10,361.98 | −22.3% |
| 2 moving-average | $10,062.00 | −24.6% |
| 3 signal-rolling | $10,144.97 | −24.0% |
| 6 evolution-strategy | $13,740.72 | +3.0% |
| 21 neuro-evolution | $12,537.79 | −6.0% |
| 22 neuro-evolution-novelty-search | **−$70,156.30** | −625.9% |
| 23 abcd-strategy | $13,054.58 | −2.1% |

## What I did
7 of the 23 agents run in modern Python (numpy 2.5.4, pandas 3.1.0): the code cells are
concatenated verbatim, only dead `%%time`/dead imports are stripped (each transform recorded
per run in the JSONs), and the scripts execute with the notebook's own data file
`dataset/GOOG-year.csv`. The other 16 agents need TensorFlow 1.x (CUDA 10) and are out of
scope here. Nothing in the clone was modified.

### 1. Reproduction (deterministic rule-based agents)

| Agent | Saved | Reproduced | Exact |
|---|---|---|---|
| 1 turtle | $10,361.98 | $10,361.98 | yes |
| 2 moving-average | $10,062.00 | $10,062.00 | yes* |
| 3 signal-rolling | $10,144.97 | $10,144.97 | yes |
| 23 abcd-strategy | $13,054.58 | $13,054.58 | yes |

\* with a one-line compatibility fix: under pandas 3 (copy-on-write) the notebook's chained
assignment `signals['signal'][short_window:] = ...` silently does nothing — the agent makes
**zero trades with no error**. The `.loc` equivalent restores the original behaviour. All four
agents underperform buy-and-hold; reproduction is honest, the edge is simply not there.

### 2. The stochastic agents ship no seeds (6, 21, 22)
Re-runs with pinned seeds (5 runs for 6/21, 3 for 22; `np.random.seed` prepended):

| Agent | Saved | Our runs | Range |
|---|---|---|---|
| 6 | $13,740.72 | $12,539.73 / $13,841.62 / $13,540.14 / $13,840.41 / $13,490.13 | +25.4%…+38.4% |
| 21 | $12,537.79 | $11,815.60 / **−$10,483.90** / $12,032.05 / $11,661.11 / $11,552.89 | −104.8%…+20.3% |
| 22 | −$70,156.30 | **−$181,312.99 / −$71,147.36 / −$79,002.02** | −711%…−1,813% |

The saved numbers are single draws from a wide distribution. Agent 22's published −$70,156
was, if anything, a lucky draw: all three of our seeds are worse or equal. During the final
`buy()` run, agent 21 ends **cash-negative in 3 of 5 seeds** (worst tracked cash −$22,280.72)
and agent 22 in every seed. Root cause (in the code, both agents): the `buy()` gate is
`if starting_money >= trend[t]` — but `starting_money` is **never decremented**; the deduction
hits a different variable (`initial_money`), so the agent keeps buying while broke. Agent 6's
gate is correct (min cash $23.04–$110.06, always positive).

### 3. Look-ahead in the "beats the market" rule agents

| Agent | Saved | With executable timing |
|---|---|---|
| 2 moving-average | $10,062.00 | $9,074.76 (execution delayed 1 day) |
| 23 abcd-strategy | $13,054.58 (+30.5%) | **$3,681.04 (−63.2%)** (signal delayed 12 days) |

Agent 2's buy decision at day `i` uses the moving averages that include `close[i]`, but trades
at `close[i]` — delaying execution one day costs $987. Agent 23's signal for day `a` reads
`ma[b], ma[c], ma[d]` with `b ≥ a, c ≥ b, d ≥ c` (step 4): the a-b-c-d pattern is only knowable
on day `d`, at least 12 days later. Delaying the signal to that day turns the only rule agent
that beat buy-and-hold into a −63% loss. Agents 1 and 3 were checked and trade causally
(1 shifts execution one day; 3's accounting follows the realized next-day price).

### 4. Out-of-sample: train on 70% of the days, evaluate on the last 30%
The evolution agents' fitness **is** the profit on the same series they trade (in-sample by
construction). Holding the trained weights fixed and evaluating on the final 76 days
(train B&H $12,403.02, +24.0%; test B&H $10,756.02, +7.6%):

| Agent | In-sample (ours, seeds) | Out-of-sample |
|---|---|---|
| 6 | $12,539.73…$13,841.62 | $1,436.61…$4,599.12 (B&H test $10,756.02) |
| 21 | $11,552.89…$12,032.05 | −$17,399.61…$8,307.09 |
| 22 | −$71,147.36…−$79,002.02 | −$42,092.26…−$2,155.75 |

Agent 6's +37% becomes −54%…−86% out of sample: the weights encode the traded window, not a
portable signal. 21 keeps trading while broke and stays negative in 2 of 5 seeds.

## Reading
- The rule-based agents reproduce byte-for-byte (with a 1-line pandas-3 fix) and none of them
  beats buy-and-hold — except 23, which needs future prices to do it.
- The stochastic agents' saved balances are single unseeded draws of wide, mostly negative
  distributions; two of them buy while broke because the cash gate watches a variable the
  code never decrements.
- Every +33%-class number in the notebooks is either an in-sample artifact (6, 21, 22) or
  look-ahead (23). With honest timing and an honest split, nothing outperforms buy-and-hold.

## Limits (what this does *not* show)
- 16 of 23 agents (TensorFlow 1.x, CUDA 10) are out of scope; conclusions cover the 7 that
  run in modern Python.
- One year, one ticker (GOOG, a bull year), notebook-era accounting; nothing here is evidence
  about current markets or the author's production work.
- Reproduction of 6/21/22 is statistical by design (no seeds shipped); we pin seeds to make
  *our* runs deterministic, but other draws will land elsewhere in the same ranges.
- The 70/30 split uses the last train day as context for the first test window (77 evaluation
  days, 76 trades); one boundary day of overlap is unavoidable with a 30-day state window.
- `min_cash` is tracked during the final `buy()` only; the fitness phase trades the same way.

## Reproduce
```bash
git clone https://github.com/huseinzol05/Stock-Prediction-Models   # commit 33266732
python3 audit_sp.py /path/to/Stock-Prediction-Models out_dir published   # seconds, no execution
python3 audit_sp.py /path/to/Stock-Prediction-Models out_dir repro       # <10 s, deterministic
python3 audit_sp.py /path/to/Stock-Prediction-Models out_dir lookahead   # <10 s, deterministic
python3 audit_sp.py /path/to/Stock-Prediction-Models out_dir stochastic  # ~8 min CPU
python3 audit_sp.py /path/to/Stock-Prediction-Models out_dir oos         # ~5 min CPU
```
Needs numpy, pandas, matplotlib (imported by the notebooks), scikit-learn (agent 22).
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
