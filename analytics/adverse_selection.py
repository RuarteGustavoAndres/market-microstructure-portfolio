"""Adverse-selection / markout analytics — pure functions, stdlib only.

Computes the markout of aggressive trades against the mid price at +h
seconds: the core execution-quality metric behind the audit service.

    markout_buy(h)  = mid(t+h) - fill_price    (you bought; good if price rises)
    markout_sell(h) = fill_price - mid(t+h)    (you sold; good if price falls)

A trade is TOXIC at horizon h if its markout is negative — the aggressor
systematically loses the spread to whoever fills it. Aggregated over many
trades this is the Loss-Versus-Rebalancing / adverse-selection ratio that
tells a market maker (or an LP in an AMM pool) whether flow is paying or
predatory.

Definitions (deliberate):
  - mid lookup uses the LAST KNOWN mid at or before t+h (no look-ahead: if
    the book is missing at t+h we do not interpolate the future).
  - trades whose horizon extends past the last quote are EXCLUDED and
    counted, never extrapolated.
"""
from bisect import bisect_right

DEFAULT_HORIZONS = (5, 30, 60)


def markouts(trades, mid, horizons=DEFAULT_HORIZONS):
    """trades: iterable of (t_sec, price, aggressor) — aggressor 'buy'|'sell'.
    mid: iterable of (t_sec, mid) sorted ascending by t.
    Returns {horizon: {'n', 'excluded', 'mean', 'buy_mean', 'sell_mean',
                       'toxic_frac'}} with Nones when n == 0."""
    mt = [t for t, _ in mid]
    mv = [v for _, v in mid]
    out = {h: {"n": 0, "excluded": 0, "mean": None, "buy_mean": None,
               "sell_mean": None, "toxic_frac": None} for h in horizons}
    acc = {h: {"all": [], "buy": [], "sell": [], "toxic": 0, "excluded": 0}
           for h in horizons}
    last_t = mt[-1] if mt else None

    for t, price, aggr in trades:
        for h in horizons:
            th = t + h
            if last_t is None or th > last_t:
                acc[h]["excluded"] += 1
                continue
            i = bisect_right(mt, th) - 1          # last mid at or before t+h
            if i < 0:
                acc[h]["excluded"] += 1
                continue
            sign = -1 if aggr == "sell" else 1
            m = sign * (mv[i] - price)
            acc[h]["all"].append(m)
            acc[h][aggr].append(m)
            if m < 0:
                acc[h]["toxic"] += 1

    for h in horizons:
        a = acc[h]
        n = len(a["all"])
        if n == 0:
            out[h]["excluded"] = a["excluded"]
            continue
        out[h].update({
            "n": n,
            "excluded": a["excluded"],
            "mean": sum(a["all"]) / n,
            "buy_mean": sum(a["buy"]) / len(a["buy"]) if a["buy"] else None,
            "sell_mean": sum(a["sell"]) / len(a["sell"]) if a["sell"] else None,
            "toxic_frac": a["toxic"] / n,
        })
    return out


def gp_ratio(values):
    """Gain/Pain ratio: sum(markouts) / sum(|markouts|). >0 = the aggressor
    keeps more than it loses (flow pays the LP); <0 = flow is adversely
    selected. Returns None on empty input."""
    values = list(values or [])
    if not values:
        return None
    denom = sum(abs(v) for v in values)
    return sum(values) / denom if denom else 0.0