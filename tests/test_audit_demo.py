"""Tests for demo/audit_demo.py (PASO 4f): the audit's detectors must fire
on the planted-bias trap client and stay quiet on the clean control; the
statistics helpers and the CSV CLI must behave."""
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "demo"))
import audit_demo as ad  # noqa: E402

N_PERM = 99  # small for test speed; effects here are enormous


@pytest.fixture(scope="module")
def cases():
    rng = random.Random(ad.SEED)
    return ad.build_cases(rng, n_days=10)


@pytest.fixture(scope="module")
def audited(cases):
    rng = random.Random(ad.SEED)
    trap_f, m_trap = ad.run_audit(cases["trap"]["booked"], cases["px"],
                                  cases["trap"]["sig"],
                                  half_spread=ad.HALF_SPREAD, rng=rng,
                                  n_perm=N_PERM)
    rng = random.Random(ad.SEED)
    clean_f, m_clean = ad.run_audit(cases["clean"]["booked"], cases["px"],
                                    cases["clean"]["sig"],
                                    half_spread=ad.HALF_SPREAD, rng=rng,
                                    n_perm=N_PERM)
    return trap_f, m_trap, clean_f, m_clean


# ── the core invariant (PASO 4d): the demo must DEMONSTRATE detection ──
def test_trap_fails_every_check(audited):
    trap_f, m_trap, _, _ = audited
    verdicts = {f["check"]: f["verdict"] for f in trap_f}
    assert len(trap_f) == 5
    # capturabilidad may WARN (look-ahead vs fast alpha indistinguishable
    # from prices+trades) — but the trap must never PASS a check
    assert all(v in ("FAIL", "WARN") for v in verdicts.values()), verdicts
    assert verdicts["Capturabilidad del edge (re-ejecución con 1 bar de retraso)"] == "WARN"
    # the trap must look attractive as reported: positive mean, big Sharpe
    assert m_trap["mean"] > 0 and m_trap["sharpe"] > 5


def test_clean_passes_every_check(audited):
    _, _, clean_f, m_clean = audited
    verdicts = {f["check"]: f["verdict"] for f in clean_f}
    assert len(clean_f) == 5
    assert all(v == "PASS" for v in verdicts.values()), verdicts
    # honest plumbing on a no-edge market = spread donor, and it says so
    assert m_clean["mean"] < 0
    assert abs(m_clean["mean"] + 2 * ad.HALF_SPREAD) < 2e-3


def test_demo_self_check_holds(cases, audited):
    trap_f, m_trap, clean_f, m_clean = audited
    ad.self_check(trap_f, clean_f, m_trap, m_clean)  # must not raise


# ── statistics helpers (PASO 4a) ──────────────────────────────────────
def test_perm_test_two_pool_detects_shift():
    rng = random.Random(1)
    a = [rng.gauss(1.0, 1.0) for _ in range(500)]
    b = [rng.gauss(0.0, 1.0) for _ in range(500)]
    p = ad.perm_test_two_pool(a, b, rng, n_perm=N_PERM)
    assert p is not None and p <= 0.05


def test_perm_test_two_pool_null_is_quiet():
    rng = random.Random(2)
    xs = [rng.gauss(0.0, 1.0) for _ in range(500)]
    p = ad.perm_test_two_pool(list(xs), list(xs), rng, n_perm=N_PERM)
    assert p is not None and p > 0.25  # identical pools: nothing to detect


def test_perm_test_mean_detects_and_is_quiet():
    rng = random.Random(3)
    xs = [rng.gauss(0.5, 1.0) for _ in range(500)]
    assert ad.perm_test_mean(xs, 0.0, "greater", rng, N_PERM) <= 0.05
    ys = [rng.gauss(0.0, 1.0) for _ in range(500)]
    assert ad.perm_test_mean(ys, 0.0, "greater", rng, N_PERM) > 0.25


def test_perm_test_too_few_trades_returns_none():
    rng = random.Random(4)
    assert ad.perm_test_mean([1.0, 2.0], 0.0, "greater", rng, N_PERM) is None


def test_tail_symmetry_passes_on_symmetric_sample():
    rng = random.Random(5)
    xs = [rng.gauss(0.0, 1.0) for _ in range(3000)]
    nl, nr, p = ad.tail_symmetry_test(xs)
    assert abs(nl - nr) < 0.5 * max(nl, nr)  # balanced tails
    assert p > 0.01


def test_tail_symmetry_fires_on_left_truncation():
    rng = random.Random(6)
    xs = [rng.gauss(0.0, 1.0) for _ in range(3000)]
    truncated = sorted(xs)[len(xs) // 10:]  # drop the worst decile
    nl, nr, p = ad.tail_symmetry_test(truncated)
    assert nl < nr and p < 0.01


def test_binom_cdf_le_exact_and_approx():
    assert abs(ad.binom_cdf_le(0, 10) - 1 / 1024) < 1e-12
    assert ad.binom_cdf_le(5, 10) > 0.6
    assert ad.binom_cdf_le(-1, 10) == 0.0
    assert 0.0 <= ad.binom_cdf_le(3, 5000) < 0.01  # normal-approx branch (underflow to 0 is fine)


# ── Sharpe overlap handling (PASO 4c) ─────────────────────────────────
def test_nonoverlapping_pnls_extracts_disjoint_trades():
    trades = [(t, 1, 1.0) for t in range(100)]  # every bar, hold 30
    no = ad.nonoverlapping_pnls(trades)
    assert no == [1.0] * 4  # bars 0, 30, 60, 90


def test_metrics_reports_both_sharpes():
    rng = random.Random(7)
    trades = [(t, 1, rng.gauss(0.01, 1.0)) for t in range(300)]
    m = ad.metrics(trades)
    assert {"n", "mean", "sharpe", "sharpe_all", "win"} <= set(m)
    assert m["sharpe_all"] != 0.0


# ── CLI on the client's own CSV (PASO 4e) ─────────────────────────────
@pytest.fixture(scope="module")
def client_csvs(cases, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("client")
    px, sig = cases["px"], cases["trap"]["sig"]
    trades_path = tmp / "trades.csv"
    prices_path = tmp / "prices.csv"
    import csv as _csv
    with open(trades_path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["entry_bar", "exit_bar", "direction",
                    "entry_price", "exit_price"])
        for t, pos, _pnl in cases["trap"]["booked"]:
            w.writerow([t, t + ad.HOLD, pos, px[t], px[t + ad.HOLD]])
    with open(prices_path, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(["bar", "price", "signal"])
        for b in range(len(px)):
            w.writerow([b, px[b], sig[b]])
    return str(trades_path), str(prices_path)


def test_cli_full_audit_fires_on_trap_csv(client_csvs, capsys):
    trades_path, prices_path = client_csvs
    rc = ad.main(["--csv", trades_path, "--prices", prices_path,
                  "--spread", str(ad.HALF_SPREAD), "--n-perm", str(N_PERM)])
    out = capsys.readouterr().out
    assert rc == 0
    # with prices + signal submitted, all five checks run — and all fire
    for check in ("look-ahead", "spread on every round trip",
                  "In-sample parameter tuning", "Random K-fold CV",
                  "Survivorship"):
        assert check in out
    assert out.count("— **FAIL**") == 4
    assert out.count("— **WARN**") == 1  # capturabilidad: reported > delayed but survives
    assert "**PASS**" not in out
    assert "**SKIP**" not in out


def test_cli_without_prices_skips_reexecution_checks(client_csvs, capsys):
    trades_path, _ = client_csvs
    rc = ad.main(["--csv", trades_path, "--n-perm", str(N_PERM)])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.count("**SKIP**") == 4  # checks 1-4 need the price series
    assert "Survivorship" in out


def test_cli_rejects_bad_schema(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text("foo,bar\n1,2\n")
    with pytest.raises(SystemExit):
        ad.main(["--csv", str(bad)])


# ── report ────────────────────────────────────────────────────────────
def test_report_contains_both_cases(cases, audited):
    trap_f, m_trap, clean_f, m_clean = audited
    rep = ad.build_report([
        ("the trap client", "planted", trap_f, m_trap),
        ("the clean control", "honest", clean_f, m_clean),
    ])
    assert "Case 1 — the trap client" in rep
    assert "Case 2 — the clean control" in rep
    assert rep.count("— **FAIL**") == 4
    assert rep.count("— **WARN**") == 1
    assert rep.count("— **PASS**") == 5
