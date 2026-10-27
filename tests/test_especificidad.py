"""Control de ESPECIFICIDAD (task_3): un cliente honesto con edge real chico
POSITIVO neto de costos y >= 200 trades NO debe ser marcado por el auditor.

Antecedente: el nulo alpha_rapido_real original casi no opera (815/1000
corridas del MC con 0 trades; las demas PIERDEN el spread) — no probaba
nada. edge_neto_positivo es el control que falta: edge neto ~ +30% del
costo de round trip, fills declarados netos (--fills-net-of-costs), plumbing
honesto de punta a punta. Si el audit lo marca, acusa por error a un
cliente legítimo.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "demo"))
import audit_demo as ad  # noqa: E402
import monte_carlo_fp_power as mc  # noqa: E402

RESULTADOS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "results", "monte_carlo_fp_power.json")


@pytest.fixture
def mc_chico(monkeypatch):
    """MC con mercado de 20 dias (rapido); la logica es identica."""
    monkeypatch.setattr(ad, "N_DAYS", 20)
    return mc


def _spec_job(i):
    return ("null", "edge_neto_positivo", i, mc.NULL_SEED_BASE + 7000 + i)


def test_operan_suficiente_y_edge_neto_positivo(mc_chico):
    """Cada corrida reporta >= 200 trades y el edge neto agregado es > 0."""
    means, ns = [], []
    for i in range(3):
        r = mc_chico.run_one(_spec_job(i), 99)
        assert r["group"] == "null" and r["kind"] == "edge_neto_positivo"
        ns.append(r["n"])
        means.append(r["mean"])
    assert min(ns) >= mc.SPEC_MIN_TRADES
    assert sum(means) > 0.0


def test_auditor_no_marca_cliente_rentable_honesto(mc_chico):
    """run_audit completo sobre el cliente honesto rentable: todo PASS
    (SKIP en fills_al_book por fills netos declarados)."""
    r = mc_chico.run_one(_spec_job(0), 99)
    assert r["verdicts"] == "PSPPP", r["verdicts"]
    assert not any(r["flags"])
    assert r["n"] >= mc.SPEC_MIN_TRADES


def test_json_especificidad_consistente():
    """El artefacto commiteado lleva el bloque de especificidad con el piso
    de trades cumplido en todas las corridas y FP dentro del presupuesto."""
    if not os.path.exists(RESULTADOS):
        pytest.skip("artefacto results/monte_carlo_fp_power.json no existe")
    art = json.load(open(RESULTADOS))
    spec = art.get("especificidad")
    assert spec is not None, "falta el bloque 'especificidad' en el JSON"
    assert spec["n_runs"] == art["params"]["n_null_per_type"]
    assert spec["runs_ge_min"] == spec["n_runs"]
    assert spec["min_trades"] >= spec["min_trades_declarado"]
    assert spec["mean_net_per_trade"] > 0.0
    assert spec["flag_rate"] <= 0.10


def test_check4_cv_temporal_sin_bloque_sin_historia(mc_chico):
    """El CV temporal del audit ya no opera el bloque 0 con umbral neutral:
    sin historia de tuning no hay trades, y m_cv no queda sistematicamente
    por debajo del reporte de un cliente rentable (regresion del fix).

    Dientes verificados contra el codigo viejo (git show HEAD): con esta
    semilla el bloque 0 con thr=1.0 hacia FAIL en check 4 sobre este cliente
    honesto (semillas i=7,16,21 de 24; i=7 la menor) — con el fix, PASS.
    """
    mc_chico.run_one(_spec_job(1), 99)          # no debe lanzar
    n = ad.N_DAYS * 60 * 24
    s_eps = mc.SPEC_S_EPS_FRAC * ad.SIGMA
    sigma_z = (ad.SIGMA ** 2 - s_eps ** 2) ** 0.5
    px, eps = mc.build_market_alpha(n, ad.P0, s_eps, sigma_z,
                                    mc.NULL_SEED_BASE + 7007)
    sig = [eps[t + 2] / s_eps if t + 2 < n else 0.0 for t in range(n)]
    booked, sig, thr = mc.build_null_client(px, sig,
                                            half_spread=mc.SPEC_HALF_SPREAD)
    import random
    f, m = ad.run_audit(booked, px, sig, half_spread=mc.SPEC_HALF_SPREAD,
                        rng=random.Random(1007), n_perm=99,
                        fills_net_of_costs=True)
    kf = [x for x in f if "K-fold" in x["check"]][0]
    assert kf["verdict"] == "PASS", kf["detail"]
    # el cliente honesto no debe ser acusado por NINGUN check
    assert all(x["verdict"] in ("PASS", "SKIP") for x in f), \
        [x["check"] for x in f if x["verdict"] not in ("PASS", "SKIP")]
