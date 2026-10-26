"""Tests de analytics/adverse_selection.py — markout puro, sin I/O.

Corre con: python -m pytest tests/ -q
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analytics.adverse_selection import markouts, gp_ratio


# mid: sube 0.01 por segundo, de 100 a 100+2*N/100
MID = [(t, 100.0 + t / 100.0) for t in range(0, 400)]

def test_compra_en_tendencia_alcista_gana():
    # compro a 100.0 en t=0; mid a +30s = 100.30 -> markout +0.30
    r = markouts([(0, 100.0, "buy")], MID, horizons=(30,))
    assert abs(r[30]["mean"] - 0.30) < 1e-9
    assert r[30]["toxic_frac"] == 0.0
    assert r[30]["n"] == 1

def test_venta_en_tendencia_alcista_pierde():
    # vender a 100.0 en t=0; mid a +60s = 100.60 -> markout -0.60
    r = markouts([(0, 100.0, "sell")], MID, horizons=(60,))
    assert abs(r[60]["mean"] - (-0.60)) < 1e-9
    assert r[60]["toxic_frac"] == 1.0

def test_compra_y_venta_simetricas_en_marca():
    # mid plano en 100: compra y venta a mid -> markout 0 exacto, no tóxico
    flat = [(t, 100.0) for t in range(0, 400)]
    r = markouts([(10, 100.0, "buy"), (10, 100.0, "sell")], flat, horizons=(5,))
    assert abs(r[5]["mean"]) < 1e-12
    assert r[5]["buy_mean"] == 0.0 and r[5]["sell_mean"] == 0.0
    assert r[5]["n"] == 2 and r[5]["excluded"] == 0

def test_horizon_pasado_del_final_se_excluye_sin_extrapolacion():
    # trade en t=395 con horizonte 30: el mid llega solo hasta 399 -> excluido
    r = markouts([(395, 100.0, "buy")], MID, horizons=(30,))
    assert r[30]["n"] == 0 and r[30]["excluded"] == 1
    assert r[30]["mean"] is None

def test_mid_interpolacion_sin_lookahead():
    # mid con hueco: el último mid ANTES de t+h se usa (no el posterior)
    gap = [(0, 100.0), (10, 101.0), (100, 200.0)]
    r = markouts([(0, 100.0, "buy")], gap, horizons=(20,))
    # en t=20 el último mid conocido es el de t=10 (101.0) -> +1.0
    assert abs(r[20]["mean"] - 1.0) < 1e-9

def test_sin_mids_todo_excluido():
    r = markouts([(0, 100.0, "buy")], [], horizons=(5,))
    assert r[5]["n"] == 0 and r[5]["excluded"] == 1

def test_gp_ratio():
    assert gp_ratio([]) is None
    assert gp_ratio([1.0, -1.0]) == 0.0            # 0/2
    assert abs(gp_ratio([2.0, -1.0]) - (1.0 / 3.0)) < 1e-9
    assert abs(gp_ratio([-2.0, 1.0]) - (-1.0 / 3.0)) < 1e-9
    assert gp_ratio([0.0, 0.0]) == 0.0             # denominador 0 -> 0.0