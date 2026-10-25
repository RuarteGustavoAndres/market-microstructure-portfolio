# P2 — Monte Carlo del auditor (estado al 25-10-2026)

## Qué se corrió
- `python3 demo/monte_carlo_fp_power.py --per-type 1000 --per-intensity 100 --jobs 12 --resume`
- 7 tipos de nulo (4 originales + 3 nuevos: `stop_loss`, `pagos_asimetricos`, `alpha_rapido_real`)
  × 1.000 seeds c/u = **7.000 nulos**; 3 trampas de look-ahead × 100 seeds c/u.
- alpha=0.01, 999 permutaciones/test, seeds deterministas, CI de Wilson 95%
  (fórmula `wilson(k, n)` en el script y en el JSON).
- **~88 min de pared** (5.252,8 s) en 12 núcleos de CPU, 1,3 corridas/s.
- Checkpoint JSONL retomable: `results/mc_fp_power_checkpoint.jsonl` (gitignored).

## Resultados → `results/monte_carlo_fp_power.json`
| detector | FP | CI 95% |
|---|---|---|
| capturabilidad | 0.00% | [0.00, 0.05] |
| fills al book | 0.00% | [0.00, 0.05] |
| tuning in-sample | 0.07% (5/7.000) | [0.03, 0.17] |
| kfold-cv | 0.64% (45/7.000) | [0.48, 0.86] |
| supervivencia | 1.19% (83/7.000) | [0.96, 1.47] |
| **unión** | **1.87% (131/7.000)** | **[1.58, 2.22]** |

Potencia (unión): 100% en las 3 trampas de look-ahead; supervivencia sola 83/100
en la trampa completa; 1-barra: los 5; 50% parcial: capturabilidad 86%.

## Nulos nuevos (FP unión)
- `stop_loss` (declarado → check 5 SKIPEA por diseño): 0.70% [0.34, 1.44]
- `pagos_asimetricos` (NO declarado, cola derecha): 6.10% [4.78, 7.76] —
  casi todo del detector de supervivencia (4.90% [3.73, 6.42]): FP honesto
  de payoff mecánico, bajo el presupuesto del 10%.
- `alpha_rapido_real` (edge = 10% del half-spread): 0.00% — ~1 trade/corrida,
  consistente con la trampa `edge_realo_pequeno`.

## Estado P2
- [x] P2a: MC corrida completa (7.000 nulos, seeds deterministas, ~88 min < 2 h)
- [x] P2b: FP por detector con CI Wilson 95% (script + JSON)
- [x] P2c: 3 nulos nuevos validados y con FP registrado por detector
- [x] P2d: fila del MC actualizada en README.md y README.es.md
- [x] Guard anti-stale extendido: 2 tests nuevos leen los números del MC desde el JSON
- [x] Suite completa: 41 passed
