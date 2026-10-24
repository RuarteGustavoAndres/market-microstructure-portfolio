# MAPA_AUDITORIA.md — guía de lectura para el auditor externo

> Doc de trabajo (2026-10-24), NO pushear sin OK de Gustavo. Lenguaje: ES para
> el auditor (Claude). Cada cifra reclamada por el README se mapea a su archivo
> fuente dentro del repo. Verificado por script (búsqueda numérica en los JSON,
> no grep de texto — los floats no siempre se ven como string).

## Qué es este repo

Portfolio público de un pipeline de microestructura real: captura 24/7 del
libro de Polymarket (y Binance USDT-M perps) a TimescaleDB, clasificador de
toxicidad de fills (LightGBM), validación walk-forward honesta, y un servicio
"backtest-audit" (detectores de sesgos de backtest validados con Monte Carlo).
El producto vendible es la HONESTIDAD de la validación, no un edge de trading.
El gate de trading está CERRADO y así se declara.

## Orden de lectura (7 pasos)

1. **`README.md` (EN, portada)** — qué es, 3 resultados con salvedades, demo
   stdlib-only, arquitectura, tabla de números honestos. Todo el ES espeja esto
   (`README.es.md`, sincronizado). No hay PT a propósito (commit `8dba7fd`:
   audiencia EN/ES).
2. **`results/CAPTURE_STATUS.md`** — estado de captura machine-generated diario
   (regenerado por cron del host de captura; hoy: 159M eventos, 958k fills,
   15.161 mercados, 237 tokens activos). Es LA fuente de verdad de las cifras
   de captura; el README las cita como cotas inferiores ("119M+", "645k+").
3. **`results/MODEL_STATUS.md`** — disciplina de procedencia del modelo: el
   número revisado por humano entra al README; lo viejo queda datado con su
   regla de veredicto (gate cerrado).
4. **`results/RESULTS.md`** — historia de la validación de toxicidad: el run
   19-oct (fill×horizonte) marcado SUPERSEDED 24-oct + el run 16-oct
   (perspectiva agresor, con fugas) kept on record. Publicar las propias
   correcciones ES parte del producto.
5. **`results/*.json`** (fuente de cada cifra — tabla abajo).
6. **`demo/`** — el servicio auditable, stdlib-only:
   `audit_demo.py` (harness + 5 detectores), `audit_controles_sinteticos.py`
   (5 controles sintéticos: 3 honestos + 2 trampas), `monte_carlo_audit.py` /
   `monte_carlo_fp_power.py` (tasa de FP y potencia de los detectores),
   `estudio_limitacion_supervivencia.py` (la falsa alarma conocida de
   spread_sniper, cuantificada).
7. **`microstructure/`** — el pipeline de producción: captura (`capture/`,
   `data/tsdb.py`), modelo walk-forward (`model/`, con JSONs de folds
   incluidos en `model/dataset/`), sim de cola (`sim/sim_cola.py`,
   `sim_maker_dataset.py`), markout Binance (`scripts/markout_binance.py`).
   Tests: `tests/` (3 suites). CI: GitHub Actions (pytest + demo determinística
   + smoke). `make demo` / `make test`.

## Cifra → fuente (verificada por script contra los JSON)

| Cifra (README EN, línea) | Fuente en el repo | Estado |
|---|---|---|
| 119M+ eventos de libro / 645k+ fills / 12,900+ mercados (L15, L47) | `results/CAPTURE_STATUS.md` (hoy 159M / 958k / 15.161 — cotas inferiores vigentes) | OK |
| 237 tokens activos últimos 7 días (L16) | `results/CAPTURE_STATUS.md` "Active tokens last 7 days (exact, indexed)" | OK (corregido 24-oct: antes decía "9,900+ active" — error de transcripción del commit `c11e975`) |
| AUC pooled OOS **0.7384** (L19, L52, L94, L122, L154, L205) | **⚠ SIN JSON** — no está en ningún JSON del repo (búsqueda numérica completa). Rango por fold sí existe: `walkforward_toxicidad.json` folds 0.6894–0.7858 | **GAP — ver abajo** |
| PR-AUC **0.506** / prevalencia **0.304** (L20, L155) | **⚠ SIN JSON** — lo más cercano es el esquema por-mercado: `folds_mercado_toxicidad.json` 0.5135 / 0.3144 (GroupKFold K=5, run 19-oct) | **GAP — ver abajo** |
| Delta +$99.24 GROSS h=300s (L23, L129, L211) | `walkforward_toxicidad_h300.json`: `oos_sin_filtro_usd` −106.75 → `oos_fijo_0.7_usd` −7.51 (delta derivado, aritmética verificada); respaldo directo en `nulo_delta_h300.json` (`reproducibilidad`: ref CPU +99.24; run GPU observado 98.23, diferencia ~1% declarada) | OK |
| IC block-bootstrap [−20.45, +121.74] incluye 0 (L24, L798205c) | `nulo_delta_h300.json` → `nulo_b.pooled.ic_low_2_5_usd` / `ic_high_97_5_usd` | OK |
| Nulo rechazado al 5% (L23) | `nulo_delta_h300.json` → `nulo_a.p95_usd` 45.27 vs delta observado 98.23 (GPU); delta > 0 en 4/5 folds; regla de decisión fijada a priori en el campo `decision_rule` | OK |
| Deltas por horizonte +18.49 / +56.60 / +89.27 / +99.24 (L159) | `walkforward_toxicidad_h{1,10,60,300}.json` (deltas derivados de `oos_sin_filtro_usd` → `oos_fijo_0.7_usd`; verificado: 96.33−77.84 / 157.92−101.32 / 129.89−40.62 / −7.51−(−106.75)) | OK |
| Rangos AUC por fold 0.637–0.740 / 0.682–0.756 / 0.666–0.774 / 0.653–0.787 (L156, L159) | `folds[].auc_oos` de los mismos 4 JSON (verificado exacto) | OK |
| Sim de cola persist: delta +$9.26 GROSS, 2/2 folds, AUC ~0.556 (L140-146) | `walkforward_toxicidad_h300_persist.json` (−15.57 → −6.31; `auc_oos_pool` 0.5563, folds 0.5328–0.5988; 1.580 filas) | OK |
| MC detectores: FP capturabilidad 0% / fills_al_book 0% / tuning 0% / kfold 1.5% / supervivencia 5% / unión 6% (L161) | `monte_carlo_fp_power.json` (`fp_rate_por_detector`, `fp_union` 12/200; alpha 0.01, 999 permutaciones) | OK |
| Auto-auditoría 4h: AUDIT OK, única marca spread_sniper×supervivencia como limitación conocida, 29.2% (7/24) a ambas escalas (L162) | `audit_controles_sinteticos.json` (`veredicto`, `limitaciones_conocidas` con evidencia binomial) + `estudio_limitacion_supervivencia.json` (`resumen.sniper_abs_por_escala` fail_rate 0.2917 a p0=0.5 y p0=100) | OK |
| "AUC inflated 0.736→0.765" / "-$106.75 unfiltered" / WFE −0.8% (L56, L131-132) | historia + `walkforward_toxicidad_h300.json` (`wfe_maker_per_op_pct`) | OK |
| Markout bps multi-símbolo (Venue 2, L179) | `markout_report.json` (`resumen_por_simbolo`, frescura del mid declarada) | OK |
| Números prohibidos 0.688 / $361 | `results/RESULTS.md` L56/L78 SOLO dentro de la sección History etiquetada ("kept on record", "That was an artifact") — patrón permitido por el checklist | OK (gate: 0 hits como cifra actual) |

## GAP declarado (lo que el auditor va a preguntar)

**AUC 0.7384 pooled y PR-AUC 0.506/0.304 no tienen JSON de respaldo en el repo.**
Son del esquema fill×horizonte (382.582 filas, refresh 22-oct), calculados en el
host de captura y volcados a mano en el README — exactamente el patrón
"número sin provenance" que la checklist de publicación (y el antecedente
0.688-vs-0.765) prohíbe. Opciones pendientes de decisión de Gustavo:

- **A. Regenerar**: correr la copia pública de `backtest_walkforward.py` sobre
  la DB real (docker `predikt_timescale`, localhost:5434) extendiendo el script
  para que el JSON incluya `auc_pooled_oos` / `pr_auc_oos` / prevalencia /
  fills distintos. El README queda como está (esas cifras obtienen provenance).
- **B. Reescribir el titular** con cifras ya respaldadas: pooled por-mercado
  0.7356 (GroupKFold, `folds_mercado_toxicidad.json`) + rangos por fold del
  esquema temporal + PR-AUC 0.5135/0.3144. Cero corridas nuevas.

Mientras tanto el auditor debe saber que estas dos cifras del titular no tienen
artefacto de respaldo en `results/` (el resto de la tabla sí, verificado).

## Otras notas para el auditor

- `results/RESULTS.md` sección CURRENT: marcada SUPERSEDED 24-oct (era el run
  19-oct presentado como actual; el README ya narraba el refresh 22-oct).
- MC round 2 (FP con IC Wilson, familias con stop-loss/asimetría/edge-corto y
  curva de potencia vs intensidad q) está corriendo consolidado LOCALMENTE en
  el host de captura (~/.hermes/cache/scratch/mc_round2/round2_resultados.json)
  — aún no está en el repo. Hallazgos preliminares: FP unión 6% [3.5–10.2] en
  200 nulas limpias (confirma el 5% de supervivencia vs 1% nominal, wilson95
  [2.7–9.0]); en nulas con stop-loss la unión dispara ~90% sin flag / 78% con
  flag → problema de FP de los detectores sobre stops, pendiente de resolver
  antes de publicar como "el auditor audita a terceros".
- Git: los docs internos (`UPWORK_PERFIL.md`, `PROPUESTA_MODELO.md`) están
  gitignoreados (check-ignore verificado). `.gitignore` también cubre
  `microstructure/model/dataset/` (los JSON de folds ahí listados son locales).
- Scan de secretos del historial: limpio (59 commits, escaneo 23-oct; solo
  matches de "token_id", identificador de negocio).
- La extensión del dataset (`bot/dataset/ml_features.csv`) NO está en el repo
  (host privado) — provenance sha256 declarado en RESULTS.md + muestra de 300
  filas en `results/dataset_sample.csv`.
- Túnel de reproducibilidad: los JSONs del walk-forward reproducen exacto
  corriendo la copia del repo público sobre la DB real (verificado 23-oct,
  nota `predikt-portfolio-pendientes-v2-claude-2026-10-22`).
