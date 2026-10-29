# Microestructura de mercado: captura 24/7 + modelo de toxicidad de fills (honesto)

> [EN](README.md) | **ES**

[![CI](https://github.com/RuarteGustavoAndres/market-microstructure-portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/RuarteGustavoAndres/market-microstructure-portfolio/actions/workflows/ci.yml)

Pipeline de microestructura de mercado 24/7 (CLOB de Polymarket + perpetuos
USDT-M de Binance): captura multi-venue del libro por WebSocket a TimescaleDB,
más un modelo de ML sin fugas que predice la **toxicidad de fills (selección
adversa)** desde features del libro. Todos los números son reales y
out-of-sample — **incluidas las partes que no funcionaron**. El detalle
denso por fila vive en [RESULTS.md](docs/RESULTS.md); los README son índice +
titulares.

**Tres resultados, con su salvedad**

1. **159M+ eventos de libro y 958k+ fills reales** capturados 24/7 a
   TimescaleDB (15,161+ mercados trackeados, 237 tokens activos en los últimos
   7 días). Estado en vivo, generado a diario:
   [`results/CAPTURE_STATUS.md`](results/CAPTURE_STATUS.md).
2. **Clasificador de toxicidad: AUC 0.7276 OOS pooled, walk-forward temporal
   (sin fugas); PR-AUC 0.4963 vs prevalencia 0.304** — pooled sobre el
   esquema fill×horizonte (382,582 filas); el esquema por mercado puntúa 0.7356
   (ver docs/RESULTS.md, "AUC provenance").** Salvedad: *rankea*
   toxicidad — es una capa de scoring/filtro, **no** un decisor. El AUC es
   la respuesta de ranking; la respuesta económica es el ítem 3 — el delta
   de P&L con su IC por bloques de mercado, **no** el AUC.
3. **Veredicto walk-forward (h=300s): el delta de P&L del filtro sobre
   sin-filtro es +$99.24 BRUTO** (nulo de filtro aleatorio rechazado al 5%).
   Salvedad: el IC bootstrap por bloques **[−$20,45, +$121,74] incluye 0**, el
   nivel absoluto @0.7 sigue negativo (−$7,51) y la simulación de cola + costos
   están pendientes — **el gate sigue cerrado.** Detalle completo abajo
   (tablas + JSONs).

## Veredicto de un vistazo

| Afirmación | Veredicto | Evidencia |
|---|---|---|
| El modelo rankea la toxicidad de fills OOS sin fugas (AUC 0.7276 fill×horizonte; 0.7356 por mercado) | ✅ **FUNCIONA** | [walkforward_toxicidad.json](results/walkforward_toxicidad.json) · [folds_mercado_toxicidad.json](results/folds_mercado_toxicidad.json) · fila *The model* |
| El delta de P&L del filtro sobre sin-filtro es positivo y rechaza el nulo de filtro aleatorio al 5% (+$99.24 BRUTO) | ✅ **FUNCIONA** — salvedad: el IC [−$20.45, +$121.74] incluye 0 | [nulo_delta_h300.json](results/nulo_delta_h300.json) · fila *Test nulo del delta* |
| El filtro gana dinero en términos absolutos / está listo para operar | ❌ **NO FUNCIONA** — nivel @0.7 negativo (−$7.51), WFE −0.8%, umbrales inestables | [walkforward_toxicidad_h300.json](results/walkforward_toxicidad_h300.json) · fila *P&L maker walk-forward* |
| Una sim de cola maker-realista (quotes persistentes) confirma que el delta persiste | ⚠️ **FUNCIONA (solo dirección)** — delta +$9.26, ambos folds; AUC ≈ 0.56 | [walkforward_toxicidad_h300_persist.json](results/walkforward_toxicidad_h300_persist.json) · fila *Sim de cola, modo persist* |
| Gate de trading en vivo (cola + costos + tamaño) | ⏳ **PENDIENTE** — el gate sigue cerrado | [PLAN_SIM_COLA.md](docs/PLAN_SIM_COLA.md) · filas *The model* / *Next steps* |

**Gráficos** — generados por [`charts/generar_graficos.py`](charts/generar_graficos.py)
(determinista, solo stdlib: dos corridas producen SVG byte-idénticos):

![Filtro de toxicidad: delta de P&L por fold temporal (umbral fijo 0.7)](charts/delta_por_fold.svg)
*Delta por fold del filtro de toxicidad (esquema fill x horizonte, umbral fijo 0.7), con el IC 95% por block-bootstrap del esquema h=300s como referencia.*

![Markout medio por horizonte y símbolo](charts/markout_por_horizonte.svg)
*Markout medio (bps del mid) por horizonte y símbolo — los scores del modelo por fill no están en los datos públicos, así que es el reemplazo honesto de un gráfico por deciles de score.*

## Servicio: auditoría de backtests

Auditamos backtests de trading cuantitativo buscando los modos de falla que
sobreviven a la revisión casual: look-ahead, fills gratis, tuning in-sample,
supervivencia, dependencia de escala. **Qué recibís:** veredicto escrito por
hallazgo con el mecanismo exacto, un JSON machine-readable por check, y un
harness reproducible solo-stdlib que podés correr vos mismo. El método de
auditoría se audita a sí mismo (ver *El auditor auditado* abajo: 8.000
estrategias nulas sin edge en 8 tipos, trampas de look-ahead plantadas,
limitaciones de
detectores documentadas con un estudio dedicado). Alcance y plazo: precio fijo
$250–500 por backtest, entrega en 5 días hábiles — ejemplo de una página del
método aplicado a este mismo pipeline: [CASE_STUDY.md](CASE_STUDY.md).
Contacto: abrir un issue en GitHub (dirección de contacto dedicada en preparación).

## Correr la demo (solo stdlib, sin dependencias)

`demo/` y los gráficos corren **standalone** (solo stdlib); `tests/` además necesita pytest — nada más.

```bash
python3 demo/audit_demo.py
```

## Lo más relevante
- **159M+ eventos de libro** (libro_eventos) + **958k+ fills** (trades_full)
  capturados 24/7 a hypertables de TimescaleDB — **15,161+ mercados trackeados,
  237 tokens activos en los últimos 7 días** (results/CAPTURE_STATUS.md,
  2026-10-24).
- **Cientos de miles de eventos de libro por hora, sostenido 24/7**, a una
  hypertable de TimescaleDB (compresión + retención).
- **Clasificador de toxicidad: LightGBM AUC 0.7276 OOS pooled (temporal, sin
  fugas, esquema fill×horizonte; por mercado 0.7356)**, rango por fold **0.689-0.786** — las filas son pares
  fill×horizonte (cada fill aparece una vez por horizonte), así que las tablas
  de P&L suman cada fill 4×. La brecha pooled/por-mercado es diferencia de
  tasas base, y se declara.
- Se detectaron y eliminaron **trampas de fuga de features** (tamaño del trade entrante, snapshots viejos, pooling de horizontes) que inflaban el AUC de 0.736 a 0.765.
- Fundamentado en la literatura: OFI (Cont 2014), micro-precio (Stoikov 2018),
  markout, Glosten-Milgrom / Kyle.
- **Demo de auditoría de backtests** — teardown ejecutable, sin dependencias
  ([`demo/audit_demo.py`](demo/audit_demo.py), resultados en
  [`results/AUDIT_DEMO.md`](results/AUDIT_DEMO.md)): un backtest sintético con
  look-ahead plantado, fills gratis y tuning in-sample reporta Sharpe 98.33;
  obligado a obedecer la realidad (información ejecutable, fills al libro,
  walk-forward) pierde exactamente el spread. Cinco checks de auditoría,
  auto-verificados.
- **Venue 2 activo: perpetuos USDT-M de Binance** (16-10-2026) — mismo
  pipeline, segunda clase de activo: aggTrades con lado agresor + libro top-5 +
  funding de BTC/ETH/SOL perp, captura systemd 24/7. Analítica de
  markout/selección adversa en
  [`analytics/adverse_selection.py`](analytics/adverse_selection.py).

### Estado de datos (declarado, al 2026-10-22)
- Captura continua **2026-10-14 → presente** (stream de trades: sin huecos).
- **Corte conocido:** stream de libro caído **2026-10-15 ~17:00–20:00 UTC**
  (~3h; los servicios systemd no estaban `enabled` y murieron con una sesión
  interactiva; arreglado — todos los capturadores hoy con
  `systemctl --user enable` + linger).
- Cualquier análisis que cruce esa ventana debe excluirla o marcarla. Lo decimos
  porque un dataset honesto declara sus huecos.
- **Estado en vivo:** [`results/CAPTURE_STATUS.md`](results/CAPTURE_STATUS.md)
  — generado a diario desde la base de producción, lo que leés siempre tiene
  <24h.

## Arquitectura
```
WebSocket (canal market, por venue)
   │  productor asyncio → cola → consumidor  (la red nunca bloquea el procesamiento)
   ▼
Libro en memoria por instrumento (snapshot + diffs incrementales)
   │  flush micro-batch: 500 filas / 2s
   ▼
Hypertables TimescaleDB (libro_eventos, trades_full, libro_top5, market_meta
                          + tablas binance_* del venue 2)
   │  feature engineering (OFI, microprice_bias, momentum, ...)
   ▼
Clasificador de toxicidad (LightGBM)  →  AUC 0.7276 OOS pooled, temporal (sin fugas; esquema fill×horizonte; por mercado 0.7356)
```

## Módulos
| Módulo | Propósito | Corre con |
| `microstructure/capture/http.py` | HTTP tolerante a fallas de ISP + fallback DNS por IP fijada (resiliencia a NXDOMAIN selectivo) | solo repo (librería) |
| `microstructure/data/tsdb.py` | Conexión TimescaleDB + helpers de flush micro-batch | TimescaleDB privada |
| `microstructure/model/features_ml.py` | Feature engineering: OFI, micro-precio, momentum multi-horizonte, etiquetas de markout | captura privada (entrada) |
| `microstructure/model/entrenar_toxicidad.py` | Entrenamiento: GroupKFold por mercado, z-score por mercado, features sin fuga | dataset privado (gitignored) |
| `microstructure/model/backtest_walkforward.py` | Backtest walk-forward del filtro de toxicidad: splits temporales, embargo 300s, P&L exacto de fill — publicado verbatim desde el bot privado | dataset privado (gitignored) |
| `microstructure/model/backtest_desglose.py` | Desglose OOS del veredicto walk-forward: por horizonte, lado, agresividad del fill, spread + ajuste maker-realista | dataset privado (gitignored) |
| `microstructure/sim/sim_cola.py` + `microstructure/sim/sim_maker_dataset.py` | Simulación de fills con cola sobre la captura REAL con tres modos de re-quote comparables (**persist** = honesto, sin auto-cancel / **cancel** = status quo / **contra** = conservador) + su constructor de dataset — publicado verbatim desde el bot privado | captura privada (CSV de fills incluido) |
| `microstructure/model/backtest_walkforward_persist.py` | El mismo walk-forward (solo adaptaciones documentadas) sobre los fills simulados en modo persist — `results/maker_fills_sim_persist.csv` (1.1 MB, incluido) → `results/walkforward_toxicidad_h300_persist.json`, totalmente reproducible | **standalone** — viene con el repo (copiar el CSV incluido a `microstructure/model/dataset/`) |
| `microstructure/model/backtest_nulo_delta.py` | Test nulo del delta del filtro: nulo de filtro aleatorio (misma retención por fold) + bootstrap por bloques de mercado + control con labels permutadas — publicado verbatim desde el bot privado | dataset privado (gitignored) |
| [VALIDATION_WINDOW.md](docs/VALIDATION_WINDOW.md) | Ventana de validación reservada pre-registrada: congelamiento 2026-10-25, evaluación única 2026-11-08, regla de decisión fijada de antemano (misma regla que el nulo del 23-10) | **standalone** — documentación, sin datos |

## El modelo (números honestos)
- **Target:** markout de cada trade real a +horizonte (mid del libro vs precio
  de fill). Markout < 0 = selección adversa.
- **Etiquetado multi-horizonte:** cada fill etiquetado a 1s/10s/60s/300s. El
  dataset walk-forward con tope de frescura es **382,582 filas fill×horizonte**
  (268,446 fuera de muestra, en **224 fills distintos**; cada fill aporta una
  fila por horizonte — son pares fill×horizonte).
- **Validación:** GroupKFold por mercado (generalización a mercados no vistos)
  + normalización z-score por mercado.
- **Fuga eliminada:** `pos_en_spread` y `fill_price` reconstruyen el target por
  construcción — se quitaron al detectarla, el AUC bajó de 0.702 a 0.6425
  (baseline MLP).
- **Upgrade a LightGBM:** AUC pooled **0.7356** OOS en mercados no vistos
  (re-corrída sin fugas, GroupKFold por mercado: snapshots con tope de
  frescura, estandarización causal) — `results/folds_mercado_toxicidad.json`.
  Mismo pipeline sin fugas que el titular temporal fill×horizonte (AUC
  0.7276): esquema de validación distinto, no una contradicción — ver
  docs/RESULTS.md, "AUC provenance".
- **Limitaciones honestas:** rankea toxicidad (AUC) pero el accuracy binario
  está bajo el baseline mayoritario → es una capa de scoring/filtro, no un
  decisor autónomo. Con markout exacto + walk-forward de horizonte único
  (h=300s, 22-10) el delta es **positivo y rechaza el nulo de filtro aleatorio al 5%**
  (test nulo 23-10 — fila abajo; el IC bootstrap por bloques es amplio e
  incluye 0, se reporta ambos) — un
  delta de P&L BRUTO de **+$99.24** sobre sin-filtro bajo supuestos de cinta —
  pero el nivel absoluto @0.7 es **negativo** (−$7.51 sobre 80,379 ops; sin
  filtro −$106.75), la eficiencia walk-forward por operación es **−0.8%**
  (por debajo de la línea de aviso de sobreajuste del 30%), los umbrales por
  fold son inestables, y una primera simulación con cola (T2/T3, 18-10) fue
  **inconclusa**: el AUC del modelo sobre fills simulados es ~0.50-0.56
  (features aproximadas, fuera de distribución), así que los números de cinta
  asumen fills ideales ([PLAN_SIM_COLA.md](docs/PLAN_SIM_COLA.md)). **Re-corrida de
  la sim con la suposición 3 corregida** (quotes persistentes — el viejo
  auto-cancel optimista eliminaba justo los fills adversos que un maker real
  sufre; motor de sim publicado verbatim en `microstructure/sim/`): el delta
  del filtro sobre sin-filtro es **+$9.26** BRUTO (sin filtro −$15.57 →
  filtrado @0.7 −$6.31; 2 de 3 folds temporales (fold 2 descartado por el embargo), 1,580 fills, 1 fill = 1
  muestra, h=300s), **positivo en ambos folds** — el delta negativo viejo de
  T2/T3 (−$682) era un artefacto de esa suposición de auto-cancel. El AUC
  sobre la sim sigue cercano al azar (pooled 0.556, folds 0.533–0.599;
  features aproximadas por diseño) y la muestra es ~80× más chica que el
  dataset real, así que esto se lee como DIRECCIÓN, no magnitud — el gate
  sigue cerrado. El nivel
  anterior +$489 @0.7 venía del esquema de 382,582 filas que sumaba los cuatro
  horizontes de cada fill (pares fill×horizonte) — una estimación anterior
  contada 4×; el esquema actual cuenta cada fill una sola vez. El gate sigue
  cerrado.

| Métrica | Valor |
|---|---|
| AUC-ROC pooled OOS (LightGBM, sin fugas, temporal, esquema fill×horizonte) | **0.7276** |
| PR-AUC OOS / prevalencia | **0.4963 / 0.304** |
| AUC por fold temporal (rango, 5 folds, h=300s) | **0.653 / 0.787** |
| Filas walk-forward (horizonte único h=300s, refresh 22-10-2026) | 129,175 totales; 80,379 ops OOS al umbral fijo 0.7 |
| **P&L maker walk-forward OOS (horizonte único h=300s, refresh 22-10-2026, 129,175 filas — BRUTO, supuestos de cinta; cola+costos pendientes)** | **DELTA-POSITIVO, NIVEL-NEGATIVO, AÚN FRÁGIL** — markout exacto, splits temporales, embargo 300s, 1 fill = 1 muestra: sin filtro **−106.75 USD** · filtro @0.7 **−7.51** (80,379 ops) · **delta sobre sin-filtro +99.24** · umbrales óptimos por fold 0.6/0.5/0.7/0.6/0.55 (**no estables**) → filtro fijo 0.7. WFE **−0.8% por operación** (P&L OOS vs in-muestra por op; por debajo de la línea de aviso de sobreajuste del 30%). Estimación anterior contada 4× (esquema de 382,582 filas, los cuatro horizontes por fill): +488.98 @0.7. Gate cerrado hasta simular cola + costos; números reportados igual. |
| **Walk-forward por horizonte (1 fill = 1 muestra por horizonte, fijo 0.7, BRUTO supuestos de cinta, re-corridas 23-10-2026)** | el delta es **positivo en los cuatro** horizontes y crece con el horizonte — h=1s: **+18.49** (AUC folds 0.637–0.740) · h=10s: **+56.60** (0.682–0.756) · h=60s: **+89.27** (0.666–0.774) · h=300s: **+99.24** (0.653–0.787). El NIVEL sin filtro cambia de signo con el horizonte (+77.84 / +101.32 / +40.62 / −106.75) — la selección adversa crece con el horizonte; el delta es la cantidad estable. El **AUC 0.7276** pooled de arriba viene del esquema fill×horizonte (382,582 filas); los AUC por horizonte son los rangos de esta fila (deltas con test nulo — filas abajo). **Los cuatro horizontes salen de los mismos fills — los deltas están correlacionados, no son cuatro confirmaciones independientes.** El h=300s del titular se fijó **a priori** por la regla de decisión del 23-10-2026 (fila de abajo); los demás horizontes son secundarios. JSONs: `results/walkforward_toxicidad_h{1,10,60,300}.json`. |
| **Test nulo del delta (regla de decisión fijada a priori el 23-10-2026, antes de correr: horizonte único h=300s, umbral 0.7, estadístico = delta; rechazo al 5% si delta pooled ≥ p95 de un nulo de filtro aleatorio Y delta > 0 en ≥4/5 folds)** | **RECHAZADO al 5%.** Delta pooled observado **+98.23** (re-corrida GPU; referencia CPU +99.24) vs nulo de filtro aleatorio (misma retención por fold, 1.000 draws seed fija) p95 **+45.27** — el observado queda sobre el percentil 100 del nulo (p99 +52.01, max +61.02), y es > 0 en **4/5 folds**. Control: 50 reentrenos con labels permutadas dentro de cada mercado → delta **−1.10 ± 10.21** (min −22.23, max +19.09): el pipeline no fabrica señal. Secundario (umbral re-elegido por draw): el delta observado queda en p98.9 de ese nulo. **Caveat honesto:** el IC bootstrap por bloques de mercado del delta, **[−20.45, +121.74], incluye 0** (volatilidad de muestreo alta; 2 de 5 folds casi nulos) — se reporta ambos resultados. El gate sigue cerrado (nivel negativo, WFE negativa, cola+costos pendientes). Código y JSON: `microstructure/model/backtest_nulo_delta.py`, `results/nulo_delta_h300.json`. |
| **El auditor auditado: Monte Carlo de los 5 detectores de fugas (27-10-2026, alpha=0.01, 999 permutaciones/test, ICs de Wilson 95%)** | Sobre **8.000 estrategias nulas sin edge** (8 tipos × 1.000 seeds, mercados de 90 días, spread 2×40 bps): **tasa de falsos positivos por detector — capturabilidad 0.00%, fills al book 0.00%, tuning in-sample 0.06% (5/8.000) [IC 0.03, 0.15], kfold-cv 0.86% (69/8.000) [IC 0.68, 1.09], supervivencia 1.50% (120/8.000) [IC 1.26, 1.79]; unión 2.39% (191/8.000) [IC 2.08, 2.75]** — todo detector ≤ 10%. **Potencia: 100% (unión) en las tres trampas de look-ahead** (completo: atrapado por tuning/kfold/supervivencia — supervivencia sola 83/100; de 1 barra: por los cinco; 50% parcial: capturabilidad 86%). **Nulos agregados 25-10-2026:** stop-loss mecánico (declarado → el check de supervivencia lo SKIPEA por diseño) FP unión 0.90% [IC 0.47, 1.70]; pagos asimétricos con cola derecha gorda, NO declarado (no se filtra nada) FP unión 5.50% [IC 4.25, 7.09] — casi todo del detector de supervivencia (4.90% [IC 3.73, 6.42]): FP honesto de payoff mecánico, debajo del presupuesto del 10%; alpha rápido real chico (edge = 10% del half-spread): 0.00% [IC 0.00, 0.38] — genera ~1 trade por corrida, consistente con la trampa de edge chico. **Nulo agregado 27-10-2026 (control de especificidad):** estrategia honesta con edge chico positivo *neto de costos declarados* (media +4,5 bps por trade, half-spread 4,5 bps, fills declarados netos) y **mínimo de 200 trades exigido (observado: mín 2.808, mediana 3.645)**: acusada **7,2% (72/1.000) [IC 5,8, 9,0]** — debajo del presupuesto de FP del 10% pero arriba del 5% nominal, casi todo del check de CV temporal reaccionando a un edge real (chico); documentado como limitación conocida, no como falla de plumbing. En síntesis: un edge chico de ~1 trade se acusa 0% de las veces, mientras que un edge real chico neto-positivo con 200+ trades se acusa 7,2% — los detectores cazan plumbing, y separar un edge real chico del ruido lleva más historia de la que dan 5 bloques de CV. Limitación conocida: capturabilidad no ve un look-ahead de hold completo (1 barra de retraso quita solo 1/30 del peek); los otros cuatro detectores lo cubren. ~105 min de pared en 6 núcleos de CPU. JSON: `results/monte_carlo_fp_power.json`. |
| **El auditor se audita a sí mismo: cinco controles sintéticos × dos escalas de precio (23-10-2026, 3 honestas + 2 trampas de look-ahead, p0=0.5 vs p0=100; las cinco son sintéticas — escritas por nosotros para esta auditoría, sin código de terceros)** | **Invariancia de escala restaurada y verificada**: el harness ahora escala el costo de round-trip con la escala del mercado (bps constantes en 80: `half_spread = HALF_SPREAD * p0 / P0`), y el rng de auditoría se siembra idéntico por estrategia entre escalas — 5/5 estrategias devuelven **veredictos idénticos en ambas escalas**, ratio de pnl exactamente 200.0×. Veredictos: las 2 trampas de look-ahead FALLAN/WARNEAN en ambas escalas (tuning in-sample, kfold, fills al book disparan); las 3 honestas PASAN — incluida `spread_sniper` desde la **auditoría forense del 24-10-2026**: la vieja marca de supervivencia (7/24 honestas = 29.2% en ambas escalas) **NO era un artefacto del costo fijo** — pnl_net = pnl_gross − 2·half_spread es una traslación pura y el test de colas rank-based es invariante por traslación ((nl, nr, p) idénticos net vs gross en 48/48 corridas; FP idéntico 7/24 con costo relativo 200× distinto). La causa raíz real era **independencia muestral rota**: los clientes selectivos de nivel entran en ráfaga — una excursión de precio genera una corrida de entradas cuyas ventanas de hold comparten el mismo camino realizado — así que el binomial trataba 9.7k–54k filas reportadas como dibujos independientes (peor seed: un solo clúster con el 61% de los trades). El check 5 ahora corre el test de colas sobre **trades NO solapados** (mismas unidades que el Sharpe honesto; <200 unidades → SKIP): **0/24 marcadas en ambas escalas, pnl absoluto y relativo**; el FP de supervivencia del MC bajó 5.0% → 1.19% (supervivencia [0.96%, 1.47%] Wilson) con potencia en trampas preservada (83–100/100). El diagnóstico resuelto queda en el JSON (`limitaciones_conocidas`); script y JSON del estudio: `demo/estudio_limitacion_supervivencia.py`, `results/estudio_limitacion_supervivencia.json`. El edge falso que antes reportaba `momentum_chaser` a p0=100 (media +0.086/trade) venía de un spread aditivo sobre precios lognormales — el fix de escalar el costo lo elimina. |
| **Sim de cola maker-realista, modo persist (walk-forward sobre fills simulados de quotes persistentes, 23-10-2026)** | La prueba más cercana a un maker real: fills por posición en cola con **quotes persistentes (sin auto-cancel)** — la suposición cuya remoción causó el viejo artefacto de −$682; motor de sim publicado verbatim en `microstructure/sim/`. Sobre 1,580 fills simulados (1 fill = 1 muestra, h=300s, 2 folds temporales): sin filtro **−$15.57** → filtro @0.7 **−$6.31**, **delta +$9.26, positivo en ambos folds**; AUC OOS pooled **0.5563** (folds 0.533–0.599 — features aproximadas a propósito, fuera de distribución); WFE por operación **−33%** (totales −19.6%). La muestra es ~80× más chica que el dataset real → se lee como **DIRECCIÓN, no magnitud**; el gate sigue cerrado. JSON: `results/walkforward_toxicidad_h300_persist.json`. |

## Venue 2 — perpetuos USDT-M de Binance (activo desde el 16-10-2026)
Segundo venue alimentando el mismo stack de TimescaleDB, mismo protocolo de
sincronización de libro (snapshot + diffs con verificación de cadena de
update-ids, detección de gaps, resync — el algoritmo oficial de Binance):
aggTrades con lado agresor, libro top-5 cada 2s, y funding/mark-price cada
5 minutos para BTCUSDT/ETHUSDT/SOLUSDT. Capturado 24/7 por un servicio systemd
(streams públicos, sin cuenta involucrada). Latencia evento→ingesta ~0.3s.
Analítica de calidad de ejecución (funciones puras, testeada):
[`analytics/adverse_selection.py`](analytics/adverse_selection.py) — markout a
múltiples horizontes, sin look-ahead (el mid usado es el último conocido en o
antes de t+h; los trades cuyo horizonte pasa la última cita se excluyen, nunca
se extrapolan).

Venue 1 (CLOB de Polymarket, arriba) = mercados de predicción de eventos
discretos; este venue 2 = libros continuos de alta frecuencia — mismo schema,
misma analítica.

### Markout en bps del mid (2026-10-16 22:54 → 2026-10-19 00:05 UTC, 3,145,683 trades agresivos)

| Símbolo | media bps @5s | mediana bps @5s | fracción tóxica @5s | media bps @300s |
|---|---|---|---|---|
| **BTCUSDT** | +0.37 | +0.09 | 46% | +0.38 |
| **ETHUSDT** | +0.57 | +0.27 | 43% | +0.43 |
| **SOLUSDT** | **−0.28** | **−0.48** | **68%** | −0.29 |

**Cómo leer esto (y qué NO afirmar):** el markout del mid mide adverse
selection del flujo *agresivo* — un maker que toma el otro lado gana el
negativo de esto. BTC/ETH muestran markout positivo chico (sus agresores
aciertan a 5s); SOL es claramente tóxico (mediana −0.48 bps, 66–68% de los
trades taker pierden). **Sobre el término de entrada:** la entrada de BTC
−0.44 bps (~$3.6 sobre ~$81k) está muy por encima del spread cotizado de un
tick, y una estratificación por staleness del mid de entrada
([`results/markout_entry_staleness_probe.json`](results/markout_entry_staleness_probe.json),
sonda de solo lectura sobre la tabla completa de captura, ~11.5 días)
muestra que **no** se encoge con mids más frescos: la entrada se mantiene
≈−0.5 bps para mids de entrada de menos de 2s (~90% de los trades:
0–0.5s −0.52, 1–2s −0.53) y solo degrada con mids de ≥2s (2–5s: −0.87).
**Origen del piso de entrada: un desfase fijo de timestamps entre trades y
snapshots; ningún piso negativo permanente sobrevive más allá de ~1 s.** Una
sonda de desfase de timestamps
([`results/markout_entry_lag_probe.json`](results/markout_entry_lag_probe.json),
script
[`microstructure/scripts/markout_entry_lag_probe.py`](microstructure/scripts/markout_entry_lag_probe.py),
solo lectura, ventana 2026-10-21→10-28, δ ∈ {−0.5 … +2 s}) recalcula la
entrada con el mid corrido δ: entry(δ) **no** es plano — la mediana colapsa
monotónicamente hacia su medio-spread estático en δ≈+0.5…+1 s (BTC −0.280 →
−0.006 a +0.5s → −0.006 a +1s; ETH −0.368 → −0.062 → −0.021; SOL plano en
−0.491 ≡ su medio-spread 0.490 en todos los δ) y se vuelve **positiva** en
δ=+2 s (BTC mediana +0.080 / media +0.228; ETH +0.153 / +0.294) cuando el mid
deriva con la tendencia. La media decae más lento que la mediana (BTC −0.591 →
−0.244 a +0.5s → −0.090 a +1s) por una cola pesada; cruza cero recién entre
+1 s y +2 s, y la media por sí sola no identifica nada — con δ positivos la
búsqueda del mid corrido mezcla la corrección del desfase con deriva real a
favor del agresor (la señal de toxicidad misma), así que solo la meseta de la
mediana localiza el desfase, y queda una media residual de ≈−0.09 bps a +1 s
sin explicar por el desfase solo. El mínimo queda encerrado, y la cola de la
media a +0.5 s es movimiento transitorio del mid, no un piso permanente. Consistente con un desfase fijo de
ingesta ≲0.5 s entre el stamp de aggTrades y el del libro (las dos sondas
constriñen relojes distintos: staleness = frescura dentro de la propia línea
de tiempo del libro; desfase = alineación entre streams). **Consecuencia para
la lectura de toxicidad: la descomposición drift/entry hereda el mismo mid(t)
— el término de deriva también se mide contra el mid (posiblemente desfasado)
— así que ninguno de los dos términos debe citarse como medida de impacto
independiente, y la tabla de arriba debería recalcularse sobre la medida
limpia raw (fill a fill: precio de salida − precio del fill, sin mid) antes
de usarse como número absoluto de costo/impacto.** **No leer esto como
"el taker paga el spread dos veces", ni como una medida de calidad de
ejecución, ni como un impacto real establecido** — los números de horizonte
corto son ruido de piso, no un modelo de costos.

**Robustez — markout crudo trade-to-trade (sin mid, sin libro):**
[`results/markout_trade_to_trade.json`](results/markout_trade_to_trade.json)
([`microstructure/scripts/markout_trade_to_trade_probe.py`](microstructure/scripts/markout_trade_to_trade_probe.py),
solo lectura, misma ventana 2026-10-21→10-28) recalcula el markout de salida
enteramente dentro del stream de aggTrades: raw(h) = 1e4·signo·(p_trade(t+h)
− p_trade(t)) / p_trade(t), con p_trade(t+h) el primer trade en o después de
t+h — sin lookup de mid, sin snapshot de libro, y por lo tanto sin reloj de
cruce entre streams que desalinear. **No** reproduce el markout negativo de
horizonte corto de las sondas basadas en snapshots: las medianas son
positivas y crecen con h (BTC +0.17 @0.5s → +0.24 @2s → +0.39 @300s; ETH
+0.26 → +0.34 → +0.38), casi idénticas para ambas manos agresoras — los
trades agresivos van seguidos de continuación en su dirección, no de
reversión. La mediana de SOL es ≈0 en todos los horizontes: con medio-spread
≈0.49 bps y grilla de precio de 1 tick la medida queda dominada por el
rebote/tick y hay que leerla con cuidado. Esta medida tiene su propia
contaminación de convención (las salidas son prints taker futuros, las
entradas pagan el spread), así que es un cross-check, no un modelo de
costos; su rol es acotar el artefacto de desfase: la medida sin desfase no
muestra piso negativo de horizonte corto para BTC/ETH, lo que acota cuánto
de la negatividad basada en snapshots puede ser costo real de selección
adversa y cuánto artefacto de alineación/convención. Todas las cifras solo
descriptivas.

**Robustez — tope estricto de frescura (0.5s):** re-corrida con
`--max-mid-age 0.5` (descarta ~78% de los fills, frescura mediana del
mid de salida 0.25s), solo mids de salida frescos;
[`results/markout_strict_freshness.json`](results/markout_strict_freshness.json)
conserva la misma forma: BTC +0.38, ETH +0.50, SOL −0.26 bps medios @5s
(deriva ~+0.81/+1.08/+0.31, entrada ~−0.43/−0.58/−0.58). Un tope adicional
de 1s ([`results/markout_max_mid_age_1s.json`](results/markout_max_mid_age_1s.json))
mueve la deriva de BTC a 0.77 @5s / 0.84 @300s y la entrada a −0.40 @5s —
la descomposición se mueve solo marginalmente con el tope de frescura.
El orden y los
signos por mercado son estables; la descomposición entrada/deriva **no** se
cita como modelo de costos — el término de deriva casi no escala con el
horizonte (0.81 @5s → 0.68 @300s), así que "deriva del mercado" vs
"desfase del mid" no se pueden separar con estos datos y la descomposición
se quitó de la lectura principal.

Regenerar con `microstructure/scripts/markout_binance.py` (solo lectura, DB
en vivo; último reporte machine-readable en
[`results/markout_report.json`](results/markout_report.json)).

## Casos de estudio (auditorías públicas de backtests)

Auditorías de backtests públicos — mismo archivo
de datos (sha256 en cada página), mismo método, scripts propios. Recorridos
completos en [`docs/cases/`](docs/cases/):

- **[Caso 01 — MachineLearningStocks](docs/cases/CASE_01_MachineLearningStocks.md)** —
  el split aleatorio reproduce aproximadamente el +28.6 pts del autor (nuestro:
  **+30.1**); el walk-forward
  temporal con embargo de 365 días cae a **+0.2 pts (IC [−4.6, +6.1])**.

## Próximos pasos (acotados)
- ~~MLP → LightGBM~~ **hecho** — superado por la re-corrída sin fugas (AUC 0.7356 por mercado / 0.7276 fill×horizonte, ambos sin fugas).
- ~~Backtestear el filtro contra P&L bajo supuestos de cinta~~ **hecho sobre features sin fugas
  (snapshots con tope de frescura, estandarización causal, folds temporales) —
  en walk-forward temporal estricto de horizonte único (h=300s, 1 fill = 1
  muestra, embargo 300s) el delta es positivo (nulo de filtro aleatorio
  rechazado al 5%; IC bootstrap por bloques amplio, incluye 0 — se reporta
  ambos):
  **+$99.24 de delta de P&L BRUTO sobre sin-filtro** (nivel @0.7: −$7.51;
  bruto, supuestos de cinta; refresh 22-10, 129,175 filas). El esquema
  anterior que contaba 4× (pares fill×horizonte, 382,582 filas) mostraba
  delta +$376 / nivel +$489 — misma historia cualitativa, nivel inflado. La
  eficiencia walk-forward por op es **−0.8%** y los umbrales óptimos por fold
  siguen **inestables** (0.6/0.5/0.7/0.6/0.55); falta simular la cola
  (docs/PLAN_SIM_COLA.md) — el gate sigue cerrado.** El AUC rankea el flujo;
  convertirlo a P&L maker rentable (cola, rewards, fees) sigue abierto.
- Investigar por qué el ranking no traduce: P&L por horizonte/régimen,
  simulación con posición en cola, calibración del modelo de costos.
- **Ventana de validación reservada (pre-registrada el 2026-10-25):**
  evaluación única sobre datos frescos el **2026-11-08**, con la regla de
  decisión exacta pre-registrada el 23-10 (h=300s, umbral fijo 0.7, delta
  pooled vs p95 del nulo de filtro aleatorio, delta > 0 en ≥4/5 folds); el
  resultado se publica incondicionalmente — ver
  [VALIDATION_WINDOW.md](docs/VALIDATION_WINDOW.md).

## Licencia
MIT — ver [LICENSE](LICENSE). Uso libre, atribución apreciada.

## Autor
Gustavo Ruarte — market microstructure & ML engineering.
Contacto: abrir un issue en GitHub (dirección de contacto dedicada en preparación).
