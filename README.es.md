# Microestructura de mercado: captura 24/7 + modelo de toxicidad de fills (honesto)

> [EN](README.md) | **ES**

Pipeline de microestructura de mercado end-to-end para CLOBs de mercados de
predicción (libros binarios tipo Polymarket) y perpetuos de cripto: captura
24/7 multi-venue del libro por WebSocket a TimescaleDB, más un modelo de ML que
predice la **toxicidad de fills (selección adversa)** desde features del libro
— y una demo ejecutable de la auditoría de backtests que mantiene los números
honestos. Todos los números son reales y out-of-sample — **incluidas las
partes que no funcionaron**.

**Resultados titular**

1. **119M+ eventos de libro y 645k+ fills reales** capturados 24/7 a hypertables
   de TimescaleDB (12,900+ mercados trackeados, 9,900+ activos en los
   últimos 7 días).
2. **Clasificador de toxicidad: AUC 0.7384 OOS en mercados no vistos (folds 0.689-0.786, sin fugas); PR-AUC 0.506 vs prevalencia 0.304** — una capa de scoring/filtro, **no** un decisor.
3. **Veredicto walk-forward (horizonte único h=300s): delta positivo, nulo de filtro
   aleatorio rechazado al 5% (IC bootstrap por bloques amplio, incluye 0 — se reporta ambos) — el delta de P&L del filtro sobre sin-filtro es
   **+$99.24** en el refresh del 22-10 de horizonte único (129,175 filas, 1
   fill = 1 muestra a h=300s), mientras el nivel sin filtro es **−$106.75** y
   el nivel filtrado @0.7 es **−$7.51** (80,379 ops) — el nivel @0.7 sigue
   siendo negativo en horizonte único (todas las cifras son BRUTAS bajo
   supuestos de cinta: se cuenta cada fill del tape, sin posición en cola, sin
   fees, sin costos). El nivel
   anterior **+$489 @0.7** venía del esquema de 382,582 filas que sumaba los
   cuatro horizontes de cada fill (pares fill×horizonte), inflando el nivel
   ~4×; el esquema actual cuenta cada fill una sola vez. La simulación de cola
   ([PLAN_SIM_COLA.md](PLAN_SIM_COLA.md)) sigue pendiente para cerrar el
   gate; una primera corrida con cola (T2/T3) fue **inconclusa** (AUC
   ~0.50-0.56 sobre features aproximadas, fuera de distribución); la
   re-corrida corregida (suposición 3 arreglada: quotes persistentes, sin
   auto-cancel optimista) da un **delta positivo** (+$9.26 BRUTO, en ambos
   folds) con AUC aún cercano al azar (~0.556) sobre una muestra chica
   (1,580 fills) — dirección consistente con los datos reales, magnitud no
   informativa (detalles abajo). La
   eficiencia walk-forward es **−0.8% por operación** (P&L OOS vs in-muestra
   por op al umbral fijo — por debajo de la línea de aviso de sobreajuste del
   30%) y los umbrales óptimos por fold son **inestables** (0.6/0.5/0.7/0.6/
   0.55), por eso corre el filtro fijo 0.7. El gate sigue cerrado.

Estado de captura en vivo (generado a diario desde la base de producción):
[`results/CAPTURE_STATUS.md`](results/CAPTURE_STATUS.md).

Contacto: abrir un issue en GitHub o escribir a gustavoaruarte@gmail.com.

## Lo más relevante
- **85M+ eventos de libro** (libro_eventos) + **382k+ fills** (trades_full)
  capturados 24/7 a hypertables de TimescaleDB — **9,900+ mercados trackeados,
  362 tokens activos en los últimos 7 días** (verificado 2026-10-18).
- **Cientos de miles de eventos de libro por hora, sostenido 24/7**, a una
  hypertable de TimescaleDB (compresión + retención).
- **Clasificador de toxicidad: LightGBM AUC 0.736 OOS pooled en mercados no
  vistos (sin fugas)**, rango por fold **0.67-0.79** — las filas son pares
  fill×horizonte (cada fill aparece una vez por horizonte), así que las tablas
  de P&L suman cada fill 4×. La brecha pooled/por-mercado es diferencia de
  tasas base, y se declara.
- Se detectaron y eliminaron **trampas de fuga de features** (tamaño del trade entrante, snapshots viejos, pooling de horizontes) que inflaban el AUC de 0.736 a 0.765.
- Fundamentado en la literatura: OFI (Cont 2014), micro-precio (Stoikov 2018),
  markout, Glosten-Milgrom / Kyle.
- **Demo de auditoría de backtests** — teardown ejecutable, sin dependencias
  ([`demo/audit_demo.py`](demo/audit_demo.py), resultados en
  [`results/AUDIT_DEMO.md`](results/AUDIT_DEMO.md)): un backtest sintético con
  look-ahead plantado, fills gratis y tuning in-sample reporta Sharpe 34;
  obligado a obedecer la realidad (información ejecutable, fills al libro,
  walk-forward) pierde exactamente el spread. Cinco checks de auditoría,
  auto-verificados.
- **Venue 2 activo: perpetuos USDT-M de Binance** (16-10-2026) — mismo
  pipeline, segunda clase de activo: aggTrades con lado agresor + libro top-5 +
  funding de BTC/ETH/SOL perp, captura systemd 24/7. Analítica de
  markout/selección adversa en
  [`analytics/adverse_selection.py`](analytics/adverse_selection.py).

### Estado de datos (declarado, al 2026-10-18)
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
Clasificador de toxicidad (LightGBM)  →  AUC 0.736 OOS pooled, mercados no vistos (sin fugas)
```

## Módulos
| Módulo | Propósito |
|---|---|
| `microstructure/capture/http.py` | HTTP tolerante a fallas de ISP + fallback DNS por IP fijada (resiliencia a NXDOMAIN selectivo) |
| `microstructure/data/tsdb.py` | Conexión TimescaleDB + helpers de flush micro-batch |
| `microstructure/model/features_ml.py` | Feature engineering: OFI, micro-precio, momentum multi-horizonte, etiquetas de markout |
| `microstructure/model/entrenar_toxicidad.py` | Entrenamiento: GroupKFold por mercado, z-score por mercado, features sin fuga |
| `microstructure/model/backtest_walkforward.py` | Backtest walk-forward del filtro de toxicidad: splits temporales, embargo 300s, P&L exacto de fill — publicado verbatim desde el bot privado |
| `microstructure/model/backtest_desglose.py` | Desglose OOS del veredicto walk-forward: por horizonte, lado, agresividad del fill, spread + ajuste maker-realista |
| `microstructure/sim/sim_cola.py` + `microstructure/sim/sim_maker_dataset.py` | Simulación de fills con cola sobre la captura REAL con tres modos de re-quote comparables (**persist** = honesto, sin auto-cancel / **cancel** = status quo / **contra** = conservador) + su constructor de dataset — publicado verbatim desde el bot privado |
| `microstructure/model/backtest_walkforward_persist.py` | El mismo walk-forward (solo adaptaciones documentadas) sobre los fills simulados en modo persist — `results/maker_fills_sim_persist.csv` (1.1 MB, incluido) → `results/walkforward_toxicidad_h300_persist.json`, totalmente reproducible |
| `microstructure/model/backtest_nulo_delta.py` | Test nulo del delta del filtro: nulo de filtro aleatorio (misma retención por fold) + bootstrap por bloques de mercado + control con labels permutadas — publicado verbatim desde el bot privado |

## El modelo (números honestos)
- **Target:** markout de cada trade real a +horizonte (mid del libro vs precio
  de fill). Markout < 0 = selección adversa.
- **Etiquetado multi-horizonte:** cada fill etiquetado a 1s/10s/60s/300s. El
  dataset walk-forward con tope de frescura es **382,582 filas fill×horizonte**
  (268,446 fuera de muestra; cada fill aporta una fila por horizonte — son
  pares fill×horizonte, *no* fills distintos).
- **Validación:** GroupKFold por mercado (generalización a mercados no vistos)
  + normalización z-score por mercado.
- **Fuga eliminada:** `pos_en_spread` y `fill_price` reconstruyen el target por
  construcción — se quitaron al detectarla, el AUC bajó de 0.702 a 0.6425
  (baseline MLP).
- **Upgrade a LightGBM:** AUC pooled **0.7384** OOS en mercados no vistos
  (re-corrída sin fugas: snapshots con tope de frescura, estandarización
  causal, folds por mercado).
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
  asumen fills ideales ([PLAN_SIM_COLA.md](PLAN_SIM_COLA.md)). **Re-corrida de
  la sim con la suposición 3 corregida** (quotes persistentes — el viejo
  auto-cancel optimista eliminaba justo los fills adversos que un maker real
  sufre; motor de sim publicado verbatim en `microstructure/sim/`): el delta
  del filtro sobre sin-filtro es **+$9.26** BRUTO (sin filtro −$15.57 →
  filtrado @0.7 −$6.31; 2 de 3 folds temporales, 1,580 fills, 1 fill = 1
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
| AUC-ROC pooled OOS (LightGBM, leak-free, mercados no vistos) | **0.7384** |
| PR-AUC OOS / prevalencia | **0.5057 / 0.304** |
| AUC por fold temporal (rango, 5 folds, h=300s) | **0.653 / 0.787** |
| Filas walk-forward (horizonte único h=300s, refresh 22-10-2026) | 129,175 totales; 80,379 ops OOS al umbral fijo 0.7 |
| **P&L maker walk-forward OOS (horizonte único h=300s, refresh 22-10-2026, 129,175 filas — BRUTO, supuestos de cinta; cola+costos pendientes)** | **DELTA-POSITIVO, NIVEL-NEGATIVO, AÚN FRÁGIL** — markout exacto, splits temporales, embargo 300s, 1 fill = 1 muestra: sin filtro **−106.75 USD** · filtro @0.7 **−7.51** (80,379 ops) · **delta sobre sin-filtro +99.24** · umbrales óptimos por fold 0.6/0.5/0.7/0.6/0.55 (**no estables**) → filtro fijo 0.7. WFE **−0.8% por operación** (P&L OOS vs in-muestra por op; por debajo de la línea de aviso de sobreajuste del 30%). Estimación anterior contada 4× (esquema de 382,582 filas, los cuatro horizontes por fill): +488.98 @0.7. Gate cerrado hasta simular cola + costos; números reportados igual. |
| **Walk-forward por horizonte (1 fill = 1 muestra por horizonte, fijo 0.7, BRUTO supuestos de cinta, re-corridas 23-10-2026)** | el delta es **positivo en los cuatro** horizontes y crece con el horizonte — h=1s: **+18.49** (AUC folds 0.637–0.740) · h=10s: **+56.60** (0.682–0.756) · h=60s: **+89.27** (0.666–0.774) · h=300s: **+99.24** (0.653–0.787). El NIVEL sin filtro cambia de signo con el horizonte (+77.84 / +101.32 / +40.62 / −106.75) — la selección adversa crece con el horizonte; el delta es la cantidad estable. El **AUC 0.7384** pooled de arriba viene del esquema fill×horizonte (382,582 filas); los AUC por horizonte son los rangos de esta fila (deltas con test nulo — filas abajo). JSONs: `results/walkforward_toxicidad_h{1,10,60,300}.json`. |
| **Test nulo del delta (regla de decisión fijada a priori el 23-10-2026, antes de correr: horizonte único h=300s, umbral 0.7, estadístico = delta; rechazo al 5% si delta pooled ≥ p95 de un nulo de filtro aleatorio Y delta > 0 en ≥4/5 folds)** | **RECHAZADO al 5%.** Delta pooled observado **+98.23** (re-corrida GPU; referencia CPU +99.24) vs nulo de filtro aleatorio (misma retención por fold, 1.000 draws seed fija) p95 **+45.27** — el observado queda sobre el percentil 100 del nulo (p99 +52.01, max +61.02), y es > 0 en **4/5 folds**. Control: 50 reentrenos con labels permutadas dentro de cada mercado → delta **−1.10 ± 10.21** (min −22.23, max +19.09): el pipeline no fabrica señal. Secundario (umbral re-elegido por draw): el delta observado queda en p98.9 de ese nulo. **Caveat honesto:** el IC bootstrap por bloques de mercado del delta, **[−20.45, +121.74], incluye 0** (volatilidad de muestreo alta; 2 de 5 folds casi nulos) — se reporta ambos resultados. El gate sigue cerrado (nivel negativo, WFE negativa, cola+costos pendientes). Código y JSON: `microstructure/model/backtest_nulo_delta.py`, `results/nulo_delta_h300.json`. |
| **El auditor auditado: Monte Carlo de los 5 detectores de fugas (23-10-2026, alpha=0.01, 999 permutaciones/test)** | Sobre 200 estrategias nulas sin edge (4 tipos × 50 seeds, mercados de 90 días, spread 2×40 bps): **tasa de falsos positivos por detector — capturabilidad 0%, fills al book 0%, tuning in-sample 0%, kfold-cv 1.5%, supervivencia 5.0%; unión 6.0%**. **Potencia: 100% (unión) en las tres trampas de look-ahead** (completo: atrapado por tuning/kfold/supervivencia; de 1 barra: por los cinco; 50% parcial: capturabilidad 86%). Un edge real pero chico se acusa **0%** de las veces — los detectores cazan plumbing, no la existencia de edge. Limitación conocida: capturabilidad no ve un look-ahead de hold completo (1 barra de retraso quita solo 1/30 del peek); los otros cuatro detectores lo cubren. JSON: `results/monte_carlo_fp_power.json`. |
| **El auditor se audita a sí mismo: cinco estrategias ajenas × dos escalas de precio (23-10-2026, 3 honestas + 2 trampas de look-ahead, p0=0.5 vs p0=100)** | **Invariancia de escala restaurada y verificada**: el harness ahora escala el costo de round-trip con la escala del mercado (bps constantes en 80: `half_spread = HALF_SPREAD * p0 / P0`), y el rng de auditoría se siembra idéntico por estrategia entre escalas — 5/5 estrategias devuelven **veredictos idénticos en ambas escalas**, ratio de pnl exactamente 200.0×. Veredictos: las 2 trampas de look-ahead FALLAN/WARNEAN en ambas escalas (tuning in-sample, kfold, fills al book disparan); las 3 honestas PASAN salvo `spread_sniper`, marcada por supervivencia en ambas escalas — **limitación conocida del detector, documentada en el JSON** (`limitaciones_conocidas`): todo trade perdedor paga el mismo costo fijo de round-trip (masa puntual en −2·half_spread), lo que comprime el IQR e infla mecánicamente el conteo de outliers de cola derecha (`nr > nl` en los 7 marcados de 24 seeds de mercado; tasa de marca en honestas del 29.2% vs alpha=0.01; controles honestos momentum/ruido: 0/16 (bajo poder); misma tasa con pnl relativo — no es artefacto de unidades). Y no es un falso positivo aleatorio: contra el FP base de 5.0% del MC 4g, P(X>=7 | Bin(24,0.05)) = 1.3e-04 — `tail_symmetry_test` detecta CUALQUIER asimetría de pnl, incluida la estructural de estrategias honestas selectivas de nivel (su familia de nulos no las cubre), así que una marca sobre esta clase no implica trampa). Script y JSON del estudio: `demo/estudio_limitacion_supervivencia.py`, `results/estudio_limitacion_supervivencia.json`. El edge falso que antes reportaba `momentum_chaser` a p0=100 (media +0.086/trade) venía de un spread aditivo sobre precios lognormales — el fix de escalar el costo lo elimina. |

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

### Markout en bps del mid (2026-10-16 22:54 → 2026-10-18 23:42 UTC, 3,111,633 trades agresivos)

| Símbolo | media bps @5s | mediana bps @5s | fracción tóxica @5s | media bps @300s |
|---|---|---|---|---|
| **BTCUSDT** | +0.22 | +0.06 | 47% | +0.34 |
| **ETHUSDT** | +0.43 | +0.27 | 44% | +0.45 |
| **SOLUSDT** | **−0.06** | **−0.48** | **66%** | −0.02 |

**Descomposición (BTCUSDT @5s): deriva +0.75 bps vs entrada −0.53 bps.** Esto
cambia la lectura honesta del markout: el markout USD positivo que reportamos
primero sobre la ventana inicial de captura era **deriva del mercado, no
habilidad de ejecución** — el taker igual **paga el spread de entrada**; y para
un maker, el flujo que gana el agresor es adverse selection en su contra
(P&L del maker = −(markout del agresor)). SOL es tóxico directo a horizontes
cortos (mediana −0.48 bps, 66% de los trades taker
pierden).

**Limitación declarada — frescura del mid de salida:** el mid de salida se
reconstruye de un libro muestreado a ~2s; su staleness mediana es **1.14s** —
hasta ~23% del horizonte de 5s. Los markouts de horizonte corto son entonces
una cota inferior de precisión.

Regenerar con `microstructure/scripts/markout_binance.py` (solo lectura, DB
en vivo; último reporte machine-readable en
[`results/markout_report.json`](results/markout_report.json)).

## Próximos pasos (acotados)
- ~~MLP → LightGBM~~ **hecho** — superado por la re-corrída sin fugas (AUC 0.7384 pooled OOS).
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
  (PLAN_SIM_COLA.md) — el gate sigue cerrado.** El AUC rankea el flujo;
  convertirlo a P&L maker rentable (cola, rewards, fees) sigue abierto.
- Investigar por qué el ranking no traduce: P&L por horizonte/régimen,
  simulación con posición en cola, calibración del modelo de costos.

## Licencia
MIT — ver [LICENSE](LICENSE). Uso libre, atribución apreciada.

## Autor
Gustavo Ruarte — market microstructure & ML engineering.
Contacto: abrir un issue en GitHub o escribir a gustavoaruarte@gmail.com.
