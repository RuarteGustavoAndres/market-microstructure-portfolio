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
3. **Veredicto walk-forward: el filtro reduce consistentemente la selección
   adversa del tape en perspectiva maker — el delta de P&L del filtro sobre
   sin-filtro fue positivo en cada refresh sin fugas: **+$321** (19-10, 286k
   filas) y **+$376** (22-10, 383k filas), mientras el nivel sin filtro cambió
   de signo (−209 → +113) (todas las cifras son BRUTAS bajo supuestos de
   cinta: se cuenta cada fill del tape, sin posición en cola, sin fees, sin
   costos; el IC del delta queda pendiente del test nulo). Nivel actual
   filtrado @0.7: **+$489**. Folds por mercado (cuantil-0.5) dan +$747, techo
   optimista — ese esquema es fuera-de-tiempo pero no fuera-de-mercado. La
   simulación de cola ([PLAN_SIM_COLA.md](PLAN_SIM_COLA.md)) sigue pendiente
   para cerrar el gate; una primera corrida con cola fue **inconclusa** (AUC
   ~0.50-0.56 sobre features aproximadas, fuera de distribución). La
   eficiencia walk-forward es 35.4% por fila (8.0% por totales; filas = pares
   fill×horizonte; el refresh del 19-10 midió 9.5%/2.1% sobre su dataset de
   286k filas); los umbrales óptimos por fold siguen **inestables** (0.5-0.7),
   por eso corre el filtro fijo 0.7. Los cuatro horizontes positivos filtrados
   (+81 h1s / +146 h10s / +167 h60s / +95 h300s). El gate sigue cerrado.

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

## El modelo (números honestos)
- **Target:** markout de cada trade real a +horizonte (mid del libro vs precio
  de fill). Markout < 0 = selección adversa.
- **Etiquetado multi-horizonte:** cada fill etiquetado a 1s/10s/60s/300s. El
  set de entrenamiento es **107,362 fills distintos × 4 horizontes = 429,448
  filas** (cada fill aporta una fila por horizonte — son pares
  fill×horizonte, no fills distintos).
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
  decisor autónomo. Con markout exacto + walk-forward temporal, el filtro
  **sí** genera P&L BRUTO positivo bajo supuestos de cinta (+$489 @0.7 en el
  refresh del 22-10; delta +$376 sobre sin-filtro) — pero los umbrales por
  fold son inestables, y una primera simulación con cola (T2/T3, 18-10) fue
  **inconclusa**: el AUC del modelo sobre fills simulados es ~0.50-0.56
  (features aproximadas, fuera de distribución), así que los números de cinta
  asumen fills ideales ([PLAN_SIM_COLA.md](PLAN_SIM_COLA.md)). El gate sigue
  cerrado.

| Métrica | Valor |
|---|---|
| AUC-ROC pooled OOS (LightGBM, leak-free, mercados no vistos) | **0.7384** |
| PR-AUC OOS / prevalencia | **0.5057 / 0.304** |
| AUC por fold temporal (rango, 5 folds) | **0.689 / 0.786** |
| Filas walk-forward (tope de frescura, refresh 22-10-2026) | 382,582 (268,446 OOS) |
| **P&L maker walk-forward OOS (refresh 22-10-2026, 382,582 filas — bruto, supuestos de cinta; cola+costos pendientes)** | **POSITIVO, AÚN FRÁGIL** — markout exacto, splits temporales, embargo 300s: sin filtro **+113.47 USD** · filtro @0.7 **+488.98** (231,334 ops) · @0.8 **+216.82** · óptimo por fold (0.65/0.5/0.7/0.65/0.65 — **no estable**). WFE **35.4% por fila** (8.0% por totales; filas = pares fill×horizonte). GroupKFold por mercado (estimación optimista): sin filtro **+186** · umbral cuantil 0.5 **+746.93** (retención 50%) · @0.7 **+560.97**. Los cuatro horizontes positivos filtrados (+81 h1s / +146 h10s / +167 h60s / +95 h300s). Gate cerrado hasta simular cola + costos; números reportados igual. |

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
- ~~Backtestear el filtro contra P&L neto~~ **hecho sobre features sin fugas
  (snapshots con tope de frescura, estandarización causal, folds por mercado) —
  el filtro reduce la selección adversa del tape en perspectiva maker:
  **+$376 de delta de P&L sobre sin-filtro** (nivel @0.7: +$489; bruto,
  supuestos de cinta) en walk-forward temporal estricto (embargo 300s) —
  positivo en cada refresh sin fugas (+$321 el 19-10, sin filtro −209→+113
  cuando el dataset creció 286k→383k filas); +$747 en GroupKFold por mercado
  cuantil-0.5 (retención 50%) es una estimación optimista porque ese esquema
  es fuera-de-tiempo pero no fuera-de-mercado. La eficiencia walk-forward es
  **35.4% por fila** (8.0% por totales) pero los umbrales óptimos por fold
  siguen **inestables** (0.5-0.7) y falta simular la cola
  (PLAN_SIM_COLA.md) — el gate sigue cerrado.** El AUC rankea el flujo;
  convertirlo a P&L neto del maker (cola, rewards, fees) sigue abierto.
- Investigar por qué el ranking no traduce: P&L por horizonte/régimen,
  simulación con posición en cola, calibración del modelo de costos.

## Licencia
MIT — ver [LICENSE](LICENSE). Uso libre, atribución apreciada.

## Autor
Gustavo Ruarte — market microstructure & ML engineering.
Contacto: abrir un issue en GitHub o escribir a gustavoaruarte@gmail.com.
