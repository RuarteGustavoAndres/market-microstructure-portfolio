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

1. **85M+ eventos de libro y 382k+ fills reales** capturados 24/7 a hypertables
   de TimescaleDB (9,900+ mercados trackeados, 362 tokens activos en los
   últimos 7 días).
2. **Clasificador de toxicidad: AUC 0.765 pooled out-of-fold / 0.688
   por-mercado** (media de 119 mercados) — una capa de scoring/filtro, **no**
   un decisor.
3. **Veredicto walk-forward: SIN EDGE TODAVÍA** — con markout exacto y
   walk-forward temporal, el filtro no produce P&L neto positivo out-of-sample
   a ningún umbral (números completos en la tabla de abajo). El gate sigue
   cerrado y se mantiene el veredicto.

Estado de captura en vivo (generado a diario desde la base de producción):
[`results/CAPTURE_STATUS.md`](results/CAPTURE_STATUS.md).

Contacto: abrir un issue en GitHub.

## Lo más relevante
- **85M+ eventos de libro** (libro_eventos) + **382k+ fills** (trades_full)
  capturados 24/7 a hypertables de TimescaleDB — **9,900+ mercados trackeados,
  362 tokens activos en los últimos 7 días** (verificado 2026-10-18).
- **Cientos de miles de eventos de libro por hora, sostenido 24/7**, a una
  hypertable de TimescaleDB (compresión + retención).
- **Clasificador de toxicidad: LightGBM AUC 0.765 pooled out-of-fold**, **0.688
  media / 0.701 mediana por mercado sobre 119 mercados** — la brecha
  pooled/por-mercado es diferencia de tasas base, y se declara. Folds
  consistentes (0.75–0.79), sin fuga.
- Se detectó y eliminó una **trampa de fuga de features** que inflaba el AUC a
  0.702.
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
Clasificador de toxicidad (LightGBM)  →  AUC 0.765 pooled OOF / 0.688 por-mercado
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
- **Upgrade a LightGBM:** AUC pooled **0.765** out-of-fold, supera al MLP
  (0.751) — mejor para features tabulares de microestructura.
- **Limitaciones honestas:** rankea toxicidad (AUC) pero el accuracy binario
  está bajo el baseline mayoritario → es una capa de scoring/filtro, no un
  decisor autónomo. Y con markout exacto + walk-forward temporal, el filtro
  todavía **no** genera P&L neto positivo (abajo).

| Métrica | Valor |
|---|---|
| AUC-ROC pooled out-of-fold (LightGBM) | **0.765** |
| AUC-ROC out-of-fold (baseline MLP) | 0.751 |
| AUC por-mercado (media / mediana, 119 mercados evaluados) | **0.688 / 0.701** |
| Fills distintos | 107,362 |
| Filas de entrenamiento (107,362 fills × 4 horizontes) | 429,448 |
| **P&L neto walk-forward OOS (dataset 18-10-2026)** | **SIN EDGE TODAVÍA** — markout exacto, splits temporales, embargo 300s, 5 folds (tamaños de test 62,064 / 56,210 / 56,847 / 99,021 / 17,532 pares fill×horizonte). Sin filtro **−1.75 USD** · filtro @0.7 **−561.58** · @0.8 **−171.00** · umbral óptimo por fold (0.9/0.9/0.9/0.9/0.85 — estable) **−8.57**. Filtrar no mejora el P&L OOS a ningún umbral; el gate queda cerrado y el resultado se reporta igual. |

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
habilidad de ejecución** — el taker igual **paga el spread de entrada**. SOL es
tóxico directo a horizontes cortos (mediana −0.48 bps, 66% de los trades taker
pierden).

**Limitación declarada — frescura del mid de salida:** el mid de salida se
reconstruye de un libro muestreado a ~2s; su staleness mediana es **1.14s** —
hasta ~23% del horizonte de 5s. Los markouts de horizonte corto son entonces
una cota inferior de precisión.

Regenerar con `microstructure/scripts/markout_binance.py` (solo lectura, DB
en vivo; último reporte machine-readable en
[`results/markout_report.json`](results/markout_report.json)).

## Próximos pasos (acotados)
- ~~MLP → LightGBM~~ **hecho** — LightGBM (0.765 pooled) supera al MLP (0.751).
- ~~Backtestear el filtro contra P&L neto~~ **hecho, veredicto honesto: sin
  edge todavía** — con markout exacto (mid futuro real) y walk-forward temporal
  (splits contiguos, embargo 300s), el filtro no produce P&L OOS positivo a
  ningún umbral. El AUC rankea toxicidad; traducirlo a P&L neto sigue abierto.
- Investigar por qué el ranking no traduce: P&L por horizonte/régimen,
  simulación con posición en cola, calibración del modelo de costos.

## Licencia
MIT — ver [LICENSE](LICENSE). Uso libre, atribución apreciada.

## Autor
Gustavo Ruarte — market microstructure & ML engineering.
Contacto: abrir un issue en GitHub.
