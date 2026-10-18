# Market Microstructure: Real-Time Capture + Fill-Toxicity Model

> **EN** | [ES](#es) | [PT](#pt)

An end-to-end market microstructure pipeline for prediction-market CLOBs (Polymarket-style
binary order books): a 24/7 WebSocket order-book capture into TimescaleDB, plus a
machine-learning model that predicts **fill toxicity (adverse selection)** from order-book
features. Every number is real, out-of-sample — including the parts that did not work.

## Highlights
- **85M+ order-book events** (libro_eventos) + **382k+ fills** (trades_full) captured
  24/7 into TimescaleDB hypertables — **9,900+ markets tracked, 362 active tokens in
  the last 7 days** (verified 2026-10-18).
- **hundreds of thousands of order-book events per hour, sustained 24/7**, into a
  TimescaleDB hypertable (compression + retention policies).
- **Toxicity classifier: LightGBM AUC 0.765 out-of-fold** on 224k labeled fills
  across 119 unseen markets (re-trained 2026-10-16; up from 0.688 at 42k samples —
  scaling data scaled the signal; fold AUCs are consistent 0.75–0.79, no leakage).
- Caught and removed a **feature-leakage trap** that had inflated AUC to 0.702.
- Grounded in the microstructure literature: OFI (Cont 2014), micro-price (Stoikov 2018), markout, Glosten-Milgrom / Kyle.
- **Backtest Audit Demo** — a runnable, dependency-free teardown
  ([`demo/audit_demo.py`](demo/audit_demo.py),
  results in [`results/AUDIT_DEMO.md`](results/AUDIT_DEMO.md)): a synthetic
  backtest with planted look-ahead, free fills and in-sample tuning reports
  Sharpe 34; forced to obey reality (tradeable information, book-priced fills,
  walk-forward tuning) it loses exactly the spread. Five audit checks,
  self-verifying. This is the checklist behind the backtest-audit service.
- **Venue 2 live: Binance USDT-M perpetuals** (2026-10-16) — same pipeline,
  second asset class: aggTrades with aggressor side + top-5 book + funding
  for BTC/ETH/SOL perps, 24/7 systemd capture. Markout / adverse-selection
  analytics in [`analytics/adverse_selection.py`](analytics/adverse_selection.py).

### Data status (declared, as of 2026-10-16)
- Continuous capture **2026-10-14 → present** (trades stream: no gaps).
- **Known outage:** order-book stream down **2026-10-15 ~17:00–20:00 UTC** (~3h,
  systemd services were not `enabled` and died with an interactive session; fixed —
  all capture services are now `systemctl --user enable`-ed with linger).
- Any analysis spanning that window must exclude or flag it. We say so because an
  honest dataset declares its holes.
- **Live status:** [`results/CAPTURE_STATUS.md`](results/CAPTURE_STATUS.md) —
  machine-generated daily from the production database, so what you read is
  always <24h old.

## Architecture
```
WebSocket (market channel)
   │  asyncio producer → queue → consumer  (network never blocks on processing)
   ▼
In-memory order book per token (snapshot + incremental diffs)
   │  micro-batch flush: 500 rows / 2s
   ▼
TimescaleDB hypertables (libro_eventos, trades_full, libro_top5, market_meta)
   │  feature engineering (OFI, microprice_bias, momentum, ...)
   ▼
Fill-toxicity classifier (PyTorch on CUDA)  →  honest OOF AUC 0.751 (LightGBM 0.765)
```

## Pipeline modules
| Module | Purpose |
|---|---|
| `microstructure/capture/http.py` | ISP-fault-tolerant HTTP + pinned-IP DNS fallback (selective NXDOMAIN resilience) |
| `microstructure/data/tsdb.py` | TimescaleDB connection + micro-batch flush helpers |
| `microstructure/model/features_ml.py` | Feature engineering: OFI, micro-price, multi-horizon momentum, markout labels |
| `microstructure/model/entrenar_toxicidad.py` | Classifier training: GroupKFold by market, per-market z-score, leakage-safe features |

## The model (honest numbers)
- **Target:** markout of each real trade at +horizon (book mid vs fill). Markout < 0 = adversely selected.
- **Multi-horizon labeling:** every fill labeled at 1s/10s/1m/5m.
- **Validation:** GroupKFold by market (unseen-market generalization) + per-market z-score normalization.
- **Leakage removed:** `pos_en_spread` and `fill_price` reconstruct the target by construction — dropped them, AUC fell from 0.702 to the honest **0.6425** (MLP).
- **LightGBM upgrade:** AUC **0.765** out-of-fold on the grown dataset (224k fills / 119 markets, re-trained 2026-10-16), beating the MLP (0.751) — better for tabular microstructure features.
- **Honest limitation:** ranks toxicity (AUC) but binary accuracy sits below the majority baseline → it is a scoring/filter layer, not an autonomous decision-maker. **And under exact-markout walk-forward, the filter does not yet produce positive net P&L** (-$225 unfiltered vs -$514 @0.7, OOS).

| Metric | Value |
|---|---|
| AUC-ROC out-of-fold (LightGBM) | **0.765** |
| AUC-ROC out-of-fold (MLP baseline) | 0.751 |
| AUC per-market (mean / median) | 0.688 / 0.701 |
| Samples / markets (2026-10-16 retrain) | 224,520 / 119 |
| **Walk-forward vs net P&L** | **no edge yet**: exact-markout walk-forward (temporal splits, 300s embargo) → OOS P&L -$225 without filter, -$514 with filter @0.7. The earlier spread-approximated backtest showed +$361 — an artifact. Gate stays closed; reported anyway. |
| Domain academic ceiling | ~0.586 |

## Venue 2 — Binance USDT-M perpetuals (live since 2026-10-16)
Second venue feeding the same TimescaleDB stack, same order-book sync protocol
(snapshot + diff with update-id chain verification, gap detection, resync —
the official Binance algorithm): aggTrades with aggressor side, top-5 book at
2s cadence, and funding/mark-price every 5 minutes for BTCUSDT/ETHUSDT/SOLUSDT.
Captured 24/7 by a systemd service (public streams, no account involved).
Preliminary sustained rates: ~15 book rows/s + ~9 aggTrades/s across 3 symbols,
event→ingest latency ~0.3s. Execution-quality analytics (pure functions, tested):
[`analytics/adverse_selection.py`](analytics/adverse_selection.py) — markout at
+5s/+30s/+60s and the gain/pain ratio per horizon, no look-ahead (the mid used
is the last one known at or before t+h; trades whose horizon passes the last
quote are excluded, never extrapolated).
Venue 1 (Polymarket CLOB, above) = discrete-event prediction markets; this
venue 2 = continuous high-frequency books — same schema, same analytics.
First real markout over ~12h of continuous capture (Fri night → Sat morning,
**still preliminary: single-session window, weekend microstructure, do not
read as edge**; the definitive rerun covers 72h incl. a full weekday):
mean markout per aggressive trade at +5s / +30s / +60s —
**BTCUSDT** +1.66 / +2.18 / +1.84 USD (63% / 53% / 50% toxic),
**ETHUSDT** +0.069 / +0.084 / +0.077 USD (52% / 47% / 48% toxic),
**SOLUSDT** −0.0039 / −0.0040 / −0.0045 USD (78% / 58% / 55% toxic —
taker flow still donates the spread, but toxicity decays with horizon).
~640k aggressive trades evaluated. Toxicity at +5s > +30s > +60s for every
symbol: aggressive flow pays the spread instantly and the markout mean
recovers with time — consistent with spread compensation, not informed
predation. Regenerate with `microstructure/scripts/markout_binance.py` (read-only, live DB).

## Next steps (scoped)
- ~~MLP → LightGBM~~ **done** — LightGBM (AUC 0.765) beats the MLP (0.751).
- ~~Backtest the toxicity filter against net P&L~~ **done, honest verdict: no edge yet** —
  with exact markout (real future mid) and **temporal walk-forward** (contiguous splits,
  300s embargo), the filter does NOT produce positive OOS P&L at any threshold
  (-$225 unfiltered vs -$514 @0.7). The earlier spread-approximated K-fold backtest
  that showed +$361 was an artifact — exactly the failure mode the literature warns
  about. AUC ranks toxicity; translating it into net P&L remains open work.
- Investigate why ranking skill does not translate: per-horizon/per-regime P&L,
  queue-position-aware simulation, cost model calibration.

---
<a name="es"></a>

# Microestructura de mercado: captura en tiempo real + modelo de toxicidad de fills

> [EN](#) | **ES** | [PT](#pt)

Pipeline de microestructura para CLOBs de mercados de predicción (libros de órdenes
binarios tipo Polymarket): captura 24/7 del libro por WebSocket a TimescaleDB, más un
modelo de ML que predice la **toxicidad de fills (selección adversa)** desde features del
libro. Todos los números son reales y out-of-sample — incluidas las partes que no funcionaron.

## Lo más relevante
- **85M+ eventos de libro** (libro_eventos) + **382k+ fills** (trades_full) capturados
  24/7 a TimescaleDB — **9,900+ mercados trackeados, 362 tokens activos en los últimos
  7 días** (verificado 2026-10-18).
- **cientos de miles de eventos de libro por hora, sostenido 24/7**, a una hypertable
  de TimescaleDB (compresión + retención).
- **Clasificador de toxicidad: LightGBM AUC 0.765 out-of-fold** sobre 224k fills
  etiquetados en 119 mercados no vistos (re-entrenado 2026-10-16; antes 0.688 con
  42k muestras — más data, más señal; folds consistentes 0.75–0.79, sin fuga).
- Se detectó y eliminó una **trampa de fuga de features** que inflaba el AUC a 0.702.
- Fundamentado en la literatura: OFI (Cont 2014), micro-precio (Stoikov 2018), markout,
  Glosten-Milgrom / Kyle.
- **Demo de auditoría de backtests** — teardown ejecutable, sin dependencias
  ([`demo/audit_demo.py`](demo/audit_demo.py), resultados en
  [`results/AUDIT_DEMO.md`](results/AUDIT_DEMO.md)): un backtest sintético con
  look-ahead, fills gratis y tuning in-sample reporta Sharpe 34; obligado a obedecer
  la realidad (información ejecutable, fills al libro, walk-forward) pierde exactamente
  el spread. Cinco checks de auditoría, auto-verificados. Ese es el checklist del
  servicio de auditoría.
- **Venue 2 activo: perpetuos USDT-M de Binance** (16-10-2026) — mismo pipeline,
  segunda clase de activo: aggTrades con lado agresor + libro top-5 + funding de
  BTC/ETH/SOL perp, captura systemd 24/7. Analítica de markout/selección adversa
  en [`analytics/adverse_selection.py`](analytics/adverse_selection.py).

## Arquitectura
```
WebSocket (canal market)
   │  productor asyncio → cola → consumidor  (la red nunca bloquea el procesamiento)
   ▼
Libro en memoria por token (snapshot + diffs incrementales)
   │  flush micro-batch: 500 filas / 2s
   ▼
Hypertables TimescaleDB (libro_eventos, trades_full, libro_top5, market_meta)
   │  feature engineering (OFI, microprice_bias, momentum, ...)
   ▼
Clasificador de toxicidad (PyTorch en CUDA)  →  AUC OOF honesto 0.751 (LightGBM 0.765)
```

## Módulos
| Módulo | Propósito |
|---|---|
| `microstructure/capture/http.py` | HTTP tolerante a fallas de ISP + fallback DNS por IP fijada (resiliencia a NXDOMAIN selectivo) |
| `microstructure/data/tsdb.py` | Conexión TimescaleDB + helpers de flush micro-batch |
| `microstructure/model/features_ml.py` | Feature engineering: OFI, micro-precio, momentum multi-horizonte, etiquetas de markout |
| `microstructure/model/entrenar_toxicidad.py` | Entrenamiento: GroupKFold por mercado, z-score por mercado, features sin fuga |

## El modelo (números honestos)
- **Target:** markout de cada trade real a +horizonte (mid del libro vs precio de fill). Markout < 0 = selección adversa.
- **Etiquetado multi-horizonte:** cada fill etiquetado a 1s/10s/1m/5m → 224.520 muestras, 119 mercados (re-entreno 2026-10-16).
- **Validación:** GroupKFold por mercado (generalización a mercados no vistos) + normalización z-score por mercado.
- **Fuga eliminada:** `pos_en_spread` y `fill_price` reconstruyen el target por construcción — se quitaron al detectarla (corría el AUC 0.702→0.6425 en el dataset original).
- **Limitación honesta:** rankea toxicidad (AUC) pero el accuracy binario (65%) sigue bajo el baseline de clase mayoritaria (73%) → es una capa de scoring/filtro, no un decisor autónomo. **Y con markout exacto + walk-forward temporal, el filtro todavía NO genera P&L neto positivo** (-$225 sin filtro vs -$514 con filtro @0.7, OOS).

## Venue 2 — perpetuos USDT-M de Binance (activo desde el 16-10-2026)
Segundo venue alimentando el mismo stack de TimescaleDB, mismo protocolo de
sincronización de libro (snapshot + diffs con verificación de cadena de
update-ids, detección de gaps, resync — el algoritmo oficial de Binance):
aggTrades con lado agresor, libro top-5 cada 2s, y funding/mark-price cada
5 minutos para BTCUSDT/ETHUSDT/SOLUSDT. Capturado 24/7 por un servicio
systemd (streams públicos, sin cuenta involucrada). Tasas sostenidas
preliminares: ~15 filas de libro/s + ~9 aggTrades/s en 3 símbolos, latencia
evento→ingesta ~0.3s. Analítica de calidad de ejecución (funciones puras,
testeada): [`analytics/adverse_selection.py`](analytics/adverse_selection.py)
— markout a +5s/+30s/+60s y ratio gain/pain por horizonte, sin look-ahead
(el mid usado es el último conocido en o antes de t+h; los trades cuyo
horizonte pasa la última cita se excluyen, nunca se extrapolan).
Venue 1 (CLOB de Polymarket, arriba) = mercados de predicción de eventos
discretos; este venue 2 = libros continuos de alta frecuencia — mismo
schema, misma analítica. Primer markout real sobre las primeras ~12h de
captura (**preliminar, pocas horas de data, sesgo de tendencia — no leer
como edge**): markout medio a +5s BTCUSDT +2.67 USD (54% tóxico),
ETHUSDT +0.09 (46%), SOLUSDT -0.004 (83% tóxico — la firma clásica de
flujo taker que dona el spread).

## Próximos pasos
- ~~MLP → LightGBM~~ **hecho** — LightGBM AUC OOF **0.765** (MLP 0.751).
- ~~Escalar a 50k-100k fills~~ **hecho** — 224k muestras / 119 mercados (2026-10-16).
- ~~Backtestear el filtro contra P&L neto~~ **hecho, veredicto honesto: sin edge todavía** —
  con markout exacto (mid futuro real) y **walk-forward temporal** (splits contiguos,
  embargo 300s), el filtro NO produce P&L OOS positivo a ningún umbral
  (-$225 sin filtro vs -$514 @0.7). El backtest anterior (+$361, aproximado por spread
  y K-fold aleatorio) era un artefacto — el modo de falla que la literatura advierte.
  El AUC rankea toxicidad; traducirlo a P&L neto sigue abierto.
- Investigar por qué el ranking no traduce: P&L por horizonte/régimen, simulación
  con posición en cola, calibración del modelo de costos.

---
<a name="pt"></a>

# Microestrutura de mercado: captura em tempo real + modelo de toxicidade de fills

> [EN](#) | [ES](#es) | **PT**

Pipeline de microestrutura para CLOBs de mercados de predição (livros de ordens binários
tipo Polymarket): captura 24/7 do livro via WebSocket para TimescaleDB, mais um modelo de
ML que prevê a **toxicidade de fills (seleção adversa)** a partir de features do livro.
Todos os números são reais e out-of-sample — incluídas as partes que não funcionaram.

## Destaques
- **85M+ eventos de livro** (libro_eventos) + **382k+ fills** (trades_full) capturados
  24/7 em TimescaleDB — **9,900+ mercados rastreados, 362 tokens ativos nos últimos
  7 dias** (verificado 2026-10-18).
- **centenas de milhares de eventos de livro por hora, sustentado 24/7**, em uma
  hypertable do TimescaleDB (compressão + retenção).
- **Classificador de toxicidade: LightGBM AUC 0.765 out-of-fold** em 224k fills
  rotulados em 119 mercados não vistos (retreinado 2026-10-16; antes 0.688 com
  42k amostras — mais dados, mais sinal; folds consistentes 0.75–0.79, sem vazamento).
- Detectou-se e removeu-se uma **armadilha de vazamento de features** que inflava o AUC a 0.702.
- Fundamentado na literatura: OFI (Cont 2014), micro-preço (Stoikov 2018), markout,
  Glosten-Milgrom / Kyle.
- **Demo de auditoria de backtests** — teardown executável, sem dependências
  ([`demo/audit_demo.py`](demo/audit_demo.py), resultados em
  [`results/AUDIT_DEMO.md`](results/AUDIT_DEMO.md)): um backtest sintético com
  look-ahead, fills grátis e tuning in-sample reporta Sharpe 34; obrigado a obedecer
  a realidade (informação executável, fills ao livro, walk-forward) perde exatamente
  o spread. Cinco verificações de auditoria, auto-verificadas. Esse é o checklist do
  serviço de auditoria.
- **Venue 2 ativo: perpétuos USDT-M da Binance** (16-10-2026) — mesmo pipeline,
  segunda classe de ativo: aggTrades com lado agressor + livro top-5 + funding de
  BTC/ETH/SOL perp, captura systemd 24/7. Analítica de markout/seleção adversa
  em [`analytics/adverse_selection.py`](analytics/adverse_selection.py).

## Arquitetura
```
WebSocket (canal market)
   │  produtor asyncio → fila → consumidor  (a rede nunca bloqueia o processamento)
   ▼
Livro em memória por token (snapshot + diffs incrementais)
   │  flush micro-batch: 500 linhas / 2s
   ▼
Hypertables TimescaleDB (libro_eventos, trades_full, libro_top5, market_meta)
   │  feature engineering (OFI, microprice_bias, momentum, ...)
   ▼
Classificador de toxicidade (PyTorch em CUDA)  →  AUC OOF honesto 0.751 (LightGBM 0.765)
```

## O modelo (números honestos)
- **Target:** markout de cada trade real a +horizonte (mid do livro vs preço de fill). Markout < 0 = seleção adversa.
- **Rotulagem multi-horizonte:** cada fill rotulado a 1s/10s/1m/5m → 224.520 amostras, 119 mercados (retreino 2026-10-16).
- **Validação:** GroupKFold por mercado (generalização a mercados não vistos) + normalização z-score por mercado.
- **Vazamento removido:** `pos_en_spread` e `fill_price` reconstroem o target por construção — removidos ao detectá-lo (inflava AUC 0.702→0.6425 no dataset original).
- **Limitação honesta:** ordena toxicidade (AUC) mas a acurácia binária (65%) fica abaixo do baseline majoritário (73%) → é uma camada de scoring/filtro, não um decisor autônomo. **E com markout exato + walk-forward temporal, o filtro ainda NÃO gera P&L líquido positivo** (-$225 sem filtro vs -$514 com filtro @0.7, OOS).

## Venue 2 — perpétuos USDT-M da Binance (ativo desde 16-10-2026)
Segundo venue alimentando o mesmo stack de TimescaleDB, mesmo protocolo de
sincronização de livro (snapshot + diffs com verificação de cadeia de
update-ids, detecção de gaps, resync — o algoritmo oficial da Binance):
aggTrades com lado agressor, livro top-5 a cada 2s, e funding/mark-price a
cada 5 minutos para BTCUSDT/ETHUSDT/SOLUSDT. Capturado 24/7 por um serviço
systemd (streams públicos, sem conta envolvida). Taxas sustentadas
preliminares: ~15 linhas de livro/s + ~9 aggTrades/s em 3 símbolos, latência
evento→ingestão ~0,3s. Analítica de qualidade de execução (funções puras,
testada): [`analytics/adverse_selection.py`](analytics/adverse_selection.py)
— markout a +5s/+30s/+60s e razão gain/pain por horizonte, sem look-ahead
(o mid usado é o último conhecido em ou antes de t+h; trades cujo horizonte
passa a última cotação são excluídos, nunca extrapolados).
Venue 1 (CLOB da Polymarket, acima) = mercados de predição de eventos
discretos; este venue 2 = livros contínuos de alta frequência — mesmo
schema, mesma analítica. Primeiro markout real sobre as primeiras ~12h de
captura (**preliminar, poucas horas de dados, viés de tendência — não ler
como edge**): markout médio a +5s BTCUSDT +2.67 USD (54% tóxico),
ETHUSDT +0.09 (46%), SOLUSDT -0.004 (83% tóxico — a assinatura clássica
de fluxo taker que doa o spread).

## Próximos passos
- ~~MLP → LightGBM~~ **feito** — LightGBM AUC OOF **0.765** (MLP 0.751).
- ~~Escalar para 50k-100k fills~~ **feito** — 224k amostras / 119 mercados (2026-10-16).
- ~~Backtest do filtro contra P&L líquido~~ **feito, veredicto honesto: sem edge ainda** —
  com markout exato (mid futuro real) e **walk-forward temporal**, o filtro NÃO gera
  P&L OOS positivo em nenhum limiar (-$225 sem filtro vs -$514 @0.7). O backtest
  anterior (+$361, aproximado por spread e K-fold aleatório) era artefato.
  O AUC ranqueia toxicidade; traduzi-lo em P&L líquido segue em aberto.
- Investigar por que o ranking não traduz: P&L por horizonte/regime, simulação
  com posição na fila, calibração do modelo de custos.

---

## License / Licencia / Licença
MIT — see [LICENSE](LICENSE). Use freely, attribution appreciated.

## Author / Autor
Gustavo Ruarte — market microstructure & ML engineering.
Contact / Contacto / Contato: available on request.
