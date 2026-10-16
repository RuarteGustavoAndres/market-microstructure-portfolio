# Market Microstructure: Real-Time Capture + Fill-Toxicity Model

> **EN** | [ES](#es) | [PT](#pt)

An end-to-end market microstructure pipeline for prediction-market CLOBs (Polymarket-style
binary order books): a 24/7 WebSocket order-book capture into TimescaleDB, plus a
machine-learning model that predicts **fill toxicity (adverse selection)** from order-book
features. Every number is real, out-of-sample — including the parts that did not work.

## Highlights
- **41.3M+ order-book events** (libro_eventos) + **173k+ fills** (trades_full) captured
  24/7 into TimescaleDB hypertables — **7,400+ markets tracked, 195 active tokens in
  the last 7 days** (verified 2026-10-16).
- **~600k order-book snapshots/hour** sustained, 24/7, into a TimescaleDB hypertable
  (compression + retention policies).
- **Toxicity classifier: LightGBM AUC 0.765 out-of-fold** on 224k labeled fills
  across 119 unseen markets (re-trained 2026-10-16; up from 0.688 at 42k samples —
  scaling data scaled the signal; fold AUCs are consistent 0.75–0.79, no leakage).
- Caught and removed a **feature-leakage trap** that had inflated AUC to 0.702.
- Grounded in the microstructure literature: OFI (Cont 2014), micro-price (Stoikov 2018),
  markout, Glosten-Milgrom / Kyle.

### Data status (declared, as of 2026-10-16)
- Continuous capture **2026-10-14 → present** (trades stream: no gaps).
- **Known outage:** order-book stream down **2026-10-15 ~17:00–20:00 UTC** (~3h,
  systemd services were not `enabled` and died with an interactive session; fixed —
  all capture services are now `systemctl --user enable`-ed with linger).
- Any analysis spanning that window must exclude or flag it. We say so because an
  honest dataset declares its holes.

## Architecture
```
WebSocket (market channel)
   │  asyncio producer → queue → consumer  (network never blocks on processing)
   ▼
In-memory order book per token (snapshot + incremental diffs)
   │  micro-batch flush: 500 rows / 2s
   ▼
TimescaleDB hypertables (libro_snapshots, trades_ml)
   │  feature engineering (OFI, microprice_bias, momentum, ...)
   ▼
Fill-toxicity classifier (PyTorch on CUDA)  →  honest OOF AUC 0.751 (LightGBM 0.765)
```

## Pipeline modules
| Module | Purpose |
|---|---|
| `capture/http.py` | ISP-fault-tolerant HTTP + pinned-IP DNS fallback (selective NXDOMAIN resilience) |
| `data/tsdb.py` | TimescaleDB connection + micro-batch flush helpers |
| `model/features_ml.py` | Feature engineering: OFI, micro-price, multi-horizon momentum, markout labels |
| `model/entrenar_toxicidad.py` | Classifier training: GroupKFold by market, per-market z-score, leakage-safe features |

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
- **41.3M+ eventos de libro** (libro_eventos) + **173k+ fills** (trades_full) capturados
  24/7 a TimescaleDB — **7,400+ mercados trackeados, 195 tokens activos en los últimos
  7 días** (verificado 2026-10-16).
- **~600k snapshots de libro/hora** sostenidos, 24/7, a una hypertable de TimescaleDB
  (compresión + retención).
- **Clasificador de toxicidad: LightGBM AUC 0.765 out-of-fold** sobre 224k fills
  etiquetados en 119 mercados no vistos (re-entrenado 2026-10-16; antes 0.688 con
  42k muestras — más data, más señal; folds consistentes 0.75–0.79, sin fuga).
- Se detectó y eliminó una **trampa de fuga de features** que inflaba el AUC a 0.702.
- Fundamentado en la literatura: OFI (Cont 2014), micro-precio (Stoikov 2018), markout,
  Glosten-Milgrom / Kyle.

## Arquitectura
```
WebSocket (canal market)
   │  productor asyncio → cola → consumidor  (la red nunca bloquea el procesamiento)
   ▼
Libro en memoria por token (snapshot + diffs incrementales)
   │  flush micro-batch: 500 filas / 2s
   ▼
Hypertables TimescaleDB (libro_snapshots, trades_ml)
   │  feature engineering (OFI, microprice_bias, momentum, ...)
   ▼
Clasificador de toxicidad (PyTorch en CUDA)  →  AUC OOF honesto 0.751 (LightGBM 0.765)
```

## Módulos
| Módulo | Propósito |
|---|---|
| `capture/http.py` | HTTP tolerante a fallas de ISP + fallback DNS por IP fijada (resiliencia a NXDOMAIN selectivo) |
| `data/tsdb.py` | Conexión TimescaleDB + helpers de flush micro-batch |
| `model/features_ml.py` | Feature engineering: OFI, micro-precio, momentum multi-horizonte, etiquetas de markout |
| `model/entrenar_toxicidad.py` | Entrenamiento: GroupKFold por mercado, z-score por mercado, features sin fuga |

## El modelo (números honestos)
- **Target:** markout de cada trade real a +horizonte (mid del libro vs precio de fill). Markout < 0 = selección adversa.
- **Etiquetado multi-horizonte:** cada fill etiquetado a 1s/10s/1m/5m → 224.520 muestras, 119 mercados (re-entreno 2026-10-16).
- **Validación:** GroupKFold por mercado (generalización a mercados no vistos) + normalización z-score por mercado.
- **Fuga eliminada:** `pos_en_spread` y `fill_price` reconstruyen el target por construcción — se quitaron al detectarla (corría el AUC 0.702→0.6425 en el dataset original).
- **Limitación honesta:** rankea toxicidad (AUC) pero el accuracy binario (65%) sigue bajo el baseline de clase mayoritaria (73%) → es una capa de scoring/filtro, no un decisor autónomo. **Y con markout exacto + walk-forward temporal, el filtro todavía NO genera P&L neto positivo** (-$225 sin filtro vs -$514 con filtro @0.7, OOS).

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
- **41.3M+ eventos de livro** (libro_eventos) + **173k+ fills** (trades_full) capturados
  24/7 em TimescaleDB — **7,400+ mercados rastreados, 195 tokens ativos nos últimos
  7 dias** (verificado 2026-10-16).
- **~600k snapshots de livro/hora** sustentados, 24/7, em uma hypertable do TimescaleDB
  (compressão + retenção).
- **Classificador de toxicidade: LightGBM AUC 0.765 out-of-fold** em 224k fills
  rotulados em 119 mercados não vistos (retreinado 2026-10-16; antes 0.688 com
  42k amostras — mais dados, mais sinal; folds consistentes 0.75–0.79, sem vazamento).
- Detectou-se e removeu-se uma **armadilha de vazamento de features** que inflava o AUC a 0.702.
- Fundamentado na literatura: OFI (Cont 2014), micro-preço (Stoikov 2018), markout,
  Glosten-Milgrom / Kyle.

## Arquitetura
```
WebSocket (canal market)
   │  produtor asyncio → fila → consumidor  (a rede nunca bloqueia o processamento)
   ▼
Livro em memória por token (snapshot + diffs incrementais)
   │  flush micro-batch: 500 linhas / 2s
   ▼
Hypertables TimescaleDB (libro_snapshots, trades_ml)
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
