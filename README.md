# Market Microstructure: Real-Time Capture + Fill-Toxicity Model

> **EN** | [ES](#es) | [PT](#pt)

An end-to-end market microstructure pipeline for prediction-market CLOBs (Polymarket-style
binary order books): a 24/7 WebSocket order-book capture into TimescaleDB, plus a
machine-learning model that predicts **fill toxicity (adverse selection)** from order-book
features. Every number is real, out-of-sample — including the parts that did not work.

## Highlights
- **~600k order-book snapshots/hour** sustained, 24/7, into a TimescaleDB hypertable
  (15M+ rows, compression + retention policies).
- **Toxicity classifier: AUC 0.688 out-of-fold** (LightGBM) across unseen markets —
  real, honest signal, above the MLP baseline (0.642).
- Caught and removed a **feature-leakage trap** that had inflated AUC to 0.702.
- Grounded in the microstructure literature: OFI (Cont 2014), micro-price (Stoikov 2018),
  markout, Glosten-Milgrom / Kyle.

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
Fill-toxicity classifier (PyTorch on CUDA)  →  honest AUC 0.642
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
- **LightGBM upgrade:** AUC **0.688** out-of-fold, beating the MLP — better for tabular microstructure features.
- **Honest limitation:** ranks toxicity (AUC) but binary accuracy sits below the majority baseline → it is a scoring/filter layer, not an autonomous decision-maker.

| Metric | Value |
|---|---|
| AUC-ROC out-of-fold | 0.6425 |
| AUC per-market (mean) | 0.6098 |
| Samples / markets | 41,828 / 45 |
| Domain academic ceiling | ~0.586 |

## Next steps (scoped)
- MLP → LightGBM (better for ~40k tabular samples: accuracy, latency, interpretability).
- Scale to 50k-100k fills.
- Backtest the toxicity filter against **net P&L**, not just AUC.

---
<a name="es"></a>

# Microestructura de mercado: captura en tiempo real + modelo de toxicidad de fills

> [EN](#) | **ES** | [PT](#pt)

Pipeline de microestructura para CLOBs de mercados de predicción (libros de órdenes
binarios tipo Polymarket): captura 24/7 del libro por WebSocket a TimescaleDB, más un
modelo de ML que predice la **toxicidad de fills (selección adversa)** desde features del
libro. Todos los números son reales y out-of-sample — incluidas las partes que no funcionaron.

## Lo más relevante
- **~600k snapshots de libro/hora** sostenidos, 24/7, a una hypertable de TimescaleDB
  (15M+ filas, compresión + retención).
- **Clasificador de toxicidad: AUC 0.688 out-of-fold** (LightGBM) en mercados no vistos —
  señal real y honesta, por encima del baseline MLP (0.642).
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
Clasificador de toxicidad (PyTorch en CUDA)  →  AUC honesto 0.642
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
- **Etiquetado multi-horizonte:** cada fill etiquetado a 1s/10s/1m/5m → 41.828 muestras, 45 mercados.
- **Validación:** GroupKFold por mercado (generalización a mercados no vistos) + normalización z-score por mercado.
- **Fuga eliminada:** `pos_en_spread` y `fill_price` reconstruyen el target por construcción — se quitaron, el AUC cayó de 0.702 al honesto **0.6425**.
- **Limitación honesta:** rankea toxicidad (AUC) pero el accuracy binario (57%) está bajo el baseline de clase mayoritaria (69%) → es una capa de scoring/filtro, no un decisor autónomo.

## Próximos pasos
- MLP → LightGBM (mejor para ~40k muestras tabulares: accuracy, latencia, interpretabilidad).
- Escalar a 50k-100k fills.
- Backtestear el filtro de toxicidad contra el **P&L neto**, no solo el AUC.

---
<a name="pt"></a>

# Microestrutura de mercado: captura em tempo real + modelo de toxicidade de fills

> [EN](#) | [ES](#es) | **PT**

Pipeline de microestrutura para CLOBs de mercados de predição (livros de ordens binários
tipo Polymarket): captura 24/7 do livro via WebSocket para TimescaleDB, mais um modelo de
ML que prevê a **toxicidade de fills (seleção adversa)** a partir de features do livro.
Todos os números são reais e out-of-sample — incluídas as partes que não funcionaram.

## Destaques
- **~600k snapshots de livro/hora** sustentados, 24/7, em uma hypertable do TimescaleDB
  (15M+ linhas, compressão + retenção).
- **Classificador de toxicidade: AUC 0.688 out-of-fold** (LightGBM) em mercados não vistos —
  sinal real e honesto, acima do baseline MLP (0.642).
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
Classificador de toxicidade (PyTorch em CUDA)  →  AUC honesto 0.642
```

## O modelo (números honestos)
- **Target:** markout de cada trade real a +horizonte (mid do livro vs preço de fill). Markout < 0 = seleção adversa.
- **Rotulagem multi-horizonte:** cada fill rotulado a 1s/10s/1m/5m → 41.828 amostras, 45 mercados.
- **Validação:** GroupKFold por mercado (generalização a mercados não vistos) + normalização z-score por mercado.
- **Vazamento removido:** `pos_en_spread` e `fill_price` reconstroem o target por construção — foram removidos, o AUC caiu de 0.702 para o honesto **0.6425**.
- **Limitação honesta:** ordena toxicidade (AUC) mas a acurácia binária (57%) fica abaixo do baseline majoritário (69%) → é uma camada de scoring/filtro, não um decisor autônomo.

## Próximos passos
- MLP → LightGBM (melhor para ~40k amostras tabulares: acurácia, latência, interpretabilidade).
- Escalar para 50k-100k fills.
- Fazer backtest do filtro de toxicidade contra o **P&L líquido**, não só o AUC.

---

## License / Licencia / Licença
MIT — see [LICENSE](LICENSE). Use freely, attribution appreciated.

## Author / Autor
Gustavo Ruarte — market microstructure & ML engineering.
Contact / Contacto / Contato: available on request.
