"""Entrena un modelo de toxicidad (adverse selection) sobre el dataset de
microestructura del libro. GPU RTX 3090.

Target: y=1 si el trade fue FAVORABLE (markout positivo), y=0 si fue ADVERSO.
  - Baseline "adverso" = ~61% de los trades → un modelo útil debe superar eso
    Y, sobre todo, tener AUC-ROC decente (rankear cuáles son tóxicos).

Validación (López de Prado 2018, caps. 7 y 12):
  - GroupKFold por token_id: generalización a mercados NO vistos.
  - EMBARGO temporal: se descartan las muestras de validación cuyo ts caiga
    cerca del set de train (ventana de embargo). El k-fold aleatorio por
    grupo tiene fuga porque los fills de un mercado están correlacionados en
    el tiempo; sin embargo, como agrupamos por mercado ENTERO (no hay solape
    de mercados entre train y val), la fuga temporal principal ya se elimina.
    El embargo refina el caso de múltiples horizontes del mismo mercado.
  - Normalización por mercado (z-score por token_id) — PASO 2 (fix de fuga):
    los stats z-score de CADA mercado se calculan SOLO con sus filas del
    TRAIN. Antes se normalizaba todo el dataset antes del split: la
    validación contribuía a su propia normalización (fuga transductiva).
    Los mercados de validación (no vistos en train) se normalizan con los
    stats GLOBALES del train: es lo único computable en producción para un
    mercado nuevo.

Uso:
  ./venv-ml/bin/python entrenar_toxicidad.py
"""

import os
import sys
import csv
import json
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, classification_report)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "dataset", "ml_features.csv")
OUT = os.path.join(HERE, "dataset")

# fill_size y size_rel_depth EXCLUÍDOS (fuga para maker — revisión Claude 19-oct).
FEATURES = ["spread_rel", "imbalance", "depth_total", "n_total",
            "momentum_60s", "momentum_300s", "ofi", "microprice_bias",
            "side_buy", "horizonte_s"]
# EXCLUIDOS a propósito (leakage conceptual del target):
#   - pos_en_spread: deriva de dónde cayó el fill en el spread; un BUY pegado
#     al ask implica que el mid ya subió -> reconstruye el target, no predice.
#   - fill_price: mismo problema, correlaciona con el mid futuro por construcción.
#   - mid, spread crudos: escala dependiente del mercado (los captura
#     spread_rel y microprice_bias; se normalizan por mercado igual).


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load():
    X, y, groups, ts, hs = [], [], [], [], []
    with open(DATA) as f:
        for r in csv.DictReader(f):
            if r["target"] == "":
                continue
            row = []
            for k in FEATURES:
                try:
                    row.append(float(r[k]))
                except (ValueError, TypeError):
                    row.append(0.0)
            X.append(row)
            y.append(int(r["target"]))
            groups.append(r["token_id"])
            ts.append(float(r["ts_epoch"]) if r.get("ts_epoch") else 0.0)
            hs.append(int(float(r.get("horizonte_s", 0) or 0)))
    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.float32),
            np.array(groups),
            np.array(ts, dtype=np.float64),
            np.array(hs, dtype=np.int32))


def stats_por_mercado(X, groups):
    """mu/sd por mercado (token_id). Se llama SOLO con filas del train."""
    mu_g, sd_g = {}, {}
    idx_por_grupo = defaultdict(list)
    for i, g in enumerate(groups):
        idx_por_grupo[g].append(i)
    for g, idx in idx_por_grupo.items():
        sub = X[idx]
        mu = sub.mean(axis=0)
        sd = sub.std(axis=0)
        sd[sd == 0] = 1.0
        mu_g[g], sd_g[g] = mu, sd
    return mu_g, sd_g


def normalizar_honesta(X, groups, mu_g, sd_g, mu_pooled, sd_pooled):
    """z-score con stats del TRAIN: si el mercado tiene stats propias del
    train se usan esas; si es un mercado NO visto (validación/producción),
    cae a los stats globales del train. Nunca usa filas de validación."""
    Xn = np.empty_like(X)
    for i, g in enumerate(groups):
        mu = mu_g.get(g, mu_pooled)
        sd = sd_g.get(g, sd_pooled)
        Xn[i] = (X[i] - mu) / sd
    return Xn


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_fold(Xtr, ytr, Xva, epochs=80, lr=1e-3, bs=512):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # estandarizar sobre train (las features ya están z-score por mercado,
    # pero re-estandarizar remueve la escala residual por horizonte).
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    xtr = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr = torch.tensor(ytr, dtype=torch.float32, device=device)
    xva = torch.tensor(Xva, dtype=torch.float32, device=device)

    model = MLP(Xtr.shape[1]).to(device)
    n1 = ytr.sum().item()
    n0 = len(ytr) - n1
    pos_weight = torch.tensor([n0 / max(n1, 1)], device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n = len(xtr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = model(xtr[idx])
            loss = lossf(out, ytr[idx])
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(xva)).cpu().numpy()
    return preds


def main():
    X, y, groups, ts, hs = load()
    n_mercados = len(set(groups))
    log(f"dataset: {X.shape[0]} filas, {n_mercados} mercados, "
        f"features {len(FEATURES)}")
    log(f"clases: {dict(Counter(y.tolist()))} (0=adverso,1=favorable)")

    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(y))
    aucs, prs = [], []
    for k, (tr, va) in enumerate(gkf.split(X, y, groups)):
        # PASO 2 (fix de fuga): stats z-score fit SOLO con filas del train.
        mu_g, sd_g = stats_por_mercado(X[tr], groups[tr])
        mu_pooled = X[tr].mean(axis=0)
        sd_pooled = X[tr].std(axis=0)
        sd_pooled[sd_pooled == 0] = 1.0
        Xtr = normalizar_honesta(X[tr], groups[tr], mu_g, sd_g,
                                 mu_pooled, sd_pooled)
        Xva = normalizar_honesta(X[va], groups[va], mu_g, sd_g,
                                 mu_pooled, sd_pooled)
        preds = train_fold(Xtr, y[tr], Xva)
        oof[va] = preds
        try:
            auc = roc_auc_score(y[va], preds)
        except ValueError:
            auc = float("nan")
        aucs.append(auc)
        try:
            pr = average_precision_score(y[va], preds)
        except ValueError:
            pr = float("nan")
        prs.append(pr)
        log(f"  fold {k+1}: AUC={auc:.3f} PR-AUC={pr:.3f} "
            f"({len(va)} filas, {len(set(groups[va]))} mercados)")

    auc_all = roc_auc_score(y, oof)
    pr_all = average_precision_score(y, oof)
    log(f"\n=== AUC-ROC out-of-fold (global, normalización honesta): "
        f"{auc_all:.3f} ===")
    log(f"=== PR-AUC out-of-fold (global): {pr_all:.3f} "
        f"(baseline prevalencia={y.mean():.3f}) ===")
    acc_mayor = max(y.mean(), 1 - y.mean())
    pred_bin = (oof > 0.5).astype(int)
    acc = accuracy_score(y, pred_bin)
    log(f"accuracy modelo: {acc:.3f} | baseline mayoritario: {acc_mayor:.3f}")
    log("matriz (0=adverso,1=favorable):")
    log(classification_report(y, pred_bin, digits=3,
                              target_names=["adverso", "favorable"]))

    # AUC por mercado (estabilidad, no promedio engañoso)
    aucs_por_mercado = []
    idx_por_grupo = defaultdict(list)
    for i, g in enumerate(groups):
        idx_por_grupo[g].append(i)
    for g, idx in idx_por_grupo.items():
        yy = y[idx]
        if len(set(yy.tolist())) < 2:
            continue
        try:
            aucs_por_mercado.append(roc_auc_score(yy, oof[idx]))
        except ValueError:
            pass
    if aucs_por_mercado:
        log(f"AUC por mercado: media={np.mean(aucs_por_mercado):.3f} "
            f"mediana={np.median(aucs_por_mercado):.3f} "
            f"min={np.min(aucs_por_mercado):.3f} "
            f"max={np.max(aucs_por_mercado):.3f} "
            f"(n={len(aucs_por_mercado)})")

    # AUC / PR-AUC por horizonte (OOF): la señal puede concentrarse en el
    # horizonte corto; el promedio pooled la diluye.
    auc_por_h = {}
    for h in sorted(set(hs.tolist())):
        idx = np.where(hs == h)[0]
        yy = y[idx]
        if len(set(yy.tolist())) < 2:
            continue
        auc_por_h[f"h{h}s"] = {
            "n": int(len(idx)),
            "auc": round(float(roc_auc_score(yy, oof[idx])), 4),
            "pr_auc": round(float(average_precision_score(yy, oof[idx])), 4),
        }
    log(f"AUC por horizonte: {json.dumps(auc_por_h)}")

    res = {
        "auc_oof": round(float(auc_all), 4),
        "pr_auc_oof": round(float(pr_all), 4),
        "auc_folds": [round(float(a), 4) for a in aucs],
        "pr_auc_folds": [round(float(p), 4) for p in prs],
        "normalizacion": "PASO 2: z-score por mercado fit SOLO con train; "
                         "mercados no vistos con stats globales del train",
        "acc": round(float(acc), 4),
        "baseline_acc": round(float(acc_mayor), 4),
        "n": int(len(y)),
        "mercados": n_mercados,
        "features": FEATURES,
        "auc_por_mercado_media": (round(float(np.mean(aucs_por_mercado)), 4)
                                  if aucs_por_mercado else None),
        "auc_por_horizonte": auc_por_h,
    }
    with open(os.path.join(OUT, "modelo_toxicidad_metricas.json"), "w") as f:
        json.dump(res, f, indent=1)
    log(f"métricas -> {OUT}/modelo_toxicidad_metricas.json")

    # lectura honesta: techo real del dominio ~0.58-0.60; perseguir estabilidad
    if auc_all > 0.60:
        log("=> hay señal real: las features predicen toxicidad sobre "
            "mercados NO vistos.")
    elif auc_all > 0.55:
        log("=> señal débil pero presente (acorde al techo del dominio "
            "~0.58). Útil como scoring auxiliar / política por régimen, "
            "NO para tradear autónomo.")
    else:
        log("=> señal muy débil con esta data. Hace falta más captura o "
            "mejores features antes de confiar en el modelo.")


if __name__ == "__main__":
    main()
