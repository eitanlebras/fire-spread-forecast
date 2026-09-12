"""Metrics on valid pixels only (target != -1). Primary: AUC-PR (average precision) and ECE."""
import numpy as np, torch
from sklearn.metrics import average_precision_score, precision_recall_curve


def flat_valid(prob, y):
    m = y >= 0
    return prob[m].astype(np.float64), y[m].astype(np.int64)


def auc_pr(prob, y):
    p, t = flat_valid(prob, y)
    return float(average_precision_score(t, p)) if t.sum() > 0 else float("nan")


def ece(prob, y, bins=15):
    p, t = flat_valid(prob, y)
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e, rows = 0.0, []
    for b in range(bins):
        m = idx == b
        if m.any():
            conf, acc, n = p[m].mean(), t[m].mean(), m.mean()
            e += n * abs(conf - acc); rows.append((float(edges[b]), float(conf), float(acc), int(m.sum())))
    return float(e), rows


def best_f1(prob, y):
    p, t = flat_valid(prob, y)
    pr, rc, th = precision_recall_curve(t, p)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-9)
    i = int(np.nanargmax(f1[:-1]))
    return float(f1[i]), float(pr[i]), float(rc[i]), float(th[i])


def evaluate(prob, y, prev_fire):
    """prob, y, prev_fire: numpy (N,H,W). Returns dict incl. persistence baseline computed on the same pixels."""
    out = {"auc_pr": auc_pr(prob, y)}
    out["ece"], out["reliability"] = ece(prob, y)
    out["f1_best"], out["precision_at_best_f1"], out["recall_at_best_f1"], out["thr_at_best_f1"] = best_f1(prob, y)
    out["auc_pr_persistence"] = auc_pr(prev_fire, y)
    growth = prev_fire == 0                      # pixels not burning at t: can the model find NEW fire?
    yg = np.where(growth, y, -1)
    out["auc_pr_growth_region"] = auc_pr(prob, yg)
    out["auc_pr_growth_region_persistence"] = auc_pr(prev_fire, yg)
    out["pos_rate"] = float((y[y >= 0] == 1).mean())
    return out
