"""Result charts from results/runs.parquet: AUC-PR by arm (mean ± 95% CI over seeds) and a calibration reliability diagram.

  python -m viz.results                                   # reads results/runs.parquet + results/detail/*.json -> results/figs/
  python -m viz.results --runs R.parquet --detail DIR --out FIGS --split test --metric auc_pr
  python -m viz.results --demo                            # synthetic rows, to check the figures before any run exists

Bars: one per arm in the fixed report order (NDWS only -> + derived -> + OlmoEarth frozen -> + OlmoEarth post-trained),
unknown arm names appended. Reference lines: persistence on the same pixels (from the runs) and the published Conv-AE
baseline. Calibration: per-arm reliability curves pooled over seeds (bins weighted by pixel count) from the detail JSONs
that fsf.train writes next to the parquet; skipped with a message if they are absent. Missing parquet -> message, exit 0.
"""
import argparse, glob, json, os, sys, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from viz import palette as P

ARM_ORDER = ["ndws12", "ndws_derived", "olmo_frozen", "olmo_posttrained"]
ARM_LABELS = {"ndws12": "NDWS 12 channels", "ndws_derived": "+ derived features", "olmo_frozen": "+ OlmoEarth, frozen",
              "olmo_posttrained": "+ OlmoEarth, post-trained", "ndws12_raw_th": "NDWS 12, raw temp/hum"}
CONV_AE_AUC_PR = 0.284      # Huot et al. 2022, published NDWS baseline


def arm_order(names):
    seen = list(dict.fromkeys(names))
    return [a for a in ARM_ORDER if a in seen] + [a for a in seen if a not in ARM_ORDER]


def arm_color(arm, arms):
    return P.CATEGORICAL[arms.index(arm) % len(P.CATEGORICAL)]     # fixed slot per arm position, never re-cycled


def ci95(s):
    return 1.96 * s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0


def summarize(df, col):
    g = df.groupby("name")[col]
    return pd.DataFrame({"mean": g.mean(), "ci": g.apply(ci95), "n": g.size()})

# ---------------------------------------------------------------- AUC-PR bars

def arm_ramp(n):
    """Ordinal blue ramp, light -> dark across the arms in report order (never lighter than step 250 on the light surface)."""
    steps = P.SEQ_BLUE[3:]                                            # 250 .. 700
    return [steps[int(round(i * (len(steps) - 1) / max(n - 1, 1)))] for i in range(n)]


def plot_auc_pr(df, split, metric, out_path, y_max=0.35):
    col = f"{split}.{metric}"
    if col not in df: sys.exit(f"column {col} not in runs (have: {[c for c in df if c.startswith(split)]})")
    arms = arm_order(df["name"]); s = summarize(df, col).loc[arms]
    pers_col = f"{col}_persistence"; pers = float(df[pers_col].mean()) if pers_col in df else None
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.82, bottom=0.16)
    x = np.arange(len(arms))
    ax.bar(x, s["mean"], width=0.46, color=arm_ramp(len(arms)), zorder=3)
    ax.errorbar(x, s["mean"], yerr=s["ci"], fmt="none", ecolor=P.INK_2, elinewidth=1.2, capsize=4, capthick=1.2, zorder=4)
    refs = [(CONV_AE_AUC_PR, "Conv-AE baseline (Huot et al. 2022)", "--")]
    if pers is not None: refs.append((pers, "persistence, same pixels", "-"))
    x0, band = -0.55, 0.018                                      # band = height of one text line in data units at this figure size
    for y, label, ls in refs:                                    # reference lines, labelled inline just above the line at the left edge
        ax.axhline(y, color=P.MUTED, lw=1, ls=ls, zorder=2)
        ax.text(x0, y + 0.004, f"{label}  {y:.3f}", ha="left", va="bottom", fontsize=8, color=P.INK_2, zorder=6,
                bbox=dict(facecolor=P.SURFACE, edgecolor="none", pad=1.2, alpha=0.9))
    colors = arm_ramp(len(arms))
    for xi, (m, c) in enumerate(zip(s["mean"], s["ci"])):
        y_lab = m + c + 0.006
        clash = xi < 2 and any(y_lab < y + 0.004 + band and y_lab + band > y for y, _, _ in refs)   # a reference label sits where the tip label would go
        if clash:                                                # then label inside the bar, just under the error bar, ink chosen by bar luminance
            rgb = matplotlib.colors.to_rgb(colors[xi]); dark = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] < 0.55
            ax.text(xi, m - c - 0.008, f"{m:.3f}", ha="center", va="top", fontsize=9.5, color=P.SURFACE if dark else P.INK, fontweight="bold", zorder=5)
        else: ax.text(xi, y_lab, f"{m:.3f}", ha="center", va="bottom", fontsize=9.5, color=P.INK, fontweight="bold", zorder=5)
    ax.set_xticks(x); ax.set_xticklabels([ARM_LABELS.get(a, a) for a in arms])
    ax.set_xlim(x0 - 0.05, len(arms) - 0.4); ax.set_ylim(0, y_max)
    top = float((s["mean"] + s["ci"]).max())
    if top > y_max * 0.96: print(f"warning: a bar reaches {top:.3f}, at or above the y cap of {y_max}; pass y_max to plot_auc_pr to raise it")
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
    ax.set_ylabel("AUC-PR" if metric == "auc_pr" else metric)
    title = {"auc_pr": "Next-day fire spread, AUC-PR by arm", "auc_pr_growth_region": "AUC-PR on growth region only (pixels not burning at t)"}.get(metric, metric)
    n = s["n"].astype(int); seeds = f"{n.min()} seeds" if n.min() == n.max() else f"{n.min()}–{n.max()} seeds per arm"
    fig.text(0.10, 0.93, title, fontsize=14, fontfamily="serif", fontweight="bold", color=P.INK)
    fig.text(0.10, 0.875, f"{split} split · mean over {seeds}, error bars = 95% CI · fire-spread-forecast-v1-small", fontsize=9, color=P.INK_2)
    fig.savefig(out_path); plt.close(fig); print("wrote", out_path)
    return s

# ---------------------------------------------------------------- calibration

def pool_bins(rows_by_arm):
    """{arm: [(edge, conf, acc, n), ...] over seeds} -> per-bin count-weighted mean conf/acc, summed n."""
    pooled = {}
    for arm, r in rows_by_arm.items():
        r = np.asarray(r, np.float64); out = []
        for e in np.unique(r[:, 0]):
            m = r[:, 0] == e; n = r[m, 3]
            if n.sum() > 0: out.append((float(e), float(np.average(r[m, 1], weights=n)), float(np.average(r[m, 2], weights=n)), float(n.sum())))
        pooled[arm] = out
    return pooled


def load_reliability(detail_dir, split, run_ids=None):
    """Reads fsf.train's detail JSONs -> ({arm: pooled bins}, {arm: mean ECE}); only runs present in the parquet if run_ids given."""
    rows, eces = {}, {}
    for p in sorted(glob.glob(os.path.join(detail_dir, "*.json"))):
        try: d = json.load(open(p))
        except Exception as e: print(f"skip {p}: {e}"); continue
        if run_ids is not None and d.get("run_id") not in run_ids: continue
        rel = (d.get(split) or {}).get("reliability")
        if not rel: continue
        rows.setdefault(d["name"], []).extend(rel); eces.setdefault(d["name"], []).append(d[split]["ece"])
    return pool_bins(rows), {a: float(np.mean(v)) for a, v in eces.items()}


def plot_calibration(pooled, eces, arms, best, split, out_path):
    """Two curves: the baseline (first arm in report order) and `best`; every other arm's ECE goes in a text block.
    The reliability panel is drawn square so the diagonal is a true 45 degrees; the histogram pools all arms."""
    arms = [a for a in arms if a in pooled]
    if not arms: print("no reliability rows found; skipping calibration diagram"); return
    shown = [arms[0]] + ([best] if best in arms and best != arms[0] else [])
    others = [a for a in arms if a not in shown]
    fw, fh = 6.2, 8.3; l, w = 0.13, 0.84                             # axes laid out in inches so the top panel is exactly square
    ax_h = w * fw / fh; hist_h = 1.15 / fh; gap = 0.28 / fh; bottom = 0.62 / fh
    fig = plt.figure(figsize=(fw, fh), dpi=150)
    axn = fig.add_axes([l, bottom, w, hist_h]); ax = fig.add_axes([l, bottom + hist_h + gap, w, ax_h], sharex=axn)
    ax.plot([0, 1], [0, 1], color=P.AXIS, lw=1, ls="--", zorder=1)
    for arm, c in zip(shown, P.CATEGORICAL):
        r = np.asarray(pooled[arm])
        ax.plot(r[:, 1], r[:, 2], color=c, lw=2, marker="o", ms=5, mec=P.SURFACE, mew=1.2, zorder=3,
                label=f"{ARM_LABELS.get(arm, arm)}   ECE {eces.get(arm, float('nan')):.3f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal", adjustable="box"); ax.set_ylabel("observed fire fraction")
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True); ax.spines["left"].set_visible(False); ax.tick_params(labelbottom=False)
    ax.legend(loc="upper left", handlelength=1.6)
    if others:
        block = "ECE, other arms\n" + "\n".join(f"{ARM_LABELS.get(a, a)}   {eces.get(a, float('nan')):.3f}" for a in others)
        ax.text(0.97, 0.03, block, transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=P.INK_2, linespacing=1.5)
    allrows = np.asarray([r for a in arms for r in pooled[a]]); edges = np.unique(allrows[:, 0])
    n = np.array([allrows[allrows[:, 0] == e, 3].sum() for e in edges]); bw = np.diff(np.append(edges, 1.0))
    axn.bar(edges + bw * 0.06, n / n.sum(), width=bw * 0.88, align="edge", color=P.AXIS, zorder=3)
    axn.set_yscale("log"); axn.set_ylabel("share of pixels", fontsize=9); axn.set_xlabel("predicted probability (bin mean)")
    axn.yaxis.grid(True, zorder=0); axn.set_axisbelow(True); axn.spines["left"].set_visible(False); axn.tick_params(axis="y", labelsize=7.5)
    fig.text(l, 0.955, "Calibration: predicted vs. observed fire fraction", fontsize=14, fontfamily="serif", fontweight="bold", color=P.INK)
    fig.text(l, 0.905, f"{split} split · bins pooled over seeds, weighted by pixel count\ndashed = perfect calibration · histogram = all arms pooled",
             fontsize=9, color=P.INK_2, linespacing=1.5)
    fig.savefig(out_path); plt.close(fig); print("wrote", out_path)

# ---------------------------------------------------------------- demo data

def demo_data(seed=0):
    """Synthetic runs + reliability rows shaped like fsf.train's output, so the figures can be checked before training."""
    rng = np.random.default_rng(seed); rows, detail = [], []
    truth = {"ndws12": 0.27, "ndws_derived": 0.285, "olmo_frozen": 0.31, "olmo_posttrained": 0.315}
    edges = np.linspace(0, 1, 16)[:-1]
    for arm, mu in truth.items():
        for s in range(10):
            auc = mu + rng.normal(0, 0.012); rid = f"{arm}_s{s}_demo"
            rel = []
            for e in edges:
                conf = e + 0.033; n = int(2e6 * np.exp(-8 * e)) + 50
                acc = np.clip(conf ** (1.15 if "olmo" in arm else 1.4) + rng.normal(0, 0.02), 0, 1)
                rel.append((float(e), float(conf), float(acc), n))
            ece = float(sum(r[3] * abs(r[1] - r[2]) for r in rel) / sum(r[3] for r in rel))
            rows.append({"run_id": rid, "name": arm, "seed": s, "test.auc_pr": auc, "test.auc_pr_persistence": 0.115 + rng.normal(0, 0.002),
                         "test.auc_pr_growth_region": auc * 0.6, "test.auc_pr_growth_region_persistence": 0.0, "test.ece": ece})
            detail.append({"run_id": rid, "name": arm, "test": {"ece": ece, "reliability": rel}})
    return pd.DataFrame(rows), detail


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="results/runs.parquet"); ap.add_argument("--detail", help="detail JSON dir (default: <runs dir>/detail)")
    ap.add_argument("--out", default="results/figs"); ap.add_argument("--split", default="test", choices=["test", "eval"])
    ap.add_argument("--metric", default="auc_pr", help="auc_pr | auc_pr_growth_region | any <split>.<metric> column")
    ap.add_argument("--demo", action="store_true", help="render from synthetic data instead of the parquet")
    a = ap.parse_args(argv); P.style(); os.makedirs(a.out, exist_ok=True)
    if a.demo:
        df, detail = demo_data(); print("demo mode: synthetic data, not real results")
    elif not os.path.exists(a.runs):
        print(f"no results yet: {a.runs} does not exist. Run fsf.train first, or try --demo to preview the figures."); return 0
    else:
        df = pd.read_parquet(a.runs); detail = None
        print(f"{len(df)} runs, arms: " + ", ".join(f"{k}×{v}" for k, v in df['name'].value_counts().items()))
    s = plot_auc_pr(df, a.split, a.metric, os.path.join(a.out, f"{a.metric}_by_arm{'_demo' if a.demo else ''}.png"))
    print(s.to_string())
    if detail is not None:
        tmp, eces = {}, {}
        for d in detail:
            tmp.setdefault(d["name"], []).extend(d["test"]["reliability"]); eces.setdefault(d["name"], []).append(d["test"]["ece"])
        pooled, eces = pool_bins(tmp), {k: float(np.mean(v)) for k, v in eces.items()}
    else:
        ddir = a.detail or os.path.join(os.path.dirname(a.runs) or ".", "detail")
        if not os.path.isdir(ddir): print(f"no detail dir at {ddir}; skipping calibration diagram"); return 0
        pooled, eces = load_reliability(ddir, a.split, set(df["run_id"]) if "run_id" in df else None)
    plot_calibration(pooled, eces, arm_order(df["name"]), str(s["mean"].idxmax()), a.split, os.path.join(a.out, f"calibration{'_demo' if a.demo else ''}.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
