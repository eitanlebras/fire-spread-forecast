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

def plot_auc_pr(df, split, metric, out_path):
    col = f"{split}.{metric}"
    if col not in df: sys.exit(f"column {col} not in runs (have: {[c for c in df if c.startswith(split)]})")
    arms = arm_order(df["name"]); s = summarize(df, col).loc[arms]
    pers_col = f"{col}_persistence"; pers = float(df[pers_col].mean()) if pers_col in df else None
    fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=150)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.82, bottom=0.20)
    x = np.arange(len(arms))
    ax.bar(x, s["mean"], width=0.46, color=P.BLUE, zorder=3)
    ax.errorbar(x, s["mean"], yerr=s["ci"], fmt="none", ecolor=P.INK_2, elinewidth=1.2, capsize=4, capthick=1.2, zorder=4)
    for xi, (m, c) in enumerate(zip(s["mean"], s["ci"])):
        ax.text(xi, m + c + 0.012, f"{m:.3f}", ha="center", va="bottom", fontsize=9.5, color=P.INK, fontweight="bold")
    refs = [(CONV_AE_AUC_PR, "Conv-AE baseline\n(Huot et al. 2022)", "--")]
    if pers is not None: refs.append((pers, "persistence,\nsame pixels", "-"))
    for y, label, ls in refs:                                    # reference lines, labelled in a right-hand margin so nothing overlaps a bar
        ax.axhline(y, color=P.MUTED, lw=1, ls=ls, zorder=2)
        ax.text(len(arms) - 0.3, y, f"{label}  {y:.3f}", ha="left", va="center", fontsize=8, color=P.INK_2, linespacing=1.15,
                bbox=dict(facecolor=P.SURFACE, edgecolor="none", pad=1.5))
    ax.set_xticks(x); ax.set_xticklabels([f"{ARM_LABELS.get(a, a)}\nn = {int(s.loc[a, 'n'])} seeds" for a in arms])
    ax.set_xlim(-0.6, len(arms) + 0.9)
    ax.set_ylim(0, max(1.0 if metric == "auc_pr" and s["mean"].max() > 0.8 else (s["mean"] + s["ci"]).max() * 1.35, CONV_AE_AUC_PR * 1.25))
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
    ax.set_ylabel("AUC-PR" if metric == "auc_pr" else metric)
    title = {"auc_pr": "Next-day fire spread, AUC-PR by arm", "auc_pr_growth_region": "AUC-PR on growth region only (pixels not burning at t)"}.get(metric, metric)
    fig.text(0.10, 0.93, title, fontsize=13, fontweight="bold", color=P.INK)
    fig.text(0.10, 0.875, f"{split} split · mean over seeds, error bars = 95% CI · fire-spread-forecast-v1-small", fontsize=9, color=P.INK_2)
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


def plot_calibration(pooled, eces, arms, split, out_path):
    arms = [a for a in arms if a in pooled]
    if not arms: print("no reliability rows found; skipping calibration diagram"); return
    fig, (ax, axn) = plt.subplots(2, 1, figsize=(6.2, 7.2), dpi=150, gridspec_kw={"height_ratios": [4, 1.1], "hspace": 0.08}, sharex=True)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.09)
    ax.plot([0, 1], [0, 1], color=P.AXIS, lw=1, ls="--", zorder=1)
    for arm in arms:
        r = np.asarray(pooled[arm]); c = arm_color(arm, arms)
        ax.plot(r[:, 1], r[:, 2], color=c, lw=2, marker="o", ms=5, mec=P.SURFACE, mew=1.2, zorder=3,
                label=f"{ARM_LABELS.get(arm, arm)}   ECE {eces.get(arm, float('nan')):.3f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_ylabel("observed fire fraction")
    ax.yaxis.grid(True, zorder=0); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
    ax.legend(loc="upper left", handlelength=1.6)
    w = 0.8 / len(arms)
    for i, arm in enumerate(arms):
        r = np.asarray(pooled[arm]); bw = np.diff(np.append(r[:, 0], 1.0)) if len(r) > 1 else np.array([1.0])
        axn.bar(r[:, 0] + bw * (0.1 + w * i), r[:, 3] / r[:, 3].sum(), width=bw * w * 0.85, align="edge", color=arm_color(arm, arms), zorder=3)
    axn.set_yscale("log"); axn.set_ylabel("share of pixels", fontsize=9); axn.set_xlabel("predicted probability (bin mean)")
    axn.yaxis.grid(True, zorder=0); axn.set_axisbelow(True); axn.spines["left"].set_visible(False); axn.tick_params(axis="y", labelsize=7.5)
    fig.text(0.13, 0.955, "Calibration: predicted probability vs. observed fire fraction", fontsize=13, fontweight="bold", color=P.INK)
    fig.text(0.13, 0.915, f"{split} split · bins pooled over seeds, weighted by pixel count · dashed = perfect calibration", fontsize=9, color=P.INK_2)
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
    plot_calibration(pooled, eces, arm_order(df["name"]), a.split, os.path.join(a.out, f"calibration{'_demo' if a.demo else ''}.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
