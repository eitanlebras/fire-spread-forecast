"""Pick demo frames that show OUTWARD spread, not infill.
  python -m fsf.growth_sweep FIRE_DIR:PRED_DIR [FIRE_DIR:PRED_DIR ...] [--out demo/frames_growth] [--top 8] [--render 3] [--thr 0.4]

Per day t (today = masks[t], new = masks[t+1] & ~masks[t], prob = probs[t+1] i.e. the forecast made on t):
  n_new          new-fire pixels (need >= 50)
  outside_frac   share of new fire beyond the OUTER boundary of today's fire (binary_fill_holes(today)), not in interior pockets
  growth_auc_pr  AUC-PR of prob on ~today pixels (the honest number; goes in the header)
  growth_overlap share of new-fire pixels with prob > thr
  max_reach_km   distance from today's fire to the furthest new-fire pixel
  detached       new-fire components not touching today's fire (8-connectivity, no shared/adjacent pixel): count, and how
                 many of them have prob > thr within 2 px (the model "saw" the spot)
Rank by outside_frac * growth_auc_pr, n_new >= 50."""
import argparse, json, os, sys, numpy as np
from scipy import ndimage as ndi
from sklearn.metrics import average_precision_score


def day_metrics(prob, today, new, thr=0.4, pixel_km=0.375):
    n_new = int(new.sum()); region = ~today
    if n_new == 0: return None
    filled = ndi.binary_fill_holes(today) if today.any() else today
    outside = new & ~filled
    d = ndi.distance_transform_edt(~today) * pixel_km if today.any() else np.full(today.shape, np.inf)
    lab, n = ndi.label(new, structure=np.ones((3, 3)))
    touch = ndi.binary_dilation(today, structure=np.ones((3, 3)))
    det = [i for i in range(1, n + 1) if not (touch & (lab == i)).any()]
    zone = prob > thr; zone2 = ndi.binary_dilation(zone, structure=np.ones((5, 5)))          # within ~2 px of a >thr pixel
    det_hit = [i for i in det if (zone2 & (lab == i)).any()]
    return dict(n_new=n_new, outside_frac=float(outside.sum() / n_new), growth_auc_pr=float(average_precision_score(new[region], prob[region])),
                growth_overlap=float((prob[new] > thr).mean()), max_reach_km=float(d[new].max()) if np.isfinite(d[new]).any() else float("nan"),
                detached=len(det), detached_px=int(sum((lab == i).sum() for i in det)), detached_hits=len(det_hit))


def sweep(fire_dir, pred_dir, thr):
    probs = np.load(f"{pred_dir}/probs.npy"); pers = np.load(f"{pred_dir}/persistence.npy") > 0; dates = json.load(open(f"{pred_dir}/dates.json"))["dates"]
    fire = fire_dir.rstrip("/").split("/")[-1]; rows = []
    for t in range(len(probs)):
        m = day_metrics(probs[t], pers[t], pers[t + 1] & ~pers[t], thr)
        if m: rows.append(dict(fire=fire, day=t, today=dates[t], forecast_for=dates[t + 1], n_today=int(pers[t].sum()), **m, score=m["outside_frac"] * m["growth_auc_pr"]))
    return rows, probs, pers, dates


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pairs", nargs="+", help="FIRE_DIR:PRED_DIR"); ap.add_argument("--out", default="demo/frames_growth")
    ap.add_argument("--top", type=int, default=8); ap.add_argument("--render", type=int, default=3); ap.add_argument("--thr", type=float, default=0.4); ap.add_argument("--min-new", type=int, default=50)
    a = ap.parse_args(); allrows, data = [], {}
    for pr in a.pairs:
        fd, pd_ = pr.split(":"); rows, probs, pers, dates = sweep(fd, pd_, a.thr); allrows += rows; data[rows[0]["fire"]] = (fd, probs, pers, dates)
    ranked = sorted([r for r in allrows if r["n_new"] >= a.min_new], key=lambda r: -r["score"])[:a.top]
    cols = ["fire", "day", "today", "n_today", "n_new", "outside_frac", "growth_auc_pr", "growth_overlap", "max_reach_km", "detached", "detached_px", "detached_hits", "score"]
    print(f"top {len(ranked)} by outside_frac x growth_auc_pr, n_new >= {a.min_new}, zone = P > {a.thr:g}")
    print(" ".join(f"{c:>14}" for c in cols))
    for r in ranked: print(" ".join(f"{r[c]:>14.3f}" if isinstance(r[c], float) else f"{str(r[c]):>14}" for c in cols))
    if a.render:
        from viz.frames import FireScene, render_day
        os.makedirs(a.out, exist_ok=True); scenes = {}
        for rank, r in enumerate(ranked[:a.render], 1):
            fd, probs, pers, dates = data[r["fire"]]; scene = scenes.setdefault(r["fire"], FireScene(fd)); t = r["day"]
            today = pers[t]; new = pers[t + 1] & ~today
            m = today | new; rr, cc = np.where(m); h = int(max(40, (rr.max() - rr.min()) * 0.7, (cc.max() - cc.min()) * 0.7)); cy, cx = (rr.max() + rr.min()) // 2, (cc.max() + cc.min()) // 2
            crop = (max(0, cy - h), min(scene.H, cy + h), max(0, cx - h), min(scene.W, cx + h))
            title = (f"#{rank} {r['fire']} · day {t}: today {r['today']} ({r['n_today']} px) → forecast for {r['forecast_for']} · new fire {r['n_new']} px, {r['outside_frac']:.0%} outside today's boundary\n"
                     f"growth-region AUC-PR {r['growth_auc_pr']:.3f} · {r['growth_overlap']:.0%} of new fire in the >{a.thr:g} zone · max reach {r['max_reach_km']:.1f} km · detached spots {r['detached']} ({r['detached_hits']} with P>{a.thr:g} within 2 px)")
            out = os.path.join(a.out, f"rank{rank}_{r['fire']}_day{t:02d}_{r['today']}.png"); render_day(scene, probs[t], today, new, title, out, crop=crop); print("wrote", out)
        json.dump(ranked, open(os.path.join(a.out, "ranked.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
