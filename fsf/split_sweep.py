"""Interior fill vs true advance. For each day: growth region = ~today, split into
  interior = unburned pixels INSIDE yesterday's drawn perimeter (smoothed mask contour, holes filled; what viz.replay outlines)
  advance  = pixels OUTSIDE that perimeter
and, for transparency, the strict split against binary_fill_holes(today) (raw speckled mask; encloses almost nothing).
Per subregion: new-fire px, AUC-PR of prob restricted to that subregion, overlap = share of its new fire with prob > thr.
Pooled rows concatenate all days' pixels. Usage: python -m fsf.split_sweep FIRE_DIR:PRED_DIR [...] [--thr 0.4] [--min-new 30]"""
import argparse, json, numpy as np
from scipy import ndimage as ndi
from sklearn.metrics import average_precision_score
from viz.replay import _blur


def regions(today):
    drawn = ndi.binary_fill_holes(_blur(today, 2.5) > 0.22) if today.any() else np.zeros_like(today)
    strict = ndi.binary_fill_holes(today) if today.any() else today
    return {"interior": drawn & ~today, "advance": ~drawn, "interior_strict": strict & ~today, "advance_strict": ~strict}


def score(prob, new, mask, thr):
    n = int((new & mask).sum())
    if n == 0: return n, float("nan"), float("nan")
    return n, float(average_precision_score(new[mask], prob[mask])), float((prob[new & mask] > thr).mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pairs", nargs="+"); ap.add_argument("--thr", type=float, default=0.4); ap.add_argument("--min-new", type=int, default=30); a = ap.parse_args()
    pooled = {k: {"p": [], "y": []} for k in ("interior", "advance", "interior_strict", "advance_strict")}; best = []
    for pr in a.pairs:
        fd, pd_ = pr.split(":"); fire = fd.rstrip("/").split("/")[-1]
        probs = np.load(f"{pd_}/probs.npy"); pers = np.load(f"{pd_}/persistence.npy") > 0; dates = json.load(open(f"{pd_}/dates.json"))["dates"]
        print(f"\n{fire}   (drawn perimeter = smoothed outline as in the replay; strict = fill_holes of the raw mask)")
        print(f"{'day':>3} {'today':>10} {'new':>4} | {'interior':>8} {'AUC-PR':>6} {'ovl':>5} | {'advance':>7} {'AUC-PR':>6} {'ovl':>5} || {'int_strict':>10} {'AUC-PR':>6} | {'adv_strict':>10} {'AUC-PR':>6}")
        for t in range(len(probs)):
            today = pers[t]; new = pers[t + 1] & ~today; p = probs[t]
            if new.sum() < 5: continue
            R = regions(today); r = {k: score(p, new, m, a.thr) for k, m in R.items()}
            for k, m in R.items(): pooled[k]["p"].append(p[m]); pooled[k]["y"].append(new[m])
            print(f"{t:>3} {dates[t]:>10} {int(new.sum()):>4} | {r['interior'][0]:>8} {r['interior'][1]:>6.3f} {r['interior'][2]:>5.2f} | {r['advance'][0]:>7} {r['advance'][1]:>6.3f} {r['advance'][2]:>5.2f} || {r['interior_strict'][0]:>10} {r['interior_strict'][1]:>6.3f} | {r['advance_strict'][0]:>10} {r['advance_strict'][1]:>6.3f}")
            if r["advance"][0] >= a.min_new: best.append((r["advance"][1], r["advance"][2], fire, t, dates[t], r["advance"][0], r["interior"][0], r["interior"][1]))
    print("\nPOOLED over all days of all fires (pixels concatenated):")
    for k, v in pooled.items():
        p = np.concatenate(v["p"]); y = np.concatenate(v["y"])
        print(f"  {k:>16}: new px {int(y.sum()):>6} of {y.size:>10}  AUC-PR {average_precision_score(y, p):.3f}  overlap {(p[y] > a.thr).mean():.2f}  prevalence {y.mean():.5f}")
    best.sort(reverse=True)
    print(f"\nBEST ADVANCE FRAMES (advance new px >= {a.min_new}, by advance AUC-PR):")
    for ap_, ov, fire, t, d, n_adv, n_int, ap_int in best[:8]: print(f"  {fire} day {t:>2} today {d}: advance AUC-PR {ap_:.3f} overlap {ov:.2f} (new px advance {n_adv}, interior {n_int}, interior AUC-PR {ap_int:.3f})")


if __name__ == "__main__":
    main()
