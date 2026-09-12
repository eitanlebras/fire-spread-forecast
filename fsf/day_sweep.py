"""Per-day forecast quality for one fire: which day makes an honest demo?
  python -m fsf.day_sweep /root/data/preds/2021/fire_X [--thr 0.4]
Uses probs.npy / persistence.npy written by fsf.predict. For each today t (0..T-2): AUC-PR of the forecast for t+1
against the new fire on t+1 (pixels not burning on t), the new-fire pixel count, and the share of that new fire that
fell inside the > thr zone. Also the persistence-free 'growth' AP and the model's direction vs the observed one."""
import sys, json, argparse, numpy as np
from sklearn.metrics import average_precision_score
from scipy import ndimage as ndi


def bearing(dy, dx): return (np.degrees(np.arctan2(dx, -dy))) % 360


def perimeter_region(prev):
    """Interior of yesterday's drawn perimeter: the same smoothed-mask contour viz.replay draws (blur sigma 2.5, level 0.22),
    holes filled. New fire inside it is infill; outside it is directional spread."""
    from viz.replay import _blur
    return ndi.binary_fill_holes(_blur(prev, 2.5) > 0.22) if prev.any() else np.zeros_like(prev)


def sweep(pred_dir, thr=0.4, pixel_km=0.375):
    probs = np.load(f"{pred_dir}/probs.npy"); pers = np.load(f"{pred_dir}/persistence.npy") > 0; dates = json.load(open(f"{pred_dir}/dates.json"))["dates"]
    yy, xx = np.mgrid[:probs.shape[1], :probs.shape[2]]; rows = []
    for t in range(len(probs)):
        p = probs[t]; today = pers[t]; new = pers[t + 1] & ~today; region = ~today
        n_new = int(new.sum()); ap = float(average_precision_score(new[region], p[region])) if n_new else float("nan")
        outside = float((new & ~perimeter_region(today)).sum() / n_new) if n_new else float("nan")
        overlap = float((p[new] > thr).mean()) if n_new else float("nan")
        zone = int(((p > thr) & region).sum()); precision = float(new[(p > thr) & region].mean()) if zone else float("nan")
        d = ndi.distance_transform_edt(~today) * pixel_km if today.any() else None
        if today.any() and n_new:
            w = p * region * (d < 6); fy, fx = yy[today].mean(), xx[today].mean()
            pb = bearing((w * yy).sum() / w.sum() - fy, (w * xx).sum() / w.sum() - fx); nb = bearing(yy[new].mean() - fy, xx[new].mean() - fx)
            ddir = abs((pb - nb + 180) % 360 - 180)
        else: ddir = float("nan")
        rows.append(dict(day=t, today=dates[t], forecast_for=dates[t + 1], n_today=int(today.sum()), n_new=n_new, auc_pr=ap, overlap=overlap, zone_px=zone, zone_precision=precision, dir_err=ddir, outside=outside))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("pred_dir"); ap.add_argument("--thr", type=float, default=0.4); a = ap.parse_args()
    rows = sweep(a.pred_dir.rstrip("/"), a.thr)
    print(f"{a.pred_dir}  (zone = P > {a.thr:g} on pixels not burning today)")
    print(f"{'day':>3} {'today':>10} {'->for':>10} {'burn_today':>10} {'new_px':>6} {'AUC-PR':>7} {'overlap':>7} {'zone_px':>7} {'zone_prec':>9} {'dir_err':>7} {'outside':>7}  (outside = share of new fire beyond yesterday's perimeter; * = >=50%, directional not infill)")
    for r in rows:
        flag = "*" if r["outside"] == r["outside"] and r["outside"] >= 0.5 and r["n_new"] >= 30 else " "
        print(f"{r['day']:>3} {r['today']:>10} {r['forecast_for']:>10} {r['n_today']:>10} {r['n_new']:>6} {r['auc_pr']:>7.3f} {r['overlap']:>7.2f} {r['zone_px']:>7} {r['zone_precision']:>9.2f} {r['dir_err']:>7.0f} {r['outside']:>7.2f} {flag}")
    good = [r for r in rows if r["n_new"] >= 30]
    if good:
        best = max(good, key=lambda r: r["overlap"] * 0.5 + (r["auc_pr"] if r["auc_pr"] == r["auc_pr"] else 0) * 0.5)
        print(f"BEST (n_new>=30, by overlap+AUC-PR): day {best['day']} today={best['today']} new_px={best['n_new']} auc_pr={best['auc_pr']:.3f} overlap={best['overlap']:.2f} outside={best['outside']:.2f}")
        d = [r for r in good if r["outside"] >= 0.5]
        if d:
            bd = max(d, key=lambda r: r["overlap"] * 0.5 + r["auc_pr"] * 0.5)
            print(f"BEST DIRECTIONAL (n_new>=30, outside>=50%): day {bd['day']} today={bd['today']} new_px={bd['n_new']} auc_pr={bd['auc_pr']:.3f} overlap={bd['overlap']:.2f} outside={bd['outside']:.2f} dir_err={bd['dir_err']:.0f}")
