"""Headless per-day frames of the demo map (matplotlib, no browser): one PNG per today t, labelled with the numbers.
  python -m viz.frames FIRE_DIR PRED_DIR OUT_DIR [--thr 0.4]
Background: hillshade from the elevation band + water from the land-cover band. Layers as in viz.demo_map: dark grey =
burning today, blue = the model's P(burn tomorrow) on the growth region within the feathered buffer, orange = new fire
observed tomorrow. Title: day, dates, new-fire px, AUC-PR (growth region), overlap of new fire with the > thr zone,
direction error (model P-mass vs observed new-fire bearing, within 6 km)."""
import argparse, glob, json, os, sys, numpy as np, rasterio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, ListedColormap
from viz import palette as P
from viz.demo_map import growth_field
from fsf.day_sweep import sweep

ELEV, LANDCOVER, WATER = 14, 16, 17


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("fire_dir"); ap.add_argument("pred_dir"); ap.add_argument("out"); ap.add_argument("--thr", type=float, default=0.4); ap.add_argument("--dpi", type=int, default=110)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    tifs = sorted(glob.glob(os.path.join(a.fire_dir, "*.tif"))); fire = a.fire_dir.rstrip("/").split("/")[-1]
    probs = np.load(f"{a.pred_dir}/probs.npy"); pers = np.load(f"{a.pred_dir}/persistence.npy") > 0; dates = json.load(open(f"{a.pred_dir}/dates.json"))["dates"]
    rows = {r["day"]: r for r in sweep(a.pred_dir, a.thr)}
    with rasterio.open(tifs[0]) as ds:
        elev = np.nan_to_num(ds.read(ELEV + 1), nan=np.nanmean(ds.read(ELEV + 1))); lc = ds.read(LANDCOVER + 1); pixel_km = abs(ds.transform.a) / 1000
    water = np.nan_to_num(lc, nan=0) == WATER
    shade = LightSource(azdeg=315, altdeg=45).hillshade(elev, vert_exag=0.3, dx=375, dy=375)
    H, W = elev.shape; km_w, km_h = W * pixel_km, H * pixel_km
    for t in range(len(probs)):
        today = pers[t]; new = pers[t + 1] & ~today; shown, _, _ = growth_field(probs[t], today, pixel_km); r = rows[t]
        fig, ax = plt.subplots(figsize=(9, 9 * H / W), dpi=a.dpi)
        ax.imshow(shade, cmap="gray", vmin=0, vmax=1.4, extent=(0, km_w, km_h, 0))
        ax.imshow(np.ma.masked_where(~water, water), cmap=ListedColormap(["#a9c8e8"]), alpha=0.9, extent=(0, km_w, km_h, 0), interpolation="nearest")
        pm = np.ma.masked_where(shown < 0.06, shown); ax.imshow(pm, cmap=P.PROB_CMAP, vmin=0, vmax=1, alpha=0.85, extent=(0, km_w, km_h, 0), interpolation="bilinear")
        ax.imshow(np.ma.masked_where(~today, today), cmap=ListedColormap(["#3a3a37"]), alpha=0.75, extent=(0, km_w, km_h, 0), interpolation="nearest")
        if new.any(): ax.contour(np.linspace(0, km_w, W), np.linspace(0, km_h, H), new.astype(float), levels=[0.5], colors=[P.ORANGE], linewidths=1.8)
        ax.set_title(f"{fire} · day {t}: today {dates[t]} ({int(today.sum())} px) → forecast for {dates[t+1]} · new fire {r['n_new']} px\n"
                     f"AUC-PR {r['auc_pr']:.3f} · overlap in >{a.thr:g} zone {r['overlap']:.2f} · direction error {r['dir_err']:.0f}°", fontsize=10, loc="left")
        ax.set_xlabel("km"); ax.set_ylabel("km"); ax.text(0.01, 0.01, "grey = burning today · blue = model P(burn in 24 h), growth region · orange = observed new fire next day · light blue = water",
                                                          transform=ax.transAxes, fontsize=7, color="#333", va="bottom")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, f"{fire}_day{t:02d}_{dates[t]}.png")); plt.close(fig)
    print(f"wrote {len(probs)} frames to {a.out}")


if __name__ == "__main__":
    main()
