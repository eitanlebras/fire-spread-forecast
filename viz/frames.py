"""Headless per-day frames of the demo map (matplotlib, no browser): one PNG per today t, labelled with the numbers.
  python -m viz.frames FIRE_DIR PRED_DIR OUT_DIR [--thr 0.4]
Background: hillshade from the elevation band + water from the land-cover band. Layers as in viz.demo_map: charcoal =
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


class FireScene:
    """Static background for one fire: hillshade, water, pixel size."""
    def __init__(self, fire_dir):
        tifs = sorted(glob.glob(os.path.join(fire_dir, "*.tif"))); self.fire = fire_dir.rstrip("/").split("/")[-1]
        with rasterio.open(tifs[0]) as ds:
            e = ds.read(ELEV + 1); self.elev = np.nan_to_num(e, nan=np.nanmean(e)); lc = ds.read(LANDCOVER + 1); self.pixel_km = abs(ds.transform.a) / 1000
        self.water = np.nan_to_num(lc, nan=0) == WATER
        self.shade = LightSource(azdeg=315, altdeg=45).hillshade(self.elev, vert_exag=0.3, dx=375, dy=375)
        self.H, self.W = self.elev.shape


def render_day(scene, prob, today, new, title, out_path, dpi=110, footer=None, crop=None):
    """One labelled PNG. crop = (r0, r1, c0, c1) rows/cols window or None for the full raster."""
    H, W, km = scene.H, scene.W, scene.pixel_km; shown, _, _ = growth_field(prob, today, km)
    r0, r1, c0, c1 = crop or (0, H, 0, W); ext = (c0 * km, c1 * km, r1 * km, r0 * km); sl = (slice(r0, r1), slice(c0, c1))
    fig, ax = plt.subplots(figsize=(9, 9 * (r1 - r0) / (c1 - c0) + 0.8), dpi=dpi)
    ax.imshow(scene.shade[sl], cmap="gray", vmin=0, vmax=1.4, extent=ext)
    ax.imshow(np.ma.masked_where(~scene.water[sl], scene.water[sl]), cmap=ListedColormap([P.WATER]), alpha=0.9, extent=ext, interpolation="nearest")
    ax.imshow(np.ma.masked_where(shown[sl] < 0.06, shown[sl]), cmap=P.PROB_CMAP, vmin=0, vmax=1, alpha=0.85, extent=ext, interpolation="bilinear")
    ax.imshow(np.ma.masked_where(~today[sl], today[sl]), cmap=ListedColormap([P.BURNING]), alpha=0.75, extent=ext, interpolation="nearest")
    if new[sl].any(): ax.contour(np.linspace(ext[0], ext[1], c1 - c0), np.linspace(ext[3], ext[2], r1 - r0), new[sl].astype(float), levels=[0.5], colors=[P.ORANGE], linewidths=1.8)
    ax.set_title(title, fontsize=10, loc="left"); ax.set_xlabel("km"); ax.set_ylabel("km")
    ax.text(0.01, 0.01, footer or "charcoal = burning today · blue = model P(burn in 24 h), growth region · orange = observed new fire next day · grey-green = water", transform=ax.transAxes, fontsize=7, color="#333", va="bottom")
    fig.tight_layout(); fig.savefig(out_path); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("fire_dir"); ap.add_argument("pred_dir"); ap.add_argument("out"); ap.add_argument("--thr", type=float, default=0.4); ap.add_argument("--dpi", type=int, default=110)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); scene = FireScene(a.fire_dir)
    probs = np.load(f"{a.pred_dir}/probs.npy"); pers = np.load(f"{a.pred_dir}/persistence.npy") > 0; dates = json.load(open(f"{a.pred_dir}/dates.json"))["dates"]
    rows = {r["day"]: r for r in sweep(a.pred_dir, a.thr)}
    for t in range(len(probs)):
        today = pers[t]; new = pers[t + 1] & ~today; r = rows[t]
        title = (f"{scene.fire} · day {t}: today {dates[t]} ({int(today.sum())} px) → forecast for {dates[t+1]} · new fire {r['n_new']} px\n"
                 f"growth-region AUC-PR {r['auc_pr']:.3f} · overlap in >{a.thr:g} zone {r['overlap']:.2f} · direction error {r['dir_err']:.0f}°")
        render_day(scene, probs[t], today, new, title, os.path.join(a.out, f"{scene.fire}_day{t:02d}_{dates[t]}.png"), a.dpi)
    print(f"wrote {len(probs)} frames to {a.out}")


if __name__ == "__main__":
    main()
