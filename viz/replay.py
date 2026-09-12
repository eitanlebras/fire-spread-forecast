"""Fire replay: forecast (left) vs. actual burn (right), scrubbed day by day, as a GIF plus one PNG per day.

  python -m viz.replay FIRE_DIR                       # WFTS fire dir of daily GeoTIFFs; placeholder probabilities
  python -m viz.replay FIRE_DIR --probs P.npy         # real model output
  python -m viz.replay masks.npz --probs P.npy        # masks from .npy/.npz instead of GeoTIFFs (keys: masks, dates)
  options: --out DIR (default out/replay/<fire>)  --fps 2  --days 3-12  --dpi 110  --frames-only

Day convention. masks[t] is the active-fire mask on day t (t = 0..T-1). The model sees day t and forecasts day t+1,
so frame k (k = 1..T-1) shows the forecast FOR day k on the left and the observed mask OF day k on the right, with
day k-1's fire perimeter outlined on both for reference. Probability arrays are accepted in either layout:
  (T-1, H, W): probs[i] is the forecast for day i+1 (made from day i)        <- what the model produces
  (T,   H, W): probs[t] is the forecast for day t; probs[0] is ignored
Without --probs a clearly-labelled placeholder is used: persistence plus a blurred halo around yesterday's fire.

WFTS GeoTIFFs (Gerard et al. 2023): <root>/<year>/<fire_id>/<YYYY-MM-DD>.tif, 23 bands, last band = active-fire
detection time (hhmm), NaN where nothing burned. Mask = band 23 > 0, exactly as fsf.wfts does it.
"""
import argparse, glob, os, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from PIL import Image
from viz import palette as P

# ---------------------------------------------------------------- data

def load_wfts_fire(fire_dir):
    """-> masks (T,H,W) bool, dates [str], name. Reads only the active-fire band of each daily GeoTIFF."""
    import rasterio
    paths = sorted(glob.glob(os.path.join(fire_dir, "*.tif")))
    if not paths: sys.exit(f"no .tif files in {fire_dir}")
    masks, dates = [], []
    for p in paths:
        with rasterio.open(p) as ds:
            af = ds.read(ds.count).astype(np.float32)             # last band: active fire hhmm, NaN = none
        masks.append(np.nan_to_num(af, nan=0.0) > 0)
        dates.append(os.path.basename(p).split("_")[0].replace(".tif", ""))
    name = "/".join(os.path.normpath(fire_dir).split(os.sep)[-2:])   # <year>/<fire_id>
    with rasterio.open(paths[0]) as ds: WATER_MASK[name] = np.nan_to_num(ds.read(17), nan=0) == 17   # land-cover band, class 17 = water
    return np.stack(masks), dates, name


WATER_MASK = {}   # name -> (H,W) bool, filled by load_wfts_fire


def load_masks_file(path):
    """masks from .npy (T,H,W) or .npz with 'masks' and optional 'dates'."""
    if path.endswith(".npz"):
        z = np.load(path, allow_pickle=True); masks = z["masks"]
        dates = [str(d) for d in z["dates"]] if "dates" in z else None
    else:
        masks, dates = np.load(path), None
    masks = np.asarray(masks) > 0
    dates = dates or [f"day {t}" for t in range(len(masks))]
    return masks, dates, os.path.splitext(os.path.basename(path))[0]


def load_fire(src):
    if os.path.isdir(src): return load_wfts_fire(src)
    if src.endswith((".npy", ".npz")): return load_masks_file(src)
    sys.exit(f"{src}: expected a directory of GeoTIFFs or a .npy/.npz masks file")


def _blur(a, sigma):
    """Separable Gaussian blur in numpy (no scipy dependency), reflect-padded."""
    r = int(3 * sigma); k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2); k /= k.sum()
    a = np.pad(a.astype(np.float32), r, mode="reflect")
    a = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, a)
    a = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 1, a)
    return a


def placeholder_probs(masks, sigma=3.0):
    """Stand-in until the model exists: P(day t+1) = 0.85 where day t burns, plus a halo decaying away from it.
    Deterministic and mask-only, so it is a persistence-flavoured baseline, not a forecast."""
    out = []
    for m in masks[:-1]:
        halo = _blur(m, sigma); halo = halo / max(halo.max(), 1e-6) * 0.6
        out.append(np.clip(np.where(m, 0.85, halo), 0, 1).astype(np.float32))
    return np.stack(out)


def align_probs(probs, T, shape):
    probs = np.asarray(probs, np.float32)
    if probs.ndim != 3 or probs.shape[1:] != shape:
        sys.exit(f"probs shape {probs.shape} does not match masks (T={T}, H×W={shape}); need (T-1,H,W) or (T,H,W)")
    if probs.shape[0] == T - 1: return probs
    if probs.shape[0] == T: return probs[1:]
    sys.exit(f"probs has {probs.shape[0]} days; masks have T={T}. Need T-1 (forecast for days 1..T-1) or T (index 0 ignored)")


def day_auc_pr(prob, y, prev):
    """(model AUC-PR full mask, persistence AUC-PR full mask, model AUC-PR on the growth region) for one day, or None.
    Growth region = pixels not burning yesterday; its positives are today's NEW fire. Persistence scores 0 there by
    construction, so the growth number is what the model adds beyond "it keeps burning"."""
    try: from sklearn.metrics import average_precision_score
    except ImportError: return None
    yf = y.ravel().astype(int)
    if yf.sum() == 0 or yf.sum() == yf.size: return None
    region = ~prev; new = (y & region)
    growth = float(average_precision_score(new[region].astype(int), prob[region])) if new.any() else float("nan")
    growth_pers = float(new[region].mean()) if new.any() else float("nan")   # a constant predictor's AP = prevalence: persistence's score there
    return float(average_precision_score(yf, prob.ravel())), float(average_precision_score(yf, prev.ravel().astype(float))), growth, growth_pers

# ---------------------------------------------------------------- rendering

def crop_box(prev, cur, H, W, margin=0.2, min_half=40):
    """Rows/cols window around today's fire + tomorrow's new fire with `margin` extra on each side (at least min_half px)."""
    m = prev | cur
    if not m.any(): return 0, H, 0, W
    r, c = np.where(m); r0, r1, c0, c1 = r.min(), r.max(), c.min(), c.max()
    hh, hw = max(min_half, (r1 - r0) * (0.5 + margin)), max(min_half, (c1 - c0) * (0.5 + margin)); half = max(hh, hw)   # square window
    cy, cx = (r0 + r1) / 2, (c0 + c1) / 2
    return int(max(0, cy - half)), int(min(H, cy + half)), int(max(0, cx - half)), int(min(W, cx + half))


def render_frame(k, masks, probs, dates, name, placeholder, dpi=110, hold_text=None):
    """One frame for day k (1..T-1): forecast for day k | observed day k. Returns an RGB uint8 array.
    Both panels are cropped to yesterday's fire + today's new fire (+20 %); water (land-cover class 17) is drawn light blue."""
    T, H, W = masks.shape
    prev, cur, prob = masks[k - 1], masks[k], probs[k - 1]
    r0, r1, c0, c1 = crop_box(prev, cur, H, W); water = WATER_MASK.get(name)
    burn = np.where(cur & prev, 1, np.where(cur, 2, 0)).astype(np.int8)
    fig = plt.figure(figsize=(11, 6.2), dpi=dpi)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.06], left=0.03, right=0.97, top=0.80, bottom=0.12, wspace=0.10, hspace=0.05)
    axL, axR = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[1, 0])
    for ax in (axL, axR):
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(True); s.set_edgecolor(P.GRID)
    # Growth region only: the model gets yesterday's fire as an input channel, so P inside it is the input echoed back,
    # not a prediction. Blank it; the outline below still shows where yesterday's fire was.
    im = axL.imshow(np.ma.masked_where(prev, prob), cmap=P.PROB_CMAP, vmin=0, vmax=1, interpolation="nearest")
    axR.imshow(burn, cmap=P.BURN_CMAP, vmin=0, vmax=2, interpolation="nearest")
    if water is not None:
        from matplotlib.colors import ListedColormap
        for ax in (axL, axR): ax.imshow(np.ma.masked_where(~water, water), cmap=ListedColormap(["#a9c8e8"]), alpha=0.9, interpolation="nearest")
    for ax in (axL, axR): ax.set_xlim(c0 - 0.5, c1 - 0.5); ax.set_ylim(r1 - 0.5, r0 - 0.5)
    if prev.any():                                   # yesterday's perimeter: outline of the lightly-smoothed mask, so speckled detections read as one front
        outline = _blur(prev, 2.5)
        for ax in (axL, axR): ax.contour(outline, levels=[0.22], colors=[P.INK_2], linewidths=0.9, alpha=0.85)
    axL.set_title("Forecast  ·  P(new fire on this day), growth region only, made the day before" + ("   [PLACEHOLDER]" if placeholder else ""))
    axR.set_title("Actual  ·  active fire on this day")
    cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.outline.set_visible(False)
    cb.set_ticks([0, 0.25, 0.5, 0.75, 1]); cax.tick_params(labelsize=8, length=0, colors=P.MUTED)
    handles = [Patch(facecolor=c, edgecolor=P.GRID, label=l) for c, l in zip(P.BURN_COLORS[1:], P.BURN_LABELS[1:])]
    handles.append(Line2D([0], [0], color=P.INK_2, lw=0.9, label="yesterday's perimeter (both panels)"))
    axR.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, -0.02), ncol=3, handlelength=1.0, columnspacing=0.8, handletextpad=0.5, fontsize=7.8, frameon=False)
    fig.text(0.03, 0.94, f"{name}   ·   day {k}/{T - 1}   ·   {dates[k]}", fontsize=13, fontweight="bold", color=P.INK)
    stats = f"burning: {int(cur.sum()):,} px   new today: {int((burn == 2).sum()):,} px   forecast made {dates[k - 1]}"
    m = day_auc_pr(prob, cur, prev)
    if m and m[2] == m[2]: stats += f"   ·   growth-region AUC-PR {m[2]:.3f} (persistence {m[3]:.3f})"
    fig.text(0.03, 0.895, stats, fontsize=9.5, color=P.INK_2)
    if hold_text: fig.text(0.03, 0.855, hold_text, fontsize=11, fontweight="bold", color=P.ORANGE)
    fig.text(0.97, 0.895, f"window {(c1 - c0) * 0.375:.0f} × {(r1 - r0) * 0.375:.0f} km", fontsize=8.5, color=P.MUTED, ha="right")
    fig.text(0.97, 0.03, "fire-spread-forecast-v1-small", fontsize=8, color=P.MUTED, ha="right")
    if placeholder: fig.text(0.03, 0.03, "placeholder probabilities: persistence + blurred halo around yesterday's fire, not a model", fontsize=8, color=P.MUTED)
    fig.canvas.draw(); rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(); plt.close(fig)
    return rgb


def render(masks, probs, dates, name, out_dir, placeholder=True, fps=2.0, days=None, dpi=110, gif=True, hold=None, hold_text=None, hold_s=6.0):
    """Writes <out_dir>/frames/day_XX_<date>.png for each day and <out_dir>/replay.gif. Returns list of frame paths.
    hold: replay day k to pause on for hold_s seconds, with hold_text in its header."""
    P.style(); T = len(masks); days = list(days or range(1, T))
    if WATER_MASK.get(name) is not None:
        wp = float(np.abs(probs[:, WATER_MASK[name]]).max()) if WATER_MASK[name].any() else 0.0
        print(f"  water mask: {int(WATER_MASK[name].sum())} px; max P over water = {wp:.4f} ({'OK, masked' if wp == 0 else 'NOT masked'})", flush=True)
    os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True); frames, paths = [], []
    for k in days:
        rgb = render_frame(k, masks, probs, dates, name, placeholder, dpi, hold_text if k == hold else None)
        p = os.path.join(out_dir, "frames", f"day_{k:02d}_{dates[k]}.png"); Image.fromarray(rgb).save(p)
        frames.append(Image.fromarray(rgb)); paths.append(p)
        print(f"  frame {k}/{T - 1} {dates[k]}  burning={int(masks[k].sum())}", flush=True)
    if gif and frames:
        dur = [int(1000 / fps)] * len(frames); dur[-1] *= 3                    # hold the last day
        if hold in days: dur[days.index(hold)] = int(hold_s * 1000)             # and the chosen day
        pal = [f.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
        gp = os.path.join(out_dir, "replay.gif")
        pal[0].save(gp, save_all=True, append_images=pal[1:], duration=dur, loop=0, optimize=False)
        print(f"wrote {gp} ({len(frames)} frames, {os.path.getsize(gp) / 1e6:.1f} MB)")
    return paths


def parse_days(s, T):
    if not s: return None
    a, _, b = s.partition("-"); a, b = int(a), int(b or a)
    return [k for k in range(a, b + 1) if 1 <= k < T]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fire", help="WFTS fire directory of daily GeoTIFFs, or a .npy/.npz masks file")
    ap.add_argument("--probs", help=".npy of probabilities, (T-1,H,W) or (T,H,W); omit for the placeholder")
    ap.add_argument("--out", help="output dir (default out/replay/<fire>)")
    ap.add_argument("--fps", type=float, default=2.0); ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--days", help="day range to render, e.g. 3-12 (1-based; day 0 has no forecast)")
    ap.add_argument("--frames-only", action="store_true", help="skip the GIF")
    ap.add_argument("--hold", type=int, help="replay day k (forecast FOR day k) to pause on"); ap.add_argument("--hold-text", default=None); ap.add_argument("--hold-s", type=float, default=6.0)
    a = ap.parse_args(argv)
    masks, dates, name = load_fire(a.fire); T = len(masks)
    if T < 2: sys.exit(f"{name}: need at least 2 days, found {T}")
    print(f"{name}: {T} days, {masks.shape[1]}x{masks.shape[2]} px, {dates[0]} .. {dates[-1]}")
    if a.probs: probs, placeholder = align_probs(np.load(a.probs), T, masks.shape[1:]), False
    else: probs, placeholder = placeholder_probs(masks), True
    out = a.out or os.path.join("out", "replay", name.replace("/", "_"))
    render(masks, probs, dates, name, out, placeholder, a.fps, parse_days(a.days, T), a.dpi, gif=not a.frames_only, hold=a.hold, hold_text=a.hold_text, hold_s=a.hold_s)


if __name__ == "__main__":
    main()
