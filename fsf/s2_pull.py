"""Sentinel-2 L2A monthly median composites for WildfireSpreadTS fire events, on each event's own grid (OlmoEarth input).

  .venv/bin/python fsf/s2_pull.py [--root /root/data/wfts/tif] [--out /root/data/s2] [--workers 8]
                                  [--limit N] [--events 2018/fire_123 ...] [--overwrite]

For every /root/data/wfts/tif/{year}/fire_{id}/: bounds+CRS from any GeoTIFF (rioxarray), bbox -> EPSG:4326, fire start
= earliest date in the filenames. Earth Search (element84, sentinel-2-l2a) items with eo:cloud_cover < 40 over the 4
calendar months before the fire start, one timestep per month (oldest first); per timestep the per-pixel nanmedian of
all scenes after SCL cloud/shadow masking, warped straight onto the event grid (average resampling, 10-60 m -> 375 m).
A month whose composite covers < MIN_VALID of the grid is widened backwards month by month (up to WIDEN_MAX extra
months) before the timestep is given up (all-NaN) -- such events are listed in {out}/short_events.log.

Output {out}/{year}/fire_{id}.npy: float16 (4, 12, H, W), NaN = no data. Values are L2A reflectance * 10000 in the
pre-04.00 processing-baseline convention (the STAC raster:bands offset is applied, i.e. 1000 is subtracted for
reprocessed scenes), which is the scale of OlmoEarth's sentinel2_l2a normalization stats.
Band axis == olmoearth_pretrain Modality.SENTINEL2_L2A.band_order, checked against the repo:
  ['B02','B03','B04','B08','B05','B06','B07','B8A','B11','B12','B01','B09']
Sidecar {out}/{year}/fire_{id}.json: months, scene counts, widening, valid fraction per timestep.
"""
import argparse, glob, json, os, re, sys, time, traceback, warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import numpy as np

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-l2a"
CLOUD_LT = 40
N_MONTHS = 4
WIDEN_MAX = 3            # extra earlier months tried before a timestep is dropped
MIN_VALID = 0.5          # composite must cover this fraction of the grid to count as usable
MAX_SCENES_PER_TILE = 5  # least-cloudy scenes kept per (month, MGRS tile)
SCL_BAD = (0, 1, 3, 8, 9, 10)  # nodata, saturated, cloud shadow, cloud med/high prob, thin cirrus

# OlmoEarth sentinel2_l2a band order (olmoearth_pretrain/data/constants.py, Modality.SENTINEL2_L2A) -> Earth Search asset key
OLMO_S2_BANDS = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"]
ASSET_OF = {"B02": "blue", "B03": "green", "B04": "red", "B08": "nir", "B05": "rededge1", "B06": "rededge2",
            "B07": "rededge3", "B8A": "nir08", "B11": "swir16", "B12": "swir22", "B01": "coastal", "B09": "nir09"}
ASSETS = [ASSET_OF[b] for b in OLMO_S2_BANDS]

GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF",
                GDAL_HTTP_MULTIPLEX="YES", GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2", VSI_CACHE="TRUE",
                GDAL_CACHEMAX=256, CPL_VSIL_CURL_CACHE_SIZE="67108864")
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def shift_month(d, k):
    """d shifted by k calendar months, day clamped."""
    import calendar
    m = d.month - 1 + k; y = d.year + m // 12; m = m % 12 + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


class Grid:
    def __init__(self, crs, transform, height, width, bbox4326):
        self.crs, self.transform, self.H, self.W, self.bbox = crs, transform, height, width, bbox4326


def event_grid(fire_dir):
    """Grid (CRS, transform, shape) + EPSG:4326 bbox + fire start date from the event's GeoTIFFs."""
    import rioxarray
    from rasterio.warp import transform_bounds
    tifs = sorted(glob.glob(os.path.join(fire_dir, "*.tif")))
    if not tifs: raise RuntimeError("no GeoTIFFs")
    dates = sorted({m.group(1) for p in tifs for m in [DATE_RE.search(os.path.basename(p))] if m})
    if not dates: raise RuntimeError("no dates in filenames")
    da = rioxarray.open_rasterio(tifs[0])
    crs, bounds, transform = da.rio.crs, da.rio.bounds(), da.rio.transform()
    H, W = da.rio.height, da.rio.width
    da.close()
    bbox = transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21)
    return Grid(crs, transform, H, W, list(bbox)), date.fromisoformat(dates[0])


def stac_items(bbox, t0, t1):
    from pystac_client import Client
    cl = Client.open(STAC_URL)
    s = cl.search(collections=[COLLECTION], bbox=bbox, datetime=f"{t0.isoformat()}T00:00:00Z/{t1.isoformat()}T23:59:59Z",
                  query={"eo:cloud_cover": {"lt": CLOUD_LT}}, limit=100)
    return list(s.items())


def item_day(it):
    return it.datetime.date()


def tile_of(it):
    p = it.properties
    return p.get("grid:code") or p.get("s2:mgrs_tile") or p.get("mgrs:utm_zone", "") + str(p.get("mgrs:latitude_band", "")) + str(p.get("mgrs:grid_square", ""))


def read_warped(href, grid, resampling, dtype):
    import rasterio
    from rasterio.vrt import WarpedVRT
    with rasterio.open(href) as src:
        with WarpedVRT(src, crs=grid.crs, transform=grid.transform, width=grid.W, height=grid.H,
                       resampling=resampling, src_nodata=0, nodata=0) as vrt:
            return vrt.read(1).astype(dtype)


def read_scene(it, grid):
    """(12, H, W) float32 harmonized DN with NaN for nodata/cloud, or None if the scene can't be read."""
    from rasterio.enums import Resampling
    out = np.full((len(ASSETS), grid.H, grid.W), np.nan, np.float32)
    try:
        scl = read_warped(it.assets["scl"].href, grid, Resampling.mode, np.uint8)
    except Exception as e:
        warnings.warn(f"{it.id} scl: {e}"); return None
    bad = np.isin(scl, SCL_BAD)
    for i, key in enumerate(ASSETS):
        a = it.assets.get(key)
        if a is None: continue
        try:
            dn = read_warped(a.href, grid, Resampling.average, np.float32)
        except Exception as e:
            warnings.warn(f"{it.id} {key}: {e}"); continue
        rb = (a.extra_fields.get("raster:bands") or [{}])[0]
        off, sc = rb.get("offset", 0) or 0, rb.get("scale", 1e-4) or 1e-4
        valid = (dn > 0) & ~bad
        if off: dn = np.maximum(dn + off / sc, 0)   # e.g. offset -0.1 / scale 1e-4 -> DN - 1000 (baseline >= 04.00)
        out[i] = np.where(valid, dn, np.nan)
    return out


def composite(items, grid, pool):
    """nanmedian over scenes -> (12, H, W) float32, valid fraction."""
    if not items: return None, 0.0
    scenes = [s for s in pool.map(lambda it: read_scene(it, grid), items) if s is not None]
    if not scenes: return None, 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(np.stack(scenes), axis=0)
    return med, float(np.isfinite(med[0]).mean())


def select_scenes(items, t0, t1):
    """items with t0 <= day < t1, capped to the least-cloudy MAX_SCENES_PER_TILE per MGRS tile."""
    by_tile = {}
    for it in items:
        if t0 <= item_day(it) < t1: by_tile.setdefault(tile_of(it), []).append(it)
    keep = []
    for tile, its in by_tile.items():
        keep += sorted(its, key=lambda it: it.properties.get("eo:cloud_cover", 100))[:MAX_SCENES_PER_TILE]
    return keep


def process_event(year, fire, root, out_dir, threads=16, overwrite=False):
    """One event -> npy + json. Returns a summary dict."""
    import rasterio
    out_npy = os.path.join(out_dir, year, f"{fire}.npy"); out_json = out_npy[:-4] + ".json"
    if os.path.exists(out_npy) and not overwrite: return {"event": f"{year}/{fire}", "status": "exists"}
    t_start = time.time()
    grid, fire_start = event_grid(os.path.join(root, year, fire))
    span0 = shift_month(fire_start, -(N_MONTHS + WIDEN_MAX))
    items = stac_items(grid.bbox, span0, fire_start)
    arr = np.full((N_MONTHS, len(ASSETS), grid.H, grid.W), np.nan, np.float32)
    meta = {"event": f"{year}/{fire}", "fire_start": fire_start.isoformat(), "H": grid.H, "W": grid.W, "crs": str(grid.crs),
            "bbox4326": grid.bbox, "bands": OLMO_S2_BANDS, "n_items_in_span": len(items), "timesteps": []}
    with rasterio.Env(**GDAL_ENV), ThreadPoolExecutor(threads) as pool:
        for i in range(N_MONTHS):
            t1 = shift_month(fire_start, -(N_MONTHS - 1 - i)); t0 = shift_month(t1, -1)
            ts = {"month": t0.strftime("%Y-%m"), "window": [t0.isoformat(), t1.isoformat()], "widened": 0, "n_scenes": 0, "valid": 0.0}
            best, best_valid = None, 0.0
            for widen in range(WIDEN_MAX + 1):
                w0 = shift_month(t0, -widen)
                sel = select_scenes(items, w0, t1)
                if widen and len(sel) == ts["n_scenes"]: continue   # nothing new in the wider window
                med, valid = composite(sel, grid, pool)
                ts.update(widened=widen, window=[w0.isoformat(), t1.isoformat()], n_scenes=len(sel))
                if valid > best_valid: best, best_valid = med, valid
                if valid >= MIN_VALID: break
            ts["valid"] = round(best_valid, 4); ts["usable"] = best_valid >= MIN_VALID
            if best is not None: arr[i] = best
            meta["timesteps"].append(ts)
    meta["n_usable"] = sum(t["usable"] for t in meta["timesteps"]); meta["short"] = meta["n_usable"] < N_MONTHS
    meta["seconds"] = round(time.time() - t_start, 1)
    os.makedirs(os.path.dirname(out_npy), exist_ok=True)
    tmp = out_npy + ".partial.npy"; np.save(tmp, arr.astype(np.float16)); os.replace(tmp, out_npy)
    with open(out_json, "w") as f: json.dump(meta, f, indent=1)
    return {"event": meta["event"], "status": "short" if meta["short"] else "ok", "shape": list(arr.shape),
            "usable": [t["usable"] for t in meta["timesteps"]], "widened": [t["widened"] for t in meta["timesteps"]],
            "valid": [t["valid"] for t in meta["timesteps"]], "seconds": meta["seconds"]}


def _worker(args):
    year, fire, root, out_dir, threads, overwrite = args
    try:
        return process_event(year, fire, root, out_dir, threads, overwrite)
    except Exception as e:
        return {"event": f"{year}/{fire}", "status": "failed", "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()}


def list_events(root, only=None):
    ev = []
    for y in sorted(os.listdir(root)):
        if not y.isdigit(): continue
        for f in sorted(os.listdir(os.path.join(root, y))):
            if f.startswith("fire_") and not f.endswith(".partial") and glob.glob(os.path.join(root, y, f, "*.tif")):
                ev.append((y, f))
    if only: ev = [e for e in ev if f"{e[0]}/{e[1]}" in set(only)]
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/data/wfts/tif"); ap.add_argument("--out", default="/root/data/s2")
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--threads", type=int, default=16, help="asset reads per worker")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--events", nargs="*", help="year/fire_id ...")
    ap.add_argument("--overwrite", action="store_true"); ap.add_argument("--progress", type=int, default=20)
    a = ap.parse_args()
    events = list_events(a.root, a.events)
    if a.limit: events = events[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    print(f"{len(events)} events, {a.workers} workers x {a.threads} threads -> {a.out}", flush=True)
    jobs = [(y, f, a.root, a.out, a.threads, a.overwrite) for y, f in events]
    short_log = open(os.path.join(a.out, "short_events.log"), "a"); fail_log = open(os.path.join(a.out, "failed_events.log"), "a")
    n = {"ok": 0, "short": 0, "failed": 0, "exists": 0}; t0 = time.time()
    if a.workers <= 1:
        results = map(_worker, jobs)
    else:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(a.workers); results = pool.imap_unordered(_worker, jobs)
    for i, r in enumerate(results, 1):
        n[r["status"]] += 1
        if r["status"] == "short":
            short_log.write(f"{r['event']} usable={r['usable']} widened={r['widened']} valid={r['valid']}\n"); short_log.flush()
        elif r["status"] == "failed":
            fail_log.write(f"{r['event']} {r['error']}\n{r['tb']}\n"); fail_log.flush()
            print(f"FAILED {r['event']}: {r['error']}", flush=True)
        if len(events) <= 3 or i % a.progress == 0 or i == len(events):
            print(f"[{i}/{len(events)} {time.time() - t0:.0f}s] ok={n['ok']} short={n['short']} failed={n['failed']} exists={n['exists']}"
                  + (f"  last={r}" if len(events) <= 3 else ""), flush=True)
    print("S2_PULL_DONE", json.dumps(n), flush=True)


if __name__ == "__main__":
    main()
