"""Sentinel-2 L2A monthly composites for each WildFireSpreadTS fire, on the fire's own grid, via Element84 Earth Search
STAC + public COGs on AWS (no auth). Output per fire: /root/data/s2_wfts/<year>_<fire>.pt with
  s2:   fp16 (T=4, 12, H, W)  bands in OlmoEarth order B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12,B01,B09 (raw L2A DN)
  mask: bool (T, H, W)        True where no clear observation was found
  months: [(year, month)] * 4 ending at the month of the fire's first day
Idempotent: skips fires whose output exists. Usage: python -m fsf.s2_pull TIF_ROOT OUT_DIR [workers]
"""
import sys, os, glob, json, time, datetime as dt, warnings
import numpy as np
from multiprocessing import Pool

STAC = "https://earth-search.aws.element84.com/v1"
ASSETS = ["blue", "green", "red", "nir", "rededge1", "rededge2", "rededge3", "nir08", "swir16", "swir22", "coastal", "nir09"]
CLOUD_SCL = {3, 8, 9, 10}     # cloud shadow, cloud medium, cloud high, thin cirrus
MAX_CC, SCENES_PER_MONTH, N_MONTHS = 40, 4, 4
GDAL_ENV = dict(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
                GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2", VSI_CACHE="TRUE", GDAL_HTTP_MULTIPLEX="YES")


def months_ending(date, n):
    y, m = date.year, date.month; out = []
    for _ in range(n):
        out.append((y, m)); m -= 1
        if m == 0: m, y = 12, y - 1
    return out[::-1]


def read_on_grid(href, crs, transform, H, W, resampling):
    import rasterio
    from rasterio.vrt import WarpedVRT
    with rasterio.open(href) as src:
        with WarpedVRT(src, crs=crs, transform=transform, width=W, height=H, resampling=resampling) as vrt:
            return vrt.read(1)


def pull_fire(fd):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds
    from pystac_client import Client
    os.environ.update(GDAL_ENV); warnings.filterwarnings("ignore")
    year, fire = fd.rstrip("/").split("/")[-2:]; out = f"{OUT}/{year}_{fire}.pt"
    if os.path.exists(out): return fire, "skip"
    tifs = sorted(glob.glob(f"{fd}/*.tif")); start = dt.date.fromisoformat(os.path.basename(tifs[0])[:10])
    with rasterio.open(tifs[0]) as ds:
        crs, transform, H, W = ds.crs, ds.transform, ds.height, ds.width; bbox = transform_bounds(crs, "EPSG:4326", *ds.bounds)
    client = Client.open(STAC); months = months_ending(start, N_MONTHS)
    s2 = np.zeros((N_MONTHS, 12, H, W), np.float16); nodata = np.ones((N_MONTHS, H, W), bool); n_scenes = []
    for ti, (y, m) in enumerate(months):
        m_end = (dt.date(y, m, 1) + dt.timedelta(days=32)).replace(day=1)
        end = min(m_end, start) if (y, m) == (start.year, start.month) else m_end     # last month: only pre-fire scenes
        rng = f"{y}-{m:02d}-01/{end.isoformat()}"
        items = list(client.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=rng, query={"eo:cloud_cover": {"lt": MAX_CC}}, max_items=40).items())
        if not items:   # fallback: whole month, any cloud cover
            items = list(client.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=f"{y}-{m:02d}-01/{m_end.isoformat()}", max_items=40).items())
        items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
        # group by date so one "scene" may be several tiles covering the bbox; take the SCENES_PER_MONTH clearest dates
        by_date = {}
        for it in items: by_date.setdefault(it.datetime.date(), []).append(it)
        dates = sorted(by_date, key=lambda d: np.mean([it.properties.get("eo:cloud_cover", 100) for it in by_date[d]]))[:SCENES_PER_MONTH]
        stack, valid = [], []
        for d in dates:
            acc = np.zeros((12, H, W), np.float32); cnt = np.zeros((H, W), np.float32)
            for it in by_date[d]:
                try:
                    scl = read_on_grid(it.assets["scl"].href, crs, transform, H, W, Resampling.nearest)
                    ok = (scl > 0) & ~np.isin(scl, list(CLOUD_SCL))
                    if not ok.any(): continue
                    bands = np.stack([read_on_grid(it.assets[a].href, crs, transform, H, W, Resampling.average) for a in ASSETS]).astype(np.float32)
                    okb = ok & (bands[0] > 0)
                    acc[:, okb] += bands[:, okb]; cnt[okb] += 1
                except Exception as e:
                    print(f"    {fire} {it.id}: {str(e)[:80]}", flush=True)
            if cnt.any(): stack.append(np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)); valid.append(cnt > 0)
        n_scenes.append(len(stack))
        if stack:
            comp = np.nanmedian(np.stack(stack), 0)                        # per-pixel median over clear scenes
            good = np.any(np.stack(valid), 0)
            s2[ti] = np.nan_to_num(comp, nan=0.0).astype(np.float16); nodata[ti] = ~good
    import torch
    torch.save({"s2": torch.from_numpy(s2), "nodata": torch.from_numpy(nodata), "months": months, "bands": ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12", "B01", "B09"],
                "crs": str(crs), "transform": list(transform)[:6], "H": H, "W": W, "fire": fire, "year": int(year), "n_scenes": n_scenes}, out + ".tmp")
    os.replace(out + ".tmp", out)
    return fire, f"ok scenes/month={n_scenes} nodata={nodata.mean():.2f}"


if __name__ == "__main__":
    root, OUT = sys.argv[1], sys.argv[2]; workers = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    os.makedirs(OUT, exist_ok=True)
    fires = [d for d in sorted(glob.glob(f"{root}/*/fire_*/")) if not d.rstrip("/").endswith(".partial")]
    todo = [d for d in fires if not os.path.exists(f"{OUT}/{d.rstrip('/').split('/')[-2]}_{d.rstrip('/').split('/')[-1]}.pt")]
    print(f"{len(fires)} fires on disk, {len(todo)} to pull, {workers} workers", flush=True); t0 = time.time(); n = 0
    with Pool(workers) as pool:
        for fire, msg in pool.imap_unordered(pull_fire, todo):
            n += 1
            if n % 10 == 0 or msg.startswith("skip") is False and n <= 3: print(f"[{n}/{len(todo)}] {fire}: {msg} ({time.time()-t0:.0f}s)", flush=True)
    print(f"S2_DONE {n} fires in {time.time()-t0:.0f}s", flush=True)
