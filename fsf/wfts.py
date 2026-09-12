"""WildFireSpreadTS (Gerard et al. 2023) -> the same fp16 tensor cache format as the NDWS pipeline.
Reuses the authors' preprocessing (stats, degree features) from github.com/SebastianGer/WildfireSpreadTS.

  python -m fsf.wfts select  ZIP  N_PER_YEAR      -> prints/saves the sampled fire list (stratified by year)
  python -m fsf.wfts extract ZIP  FIRES_JSON OUT  -> unzip only those fires
  python -m fsf.wfts cache   ROOT OUT.pt          -> manifest + fp16 tiles

Channels stored (24): the 23 raw bands after the authors' preprocessing (active-fire time in hours, sin of degree
features, standardized with their 2018+2019 stats, NaN->0; landcover kept as its integer class in channel 16) plus
the binary active-fire mask. Landcover is one-hot expanded (17 classes) on the GPU at load time, giving the authors'
40-feature input. Target: next day's active fire (>0). Padded pixels get target -1 (excluded), like NDWS uncertain.
Split = the authors' fold 0: train 2018+2019, val 2020, test 2021.
"""
import sys, os, json, glob, zipfile, subprocess, random, numpy as np, torch
sys.path.insert(0, os.environ.get("WFTS_CODE", "/workspace/WildfireSpreadTS_code"))
from src.dataloader.utils import get_means_stds_missing_values, get_indices_of_degree_features  # noqa: E402

TILE = 128
LANDCOVER_IDX = 16
SPLIT = {"train": [2018, 2019], "eval": [2020], "test": [2021]}


def select(zip_path, n_per_year, seed=0):
    with zipfile.ZipFile(zip_path) as z: names = z.namelist()
    fires = {}
    for n in names:
        p = n.split("/")
        if len(p) >= 4 and p[-1].endswith(".tif"): fires.setdefault((p[-3], p[-2]), []).append(n)
    rng = random.Random(seed); chosen = []
    for year in sorted({y for y, _ in fires}):
        ids = sorted(f for y, f in fires if y == year and len(fires[(y, f)]) >= 3)
        chosen += [{"year": int(year), "fire": f, "n_days": len(fires[(year, f)]), "prefix": os.path.dirname(fires[(year, f)][0])}
                   for f in rng.sample(ids, min(n_per_year, len(ids)))]
    print(f"{len(fires)} fires in zip; selected {len(chosen)}: " + ", ".join(f"{y}:{sum(c['year']==y for c in chosen)}" for y in sorted({c['year'] for c in chosen})))
    return chosen


def extract(zip_path, fires, out_dir):
    pats = [f"{f['prefix']}/*" for f in fires]
    subprocess.run(["unzip", "-q", "-o", zip_path, *pats, "-d", out_dir], check=True)
    print("extracted", len(fires), "fires to", out_dir)


def preprocess(imgs, means, stds, deg_idx):
    """imgs (T,23,H,W) raw -> (T,24,H,W) float32, exactly the authors' preprocess_and_augment minus crop/one-hot."""
    x = imgs.astype(np.float32).copy()
    x[:, -1] = np.nan_to_num(x[:, -1], nan=0.0); x[:, -1] = np.floor_divide(x[:, -1], 100)   # hhmm -> hours
    x[:, deg_idx] = np.sin(np.deg2rad(x[:, deg_idx]))
    binary_af = (x[:, -1:] > 0).astype(np.float32)
    x = (x - means[None, :, None, None]) / stds[None, :, None, None]
    x = np.nan_to_num(x, nan=0.0)
    return np.concatenate([x, binary_af], 1)


def build_cache(root, out_path, tile=TILE):
    import rasterio
    from rasterio.warp import transform_bounds
    means, stds, _ = get_means_stds_missing_values((2018, 2019)); means, stds = means.numpy(), stds.numpy()
    deg_idx = get_indices_of_degree_features()
    fires_meta, X, Y, META = [], {s: [] for s in SPLIT}, {s: [] for s in SPLIT}, {s: [] for s in SPLIT}
    fire_dirs = sorted(glob.glob(f"{root}/*/*/"))
    for fi, fd in enumerate(fire_dirs):
        year, fire = int(fd.split("/")[-3]), fd.split("/")[-2]
        split = next(s for s, ys in SPLIT.items() if year in ys)
        paths = sorted(glob.glob(f"{fd}/*.tif")); dates = [os.path.basename(p).split("_")[0].replace(".tif", "") for p in paths]
        imgs = []
        with rasterio.open(paths[0]) as ds:
            crs, transform, H, W = ds.crs, ds.transform, ds.height, ds.width
            bbox = transform_bounds(crs, "EPSG:4326", *ds.bounds)
        for p in paths:
            with rasterio.open(p) as ds: imgs.append(ds.read())
        imgs = np.stack(imgs)                                             # (T,23,H,W)
        x = preprocess(imgs, means, stds, deg_idx)                        # (T,24,H,W)
        af_raw = np.nan_to_num(imgs[:, -1].astype(np.float32), nan=0.0); y_all = (af_raw > 0).astype(np.int8)
        fires_meta.append({"idx": fi, "year": year, "fire": fire, "split": split, "dates": dates, "H": int(H), "W": int(W),
                           "crs": str(crs), "transform": list(transform)[:6], "bbox_wgs84": [float(b) for b in bbox]})
        nr, nc = -(-H // tile), -(-W // tile)
        for t in range(len(paths) - 1):
            xt = np.zeros((24, nr * tile, nc * tile), np.float32); xt[:, :H, :W] = x[t]
            yt = np.full((nr * tile, nc * tile), -1, np.int8); yt[:H, :W] = y_all[t + 1]
            for r in range(nr):
                for c in range(nc):
                    xw = xt[:, r * tile:(r + 1) * tile, c * tile:(c + 1) * tile]; yw = yt[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile]
                    if split == "train" and not (xw[-1].any() or (yw == 1).any()): continue   # authors' crop preference: fire in input or target
                    X[split].append(torch.from_numpy(xw).half()); Y[split].append(torch.from_numpy(yw)); META[split].append((fi, t, r * tile, c * tile))
        print(f"[{fi+1}/{len(fire_dirs)}] {year}/{fire} days={len(paths)} {H}x{W} split={split} tiles_so_far={sum(len(v) for v in X.values())}", flush=True)
    blob = {"names": ["wfts_%02d" % i for i in range(23)] + ["binary_af"], "feature_set": "wfts", "landcover_idx": LANDCOVER_IDX, "fires": fires_meta,
            "vec_idx": [], "stats": {"means": means.tolist(), "stds": stds.tolist()}}
    for s in SPLIT:
        blob[f"x_{s}"] = torch.stack(X[s]); blob[f"y_{s}"] = torch.stack(Y[s]); blob[f"meta_{s}"] = torch.tensor(META[s], dtype=torch.int32)
        print(f"{s}: x={tuple(blob[f'x_{s}'].shape)} pos_rate={(blob[f'y_{s}']==1).float().mean():.5f}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True); torch.save(blob, out_path)
    json.dump(fires_meta, open(os.path.join(os.path.dirname(out_path), "wfts_fires.json"), "w"), indent=1)
    print("saved", out_path)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "select":
        chosen = select(sys.argv[2], int(sys.argv[3])); json.dump(chosen, open(sys.argv[4], "w"), indent=1)
    elif cmd == "extract": extract(sys.argv[2], json.load(open(sys.argv[3])), sys.argv[4])
    elif cmd == "cache": build_cache(sys.argv[2], sys.argv[3])
