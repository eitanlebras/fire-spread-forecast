"""NDWS loading -> channel tensors. Config-driven feature sets. Everything lives on the GPU (dataset is ~3 GB)."""
import os, zipfile, numpy as np, torch, xarray as xr
from . import features as F

RAW = ["elevation", "th", "vs", "tmmn", "tmmx", "sph", "pr", "pdsi", "NDVI", "population", "erc"]
# Feature sets. Vector pairs must be listed in VECTOR_PAIRS so augmentation transforms them correctly.
FEATURE_SETS = {
    "ndws12": ["elevation", "wind_u", "wind_v", "vs", "tmmn", "tmmx", "sph", "pr", "pdsi", "NDVI", "population", "erc",
               "prev_fire", "prev_uncertain"],
    "ndws12_raw_th": RAW + ["prev_fire", "prev_uncertain"],  # th as raw degrees, for the ablation
    "derived": ["elevation", "wind_u", "wind_v", "vs", "tmmn", "tmmx", "sph", "pr", "pdsi", "NDVI", "population", "erc",
                "prev_fire", "prev_uncertain", "slope", "aspect_x", "aspect_y", "wind_slope_align", "fm1", "fm10", "rh"],
}
VECTOR_PAIRS = [("wind_u", "wind_v"), ("aspect_x", "aspect_y")]
NO_NORM = {"prev_fire", "prev_uncertain"}


def open_split(root, split):
    d = f"{root}/{split}.zarr"
    if not os.path.exists(d):
        with zipfile.ZipFile(f"{root}/{split}.zarr.zip") as z: z.extractall(d)
    return xr.open_zarr(f"{d}/{split}.zarr")


def build_channels(ds):
    """Return dict name -> float32 (N,H,W) of every channel we know how to make, plus target."""
    c = {k: ds[k].values.astype(np.float32) for k in RAW}
    pm, fm = ds["PrevFireMask"].values, ds["FireMask"].values
    c["prev_fire"] = (pm == 1).astype(np.float32)
    c["prev_uncertain"] = (pm == -1).astype(np.float32)
    c["wind_u"], c["wind_v"] = F.wind_uv(c["th"], c["vs"])
    c["slope"], c["aspect_x"], c["aspect_y"] = F.slope_aspect(c["elevation"])
    c["wind_slope_align"] = F.wind_slope_alignment(c["wind_u"], c["wind_v"], c["slope"], c["aspect_x"], c["aspect_y"])
    c["fm1"], c["fm10"], c["rh"] = F.dead_fuel_moisture(c["sph"], c["tmmx"], c["elevation"])
    target = fm.astype(np.float32)          # -1 uncertain, 0, 1
    return c, target


class Stats:
    """Per-channel clip bounds (train percentiles) + mean/std after clipping. Vector pairs share a symmetric scale."""
    def __init__(self, chans, names, lo=0.5, hi=99.5):
        self.s = {}
        for n in names:
            if n in NO_NORM: continue
            a = chans[n]
            l, h = np.percentile(a, [lo, hi])
            if any(n in p for p in VECTOR_PAIRS):
                m = max(abs(l), abs(h)); l, h = -m, m
            b = np.clip(a, l, h)
            mu, sd = (0.0, float(b.std()) + 1e-6) if any(n in p for p in VECTOR_PAIRS) else (float(b.mean()), float(b.std()) + 1e-6)
            self.s[n] = (float(l), float(h), mu, sd)

    def apply(self, chans, names):
        out = []
        for n in names:
            a = chans[n]
            if n in self.s:
                l, h, mu, sd = self.s[n]; a = (np.clip(a, l, h) - mu) / sd
            out.append(a)
        return np.stack(out, 1).astype(np.float32)  # (N,C,H,W)


def load(root, feature_set, device, splits=("train", "eval", "test")):
    names = FEATURE_SETS[feature_set]
    data, stats = {}, None
    for sp in splits:
        chans, target = build_channels(open_split(root, sp))
        if stats is None: stats = Stats(chans, names)
        x = torch.from_numpy(stats.apply(chans, names)).to(device)
        y = torch.from_numpy(target).to(device)
        data[sp] = (x, y)
    vec_idx = [(names.index(a), names.index(b)) for a, b in VECTOR_PAIRS if a in names and b in names]
    return data, names, vec_idx, stats


def dihedral(x, y, k, vec_idx):
    """Apply one of 8 dihedral ops (k in 0..7) to batch x (B,C,H,W) and y (B,H,W), rotating vector channels too.
    k&1: horizontal flip (x -> -x), k&2: vertical flip (y -> -y), k&4: transpose (swap x,y)."""
    x = x.clone()
    if k & 1:
        x = x.flip(-1); y = y.flip(-1)
        for i, _ in vec_idx: x[:, i] = -x[:, i]
    if k & 2:
        x = x.flip(-2); y = y.flip(-2)
        for _, j in vec_idx: x[:, j] = -x[:, j]
    if k & 4:
        x = x.transpose(-1, -2); y = y.transpose(-1, -2)
        for i, j in vec_idx: x[:, [i, j]] = x[:, [j, i]]
    return x.contiguous(), y.contiguous()
