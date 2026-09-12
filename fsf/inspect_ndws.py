"""Step 1: load NDWS zarr splits, print shapes, channel keys, class balance. -1 = uncertain, handled explicitly."""
import sys, zipfile, os, numpy as np, xarray as xr
root = sys.argv[1] if len(sys.argv) > 1 else "/workspace/data/ndws/data"
for split in ["train", "eval", "test"]:
    zp = f"{root}/{split}.zarr.zip"; d = f"{root}/{split}.zarr"
    if not os.path.exists(d):
        with zipfile.ZipFile(zp) as z: z.extractall(d)
    ds = xr.open_zarr(f"{d}/{split}.zarr")  # zip nests split.zarr/split.zarr
    print(f"\n=== {split} ===")
    print("dims:", dict(ds.sizes))
    print("vars:", list(ds.data_vars))
    fm = ds["FireMask"].values; pm = ds["PrevFireMask"].values
    n = fm.size
    for name, m in [("FireMask", fm), ("PrevFireMask", pm)]:
        u, c = np.unique(m, return_counts=True)
        print(f"{name} shape={m.shape} values={dict(zip(u.tolist(), (c/n).round(5).tolist()))}")
    valid = fm >= 0
    print(f"FireMask positive rate (valid px only): {fm[valid].mean():.5f}; uncertain px: {(~valid).mean():.5f}")
    growth = (fm == 1) & (pm == 0); persist = (fm == 1) & (pm == 1)
    print(f"growth px (0->1): {growth.mean():.5f}  persisting (1->1): {persist.mean():.5f}  samples w/ any fire t+1: {(fm==1).any(axis=(1,2)).mean():.3f}")
    if split == "train":
        for v in ds.data_vars:
            a = ds[v].values.astype(np.float32); f = a[np.isfinite(a)]
            print(f"  {v:14s} dtype={ds[v].dtype} nan={np.isnan(a).mean():.4f} min={f.min():.3f} mean={f.mean():.3f} max={f.max():.3f}")
