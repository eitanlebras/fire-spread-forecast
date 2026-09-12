"""Run a trained checkpoint over one WildfireSpreadTS fire and write per-day next-day burn probability GeoTIFFs.

  python -m fsf.predict CKPT.pt FIRE_DIR [--out /root/data/preds] [--device cuda]

Output /root/data/preds/{year}/{fire}/{date}.tif: float32 P(active fire on {date}), in the event's own CRS/grid, made
from the previous day's observation. Tags: forecast_for, made_from, checkpoint. Also probs.npy (T-1, H, W) with
probs[i] = forecast for day i+1 (the layout viz.replay / viz.demo_map take), persistence.npy and dates.json.

Normalization: the checkpoint's cfg["norm"] (means/stds/landcover index) is applied exactly as the training cache was
built (fsf.wfts.preprocess + landcover one-hot). A checkpoint without cfg["norm"] falls back to the authors' 2018+2019
stats, which is what every cache so far was built with; the fallback is reported. Only proj_ch=0 (no OlmoEarth)
checkpoints are supported here; OlmoEarth heads need the S2 composites + encoder (see fsf.train_olmo stage 2)."""
import argparse, glob, json, os, sys, numpy as np, torch, torch.nn.functional as Fn
import rasterio
from . import wfts as W
from .olmo import OlmoUNet
from .train_olmo import expand_landcover, N_LANDCOVER


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg, sd = ck["config"], ck["state_dict"]; mc = cfg["model"]
    if mc.get("proj_ch", 0): raise NotImplementedError("OlmoEarth checkpoints need the S2 composites + encoder; not wired into fsf.predict yet")
    norm = cfg.get("norm")
    if norm is None:
        from src.dataloader.utils import get_means_stds_missing_values  # noqa
        m, s, _ = get_means_stds_missing_values((2018, 2019)); norm = {"means": list(map(float, m)), "stds": list(map(float, s)), "landcover_idx": W.LANDCOVER_IDX, "n_landcover": N_LANDCOVER}
        print("WARNING: checkpoint has no cfg['norm']; using the authors' 2018+2019 stats (what all caches were built with)", file=sys.stderr)
    in_ch = len(ck["names"]) if "names" in ck else 40
    model = OlmoUNet(in_ch, 1, 0, mc["width"], mc["depth"], mc["dropout"]).to(device)
    model.load_state_dict({k: v.to(device) for k, v in sd.items()}); model.eval()
    return model, norm, cfg


@torch.no_grad()
def predict_fire(model, norm, fire_dir, device, batch=8):
    paths = sorted(glob.glob(os.path.join(fire_dir, "*.tif"))); dates = [os.path.basename(p).split("_")[0].replace(".tif", "") for p in paths]
    imgs, profile = [], None
    for p in paths:
        with rasterio.open(p) as ds:
            imgs.append(ds.read()); profile = profile or ds.profile
    imgs = np.stack(imgs)                                                                    # (T,23,H,W) raw
    means, stds = np.asarray(norm["means"], np.float32), np.asarray(norm["stds"], np.float32)
    from src.dataloader.utils import get_indices_of_degree_features  # noqa  (authors' code, same as the cache build)
    x = W.preprocess(imgs, means, stds, get_indices_of_degree_features())                    # (T,24,H,W)
    persistence = (np.nan_to_num(imgs[:, -1], nan=0.0) > 0).astype(np.float32)               # today's active fire
    T, _, H, W_ = x.shape; ph, pw = (-H) % 32, (-W_) % 32
    probs = np.zeros((T - 1, H, W_), np.float32)
    for i in range(0, T - 1, batch):
        xb = torch.from_numpy(x[i:min(i + batch, T - 1)]).to(device).float()               # day T-1 has no next day
        xb = expand_landcover(xb, norm["landcover_idx"])                                     # (B,40,H,W)
        xb = Fn.pad(xb, (0, pw, 0, ph))                                                      # UNet needs multiples of 8; authors crop to x32
        with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
            p = torch.sigmoid(model(xb).float())[:, 0, :H, :W_]
        probs[i:i + len(p)] = p.cpu().numpy()
    return probs, persistence, dates, profile


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("fire_dir"); ap.add_argument("--out", default="/root/data/preds"); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(); fire_dir = a.fire_dir.rstrip("/"); year, fire = fire_dir.split("/")[-2:]
    model, norm, cfg = load_checkpoint(a.ckpt, a.device)
    probs, persistence, dates, profile = predict_fire(model, norm, fire_dir, a.device)
    out = os.path.join(a.out, year, fire); os.makedirs(out, exist_ok=True)
    profile.update(count=1, dtype="float32", nodata=None, compress="deflate")
    for i in range(len(probs)):
        with rasterio.open(os.path.join(out, f"{dates[i + 1]}.tif"), "w", **profile) as dst:
            dst.write(probs[i], 1); dst.update_tags(forecast_for=dates[i + 1], made_from=dates[i], checkpoint=os.path.basename(a.ckpt), run=cfg.get("name", ""))
    np.save(os.path.join(out, "probs.npy"), probs); np.save(os.path.join(out, "persistence.npy"), persistence)
    json.dump({"dates": dates, "checkpoint": a.ckpt, "run": cfg.get("name"), "layout": "probs[i] = forecast for dates[i+1]"}, open(os.path.join(out, "dates.json"), "w"), indent=1)
    nxt = persistence[1:]; valid = np.ones_like(nxt, bool)
    print(f"{year}/{fire}: {len(probs)} forecasts {probs.shape[1]}x{probs.shape[2]} -> {out}")
    for i in range(len(probs)):
        p, y = probs[i], nxt[i]; print(f"  {dates[i]} -> {dates[i+1]}: P mean {p.mean():.4f} max {p.max():.3f} | pixels>0.5: {(p>0.5).sum():5d} | actual fire px: {int(y.sum()):5d} | mean P on burned {p[y>0].mean() if y.any() else float('nan'):.3f} vs unburned {p[y==0].mean():.4f}")


if __name__ == "__main__":
    main()
