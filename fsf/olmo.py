"""OlmoEarth encoder over the Sentinel-2 monthly composites from fsf/s2_pull.py.

  python -m fsf.olmo cache S2_ROOT OUT.pt [--model OLMOEARTH_V1_2_SMALL] [--tile 128] [--patch 8] [--input-res 10]
                                          [--batch 16] [--limit N]

Per fire event: the (4, 12, H, W) composite is padded with NaN to a multiple of `tile`, cut into tile x tile windows
(the same windows fsf.wfts uses for the tensor cache: row/col offsets are multiples of the tile), each window is
normalized with OlmoEarth's own Normalizer, pushed through the frozen encoder with the real month/year timestamps,
and the patch tokens are averaged over time and band sets -> (D, tile/patch, tile/patch) fp16 per window.
OUT.pt = {"emb": {"year/fire_id": (nr, nc, D, p, p) fp16}, "dim", "tile", "patch", "model_id", "input_res", "months"}.
Trainers index it with the tile cache's (fire, row_off, col_off) so no tile is ever embedded twice.

Missing pixels (NaN: no cloud-free scene) are tokens marked MISSING; a patch is missing when less than half its
pixels are valid. Embeddings are independent of the fire day t: the composites are the months before the fire.

Requires `olmoearth-pretrain-minimal` (PyPI) in the venv; weights come from Hugging Face (OlmoEarth Artifact
License: cite Ai2, ship the license, carry the no-military / no-extractive restrictions downstream).
"""
import argparse, glob, json, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from .unet import UNet

PATCH = 8
TILE = 128
ONLINE, MISSING = 0, 3   # olmoearth_pretrain_minimal ... utils.datatypes.MaskValue.{ONLINE_ENCODER, MISSING}.value


def load_encoder(model_id="OLMOEARTH_V1_2_SMALL", device="cuda"):
    """Pretrained encoder (eval mode, on device) + its embedding size. Model ids: olmoearth_pretrain_minimal.ModelID names."""
    from olmoearth_pretrain_minimal import ModelID, load_model_from_id
    model = load_model_from_id(ModelID[model_id], load_weights=True)
    enc = model.encoder.to(device).eval()
    return enc, int(enc.embedding_size)


class S2Norm:
    """OlmoEarth's Normalizer for sentinel2_l2a as an affine map on GPU tensors: x -> x * scale + offset."""
    def __init__(self, device="cuda"):
        from olmoearth_pretrain_minimal import Normalizer
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality
        n = Normalizer(std_multiplier=2.0); z = np.zeros((1, 12), np.float32)
        a = n.normalize(Modality.SENTINEL2_L2A, z); b = n.normalize(Modality.SENTINEL2_L2A, z + 1) - a
        self.offset = torch.tensor(a[0], dtype=torch.float32, device=device); self.scale = torch.tensor(b[0], dtype=torch.float32, device=device)
        self.bands = list(Modality.SENTINEL2_L2A.band_order)
    def __call__(self, x): return x * self.scale + self.offset   # (..., 12)


def n_bandsets(enc, modality="sentinel2_l2a"):
    try: return int(enc.patch_embeddings.tokenization_config.get_num_bandsets(modality))
    except Exception: return 3   # S2 L2A band sets: (B02,B03,B04,B08), (B05,B06,B07,B8A,B11,B12), (B01,B09)


def embed(enc, norm, s2, months, years, patch=PATCH, input_res=10, autocast=True):
    """s2 (B,T,12,h,w) float DN with NaN=missing, months (B,T) long 0-11, years (B,T) long
    -> (B, D, h/patch, w/patch) float32: mean of the encoder's unmasked patch tokens over time and band sets."""
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import MaskedOlmoEarthSample
    B, T, C, h, w = s2.shape
    x = s2.float().permute(0, 3, 4, 1, 2)                                      # (B,h,w,T,C)
    valid = torch.isfinite(x).all(-1)                                          # (B,h,w,T)
    x = torch.nan_to_num(norm(x), nan=0.0)
    vf = F.avg_pool2d(valid.permute(0, 3, 1, 2).float(), patch)                # (B,T,h/p,w/p) valid fraction per patch
    missing = vf < 0.5
    missing[missing.flatten(1).all(1)] = False                                 # fully-missing sample: encode zeros rather than nothing
    mask = F.interpolate(missing.float(), scale_factor=patch, mode="nearest").bool().permute(0, 2, 3, 1)   # (B,h,w,T)
    mask = torch.where(mask, MISSING, ONLINE).long()
    mask = mask[..., None].expand(-1, -1, -1, -1, n_bandsets(enc)).contiguous()   # (B,h,w,T,band_sets): last axis indexed per band set
    ts = torch.stack([torch.full_like(months, 15), months, years], -1)         # (B,T,3) = day, month (0-indexed), year
    sample = MaskedOlmoEarthSample(timestamps=ts, sentinel2_l2a=x, sentinel2_l2a_mask=mask)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        out = enc(sample, patch_size=patch, input_res=input_res, fast_pass=False)["tokens_and_masks"]
    tok, m = out.sentinel2_l2a.float(), out.sentinel2_l2a_mask                 # (B,PH,PW,T,BS,D), (B,PH,PW,T,BS)
    keep = (m == ONLINE).float().unsqueeze(-1)
    e = (tok * keep).sum((3, 4)) / keep.sum((3, 4)).clamp(min=1)               # (B,PH,PW,D)
    return e.permute(0, 3, 1, 2).contiguous()


def pad_tiles(a, tile=TILE):
    """(T,C,H,W) -> (nr*nc, T, C, tile, tile) NaN-padded windows in row-major order, plus (nr, nc)."""
    T, C, H, W = a.shape; nr, nc = -(-H // tile), -(-W // tile)
    p = np.full((T, C, nr * tile, nc * tile), np.nan, np.float32); p[:, :, :H, :W] = a
    t = p.reshape(T, C, nr, tile, nc, tile).transpose(2, 4, 0, 1, 3, 5).reshape(nr * nc, T, C, tile, tile)
    return t, (nr, nc)


def fire_months(json_path):
    ts = json.load(open(json_path))["timesteps"]
    ym = [t["month"].split("-") for t in ts]
    return [int(m) - 1 for _, m in ym], [int(y) for y, _ in ym]


def cache_embeddings(s2_root, out_path, model_id="OLMOEARTH_V1_2_SMALL", tile=TILE, patch=PATCH, input_res=10, batch=16, limit=0, device="cuda"):
    enc, D = load_encoder(model_id, device); norm = S2Norm(device); t0 = time.time()
    paths = sorted(glob.glob(f"{s2_root}/*/fire_*.npy"))
    if limit: paths = paths[:limit]
    emb, months_all = {}, {}
    print(f"{model_id}: D={D}, {len(paths)} fires from {s2_root}", flush=True)
    for i, p in enumerate(paths):
        key = f"{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)[:-4]}"
        a = np.load(p).astype(np.float32); months, years = fire_months(p[:-4] + ".json")
        tiles, (nr, nc) = pad_tiles(a, tile); out = []
        mo = torch.tensor(months, device=device); yr = torch.tensor(years, device=device)
        with torch.no_grad():
            for j in range(0, len(tiles), batch):
                x = torch.from_numpy(tiles[j:j + batch]).to(device); b = len(x)
                out.append(embed(enc, norm, x, mo.expand(b, -1), yr.expand(b, -1), patch, input_res).half().cpu())
        e = torch.cat(out).reshape(nr, nc, D, tile // patch, tile // patch)
        emb[key] = e; months_all[key] = months
        if i % 20 == 0 or i == len(paths) - 1:
            print(f"[{i + 1}/{len(paths)} {time.time() - t0:.0f}s] {key} {a.shape} -> {tuple(e.shape)} nan_frac={np.isnan(a[:, 0]).mean():.3f}", flush=True)
    blob = {"emb": emb, "dim": D, "tile": tile, "patch": patch, "model_id": model_id, "input_res": input_res, "months": months_all}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True); torch.save(blob, out_path)
    print(f"saved {out_path}: {len(emb)} fires, {sum(v.numel() for v in emb.values()) * 2 / 1e6:.0f} MB fp16, {time.time() - t0:.0f}s", flush=True)


class OlmoUNet(nn.Module):
    """WFTS channels + a learned 1x1 projection of OlmoEarth patch embeddings (bilinearly upsampled to the pixel grid) -> UNet.
    proj_ch=0 gives the no-OlmoEarth ablation with an identical decoder."""
    def __init__(self, in_ch, emb_dim, proj_ch=16, width=32, depth=3, dropout=0.1):
        super().__init__()
        self.proj = nn.Conv2d(emb_dim, proj_ch, 1) if proj_ch else None
        self.unet = UNet(in_ch + proj_ch, width, depth, dropout)

    def forward(self, x, emb=None):   # x (B,C,H,W), emb (B,D,h,w)
        if self.proj is not None:
            e = F.interpolate(self.proj(emb.float()), size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, e], 1)
        return self.unet(x)


class OlmoUNetE2E(nn.Module):
    """Stage 2: encoder in the graph. Everything frozen except the last `unfreeze_blocks` transformer blocks
    (patch embed, encodings and the rest stay frozen). forward(x, s2, months, years)."""
    def __init__(self, head, enc, norm, unfreeze_blocks=2, patch=PATCH, input_res=10):
        super().__init__()
        self.head, self.enc, self.norm, self.patch, self.input_res = head, enc, norm, patch, input_res
        for p in enc.parameters(): p.requires_grad_(False)
        self.trainable_enc = list(enc.blocks[len(enc.blocks) - unfreeze_blocks:]) if unfreeze_blocks else []
        for blk in self.trainable_enc:
            for p in blk.parameters(): p.requires_grad_(True)

    def encoder_params(self): return [p for blk in self.trainable_enc for p in blk.parameters()]

    def forward(self, x, s2, months, years):
        emb = embed(self.enc, self.norm, s2, months, years, self.patch, self.input_res, autocast=True)
        return self.head(x, emb)

    def train(self, mode=True):   # encoder stays in eval mode (no drop-path / band dropout); grads still flow through it
        super().train(mode); self.enc.eval(); return self


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["cache"]); ap.add_argument("s2_root"); ap.add_argument("out")
    ap.add_argument("--model", default="OLMOEARTH_V1_2_SMALL"); ap.add_argument("--tile", type=int, default=TILE); ap.add_argument("--patch", type=int, default=PATCH)
    ap.add_argument("--input-res", type=int, default=10); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cache_embeddings(a.s2_root, a.out, a.model, a.tile, a.patch, a.input_res, a.batch, a.limit)
