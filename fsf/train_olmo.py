"""WildfireSpreadTS tiles (+ OlmoEarth embeddings of the pre-fire Sentinel-2 composites) -> next-day active fire.
  python -m fsf.train_olmo configs/wfts_olmo_frozen.yaml --seeds 0-9 [--set key=value ...]

stage 1: frozen encoder. Embeddings come from `python -m fsf.olmo cache` (per fire tile, (D, 16, 16) for 128 px tiles),
         a learned 1x1 conv projects them to model.proj_ch channels, upsampled and concatenated onto the WFTS channels
         (23 bands after the authors' preprocessing + binary active fire, landcover one-hot expanded per batch -> 40).
         All seeds train together as vmapped copies on GPU-resident data, exactly like fsf.train. use_olmo: false is
         the no-OlmoEarth ablation with the identical decoder.
stage 2: encoder in the graph, only its last olmo.unfreeze_blocks transformer blocks trainable (patch embed and
         encodings frozen), AdamW with two groups: head lr = train.base_lr (x batch scaling), encoder lr =
         olmo.encoder_lr. Seeds run sequentially; head can start from a stage-1 run's weights (init_from).
Every seed appends one row to <results_dir>/runs.parquet (same columns as fsf.train plus stage / olmo fields) and a
detail json with the curve, eval/test metrics next to the persistence baseline and the reliability diagram."""
import argparse, copy, json, math, os, time, uuid
import numpy as np, torch, torch.nn.functional as Fn, yaml
from torch.func import stack_module_state, functional_call, vmap
from . import data as D, metrics as M
from .train import loss_fn, auc_pr_torch, append_parquet, git_sha, parse_seeds, set_nested, clip_per_model
from .olmo import OlmoUNet, OlmoUNetE2E, load_encoder, S2Norm, pad_tiles, fire_months

N_LANDCOVER = 17


def expand_landcover(x, idx):
    """(...,24,H,W) with integer landcover class 1..17 in channel idx -> (...,40,H,W) one-hot (authors' 40-feature input)."""
    oh = Fn.one_hot((x[..., idx, :, :].long() - 1).clamp(0, N_LANDCOVER - 1), N_LANDCOVER).movedim(-1, -3).to(x.dtype)
    return torch.cat([x[..., :idx, :, :], oh, x[..., idx + 1:, :, :]], -3)


def dihedral_nd(t, k):
    """Same op as data.dihedral on the last two dims of any tensor (no vector channels)."""
    if k & 1: t = t.flip(-1)
    if k & 2: t = t.flip(-2)
    if k & 4: t = t.transpose(-1, -2)
    return t.contiguous()


def tile_embeddings(fires, meta, ecache):
    """(N,D,p,p) fp16: the cached embedding of each tile's (fire, row_off, col_off)."""
    tile = ecache["tile"]; emb = ecache["emb"]
    return torch.stack([emb[f"{fires[fi]['year']}/{fires[fi]['fire']}"][r // tile, c // tile] for fi, _, r, c in meta.tolist()])


class S2Tiles:
    """Stage 2: raw composites per fire (NaN-padded to the tile grid) kept on `device`; gather(meta) crops the batch."""
    def __init__(self, fires, s2_root, tile, device):
        self.tile, self.a, self.mo, self.yr = tile, [], [], []
        for f in fires:
            p = f"{s2_root}/{f['year']}/{f['fire']}.npy"; a = np.load(p).astype(np.float32)
            T, C, H, W = a.shape; nr, nc = -(-H // tile), -(-W // tile)
            pad = np.full((T, C, nr * tile, nc * tile), np.nan, np.float32); pad[:, :, :H, :W] = a
            self.a.append(torch.from_numpy(pad).half().to(device))
            m, y = fire_months(p[:-4] + ".json"); self.mo.append(m); self.yr.append(y)
        self.mo, self.yr = torch.tensor(self.mo, device=device), torch.tensor(self.yr, device=device)

    def gather(self, meta):   # meta (B,4) int -> s2 (B,T,12,tile,tile), months (B,T), years (B,T)
        rows = meta.tolist(); t = self.tile
        s2 = torch.stack([self.a[fi][:, :, r:r + t, c:c + t] for fi, _, r, c in rows])
        fi = torch.tensor([m[0] for m in rows], device=self.mo.device)
        return s2, self.mo[fi], self.yr[fi]


class Ensemble:
    """S independent OlmoUNets as stacked parameters; forward/loss vmapped over the seed axis (as fsf.train.Ensemble)."""
    def __init__(self, seeds, in_ch, emb_dim, mc, dev):
        models = []
        for s in seeds:
            torch.manual_seed(s); models.append(OlmoUNet(in_ch, emb_dim, mc.get("proj_ch", 16), mc["width"], mc["depth"], mc["dropout"]).to(dev))
        self.params, self.buffers = stack_module_state(models)
        self.base = copy.deepcopy(models[0]).to("meta"); self.S = len(seeds)
        self.n_params = sum(p[0].numel() for p in self.params.values())

    def _f(self, params, buffers, x, e): return functional_call(self.base, (params, buffers), (x, e))

    def forward_train(self, x, e):   # (S,B,C,H,W), (S,B,D,p,p) -> (S,B,1,H,W)
        return vmap(self._f, in_dims=(0, 0, 0, 0), randomness="different")(self.params, self.buffers, x, e)

    @torch.no_grad()
    def predict(self, x, e, lc_idx, bs=128):   # shared x (N,24,H,W), e (N,D,p,p) -> probs (S,N,H,W)
        self.base.eval(); out = []
        for i in range(0, len(x), bs):
            xb = expand_landcover(x[i:i + bs].float(), lc_idx); eb = e[i:i + bs].float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(torch.sigmoid(vmap(self._f, in_dims=(0, 0, None, None))(self.params, self.buffers, xb, eb).float())[:, :, 0])
        self.base.train(); return torch.cat(out, 1)

    def state(self, i): return {k: v[i].detach().clone() for k, v in self.params.items()}
    def load_state(self, i, st):
        with torch.no_grad():
            for k, v in st.items(): self.params[k][i].copy_(v)


def lr_lambda(warm, total): return lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total))


def finish(cfg, seed, S, names, hist, best, best_ep, ep, probs, ys, prevs, T0, t_train, extra, weights):
    tc, mc = cfg["train"], cfg["model"]; rd = cfg["results_dir"]; os.makedirs(f"{rd}/detail", exist_ok=True)
    run_id = f"{cfg['name']}_s{seed}_{uuid.uuid4().hex[:6]}"
    res = {"run_id": run_id, "name": cfg["name"], "seed": seed, "features": cfg.get("features", "wfts"), "n_channels": len(names), "git_sha": git_sha(),
           "wall_s_arm": time.time() - T0, "epochs_run": ep + 1, "best_epoch": best_ep, "best_eval_auc_pr": float(best), "n_seeds_in_process": S,
           "stage": cfg.get("stage", 1), "use_olmo": bool(cfg.get("use_olmo", True)), "config": json.dumps(cfg), **extra}
    res.update({f"model.{k}": v for k, v in mc.items()}); res.update({f"train.{k}": v for k, v in tc.items()}); res.update({f"olmo.{k}": v for k, v in cfg.get("olmo", {}).items()})
    detail = {"history": hist}
    for sp in ("eval", "test"):
        m = M.evaluate(probs[sp], ys[sp], prevs[sp]); detail[sp] = m; res.update({f"{sp}.{k}": v for k, v in m.items() if k != "reliability"})
    append_parquet(f"{rd}/runs.parquet", res); json.dump({**res, **detail}, open(f"{rd}/detail/{run_id}.json", "w"), indent=1)
    if cfg.get("save_model"): torch.save({"state_dict": weights, "config": cfg, "names": names}, f"{rd}/detail/{run_id}.pt")
    print(f"[{run_id}] TEST auc_pr={res['test.auc_pr']:.4f} persistence={res['test.auc_pr_persistence']:.4f} ece={res['test.ece']:.4f} "
          f"growth_auc_pr={res['test.auc_pr_growth_region']:.4f} (persistence {res['test.auc_pr_growth_region_persistence']:.4f}) best_ep={best_ep}", flush=True)
    return run_id


def load_all(cfg, dev):
    b = torch.load(cfg["wfts_cache"]); fires = b["fires"]; lc = b["landcover_idx"]; names = b["names"][:lc] + [f"landcover_{i + 1}" for i in range(N_LANDCOVER)] + b["names"][lc + 1:]
    # normalization stats travel with every checkpoint (fsf.predict needs them; without them inference is garbage)
    cfg["norm"] = {"means": [float(v) for v in b["stats"]["means"]], "stds": [float(v) for v in b["stats"]["stds"]], "landcover_idx": int(lc), "n_landcover": N_LANDCOVER}
    dd = cfg.get("data_device", dev)
    data = {sp: (b[f"x_{sp}"].to(dd), b[f"y_{sp}"].to(dd), b[f"meta_{sp}"]) for sp in ("train", "eval", "test")}
    ecache = torch.load(cfg["emb_cache"]) if cfg.get("use_olmo", True) and cfg.get("emb_cache") else None
    if ecache is not None:
        emb = {sp: tile_embeddings(fires, data[sp][2], ecache).to(dd) for sp in data}; Dm = ecache["dim"]
    else:
        emb = {sp: torch.zeros(len(data[sp][0]), 1, 1, 1, dtype=torch.float16, device=dd) for sp in data}; Dm = 1
    return b, fires, lc, names, data, emb, Dm


def stage1(cfg, seeds, dev):
    T0 = time.time(); tc, mc = cfg["train"], cfg["model"]; S = len(seeds)
    if not cfg.get("use_olmo", True): mc = {**mc, "proj_ch": 0}
    b, fires, lc, names, data, emb, Dm = load_all(cfg, dev)
    xtr, ytr, _ = data["train"]; xev, yev, _ = data["eval"]; xte, yte, _ = data["test"]; N = len(xtr); vec_idx = b.get("vec_idx", [])
    ens = Ensemble(seeds, len(names), Dm, mc, dev)
    print(f"[{cfg['name']}] stage1 C={len(names)} D={Dm} proj_ch={mc.get('proj_ch', 16)} train={N} eval={len(xev)} test={len(xte)} seeds={seeds} "
          f"params/model={ens.n_params/1e6:.2f}M gpu_mem={torch.cuda.memory_allocated()/1e9:.1f}GB loaded in {time.time()-T0:.0f}s", flush=True)
    bs = tc["batch_size"]; lr = tc["base_lr"] * bs / tc["base_batch"]; spe = max(1, N // bs); total = spe * tc["max_epochs"]
    opt = torch.optim.AdamW(list(ens.params.values()), lr=lr, weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(tc["warmup_steps"], total))
    gens = [torch.Generator(device=dev).manual_seed(s) for s in seeds]; aug_rng = [np.random.default_rng(s) for s in seeds]

    def step(xb, yb, eb):
        with torch.autocast("cuda", dtype=torch.bfloat16): logits = ens.forward_train(xb, eb)
        return vmap(loss_fn, in_dims=(0, 0, None, None))(logits, yb, tc["loss"], tc["pos_weight"])
    step_c = torch.compile(step, mode=tc.get("compile_mode", "reduce-overhead")) if tc.get("compile", False) else step

    yev_t = (yev >= 0).to(dev); yev_pos = (yev == 1).float().to(dev)[yev_t]
    best = torch.full((S,), -1.0); best_ep = [-1] * S; best_state = [None] * S; bad = [0] * S; done = [False] * S; hist = [[] for _ in seeds]
    t_train = time.time(); ep = -1
    for ep in range(tc["max_epochs"]):
        tl = torch.zeros(S, device=dev)
        for _ in range(spe):
            idx = torch.stack([torch.randint(0, N, (bs,), device=dev, generator=g) for g in gens]).to(xtr.device)   # (S,B)
            xb = expand_landcover(xtr[idx].to(dev).float(), lc); yb = ytr[idx].to(dev); eb = emb["train"][idx].to(dev).float()
            if tc["augment"]:
                for i in range(S):
                    k = int(aug_rng[i].integers(8)); xb[i], yb[i] = D.dihedral(xb[i], yb[i], k, vec_idx); eb[i] = dihedral_nd(eb[i], k)
            losses = step_c(xb, yb, eb)
            opt.zero_grad(set_to_none=True); losses.sum().backward(); clip_per_model(ens.params, 1.0); opt.step(); sched.step()
            tl += losses.detach()
        if (ep + 1) % tc["eval_every"] == 0:
            pev = ens.predict(xev.to(dev), emb["eval"].to(dev), lc)
            ev = torch.stack([auc_pr_torch(pev[i][yev_t], yev_pos) for i in range(S)]).cpu(); tl = (tl / spe).cpu()
            for i in range(S):
                if done[i]: continue
                hist[i].append({"epoch": ep, "train_loss": float(tl[i]), "eval_auc_pr": float(ev[i])})
                if ev[i] > best[i]: best[i], best_ep[i], bad[i], best_state[i] = ev[i], ep, 0, ens.state(i)
                else:
                    bad[i] += 1
                    if bad[i] >= tc["patience"]: done[i] = True
            print(f"[{cfg['name']}] ep {ep:3d} loss {tl.mean():.4f} eval_auc_pr mean {ev.mean():.4f} [{' '.join(f'{v:.3f}' for v in ev)}] "
                  f"done {sum(done)}/{S} ({time.time()-t_train:.0f}s)", flush=True)
            if all(done): break
    for i in range(S): ens.load_state(i, best_state[i])
    probs = {sp: ens.predict(x.to(dev), emb[sp].to(dev), lc).cpu().numpy() for sp, (x, _, _) in [("eval", data["eval"]), ("test", data["test"])]}
    ys = {"eval": yev.cpu().numpy(), "test": yte.cpu().numpy()}; prevs = {sp: (data[sp][0][:, -1] > 0.5).float().cpu().numpy() for sp in ("eval", "test")}
    for i, seed in enumerate(seeds):
        finish(cfg, seed, S, names, hist[i], best[i], best_ep[i], ep, {sp: probs[sp][i] for sp in probs}, ys, prevs, T0, t_train, {}, best_state[i])
    print(f"[{cfg['name']}] {S} seeds done in {time.time()-T0:.0f}s (train {time.time()-t_train:.0f}s)", flush=True)


def stage2(cfg, seeds, dev):
    T0 = time.time(); tc, mc, oc = cfg["train"], cfg["model"], cfg["olmo"]
    b, fires, lc, names, data, _, _ = load_all({**cfg, "use_olmo": False}, dev)
    xtr, ytr, mtr = data["train"]; xev, yev, mev = data["eval"]; xte, yte, mte = data["test"]; N = len(xtr); vec_idx = b.get("vec_idx", [])
    enc, Dm = load_encoder(oc.get("model_id", "OLMOEARTH_V1_2_SMALL"), dev); norm = S2Norm(dev); enc_init = copy.deepcopy(enc.state_dict())
    s2 = S2Tiles(fires, cfg["s2_root"], xtr.shape[-1], cfg.get("data_device", dev))
    init = torch.load(cfg["init_from"])["state_dict"] if cfg.get("init_from") else None
    print(f"[{cfg['name']}] stage2 C={len(names)} D={Dm} train={N} eval={len(xev)} test={len(xte)} unfreeze={oc.get('unfreeze_blocks', 2)} "
          f"encoder_lr={oc.get('encoder_lr', 1e-5)} gpu_mem={torch.cuda.memory_allocated()/1e9:.1f}GB loaded in {time.time()-T0:.0f}s", flush=True)
    bs = tc["batch_size"]; lr = tc["base_lr"] * bs / tc["base_batch"]; spe = max(1, N // bs); total = spe * tc["max_epochs"]

    def batch(x, y, meta, idx, aug_k=None):
        xb = expand_landcover(x[idx].to(dev).float(), lc); yb = y[idx].to(dev); sb, mo, yr = s2.gather(meta[idx.cpu()]); sb = sb.to(dev)
        if aug_k is not None: xb, yb = D.dihedral(xb, yb, aug_k, vec_idx); sb = dihedral_nd(sb, aug_k)
        return xb, yb, sb, mo, yr

    def predict(model, x, y, meta, cbs=32):
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(x), cbs):
                idx = torch.arange(i, min(i + cbs, len(x)), device=x.device); xb, _, sb, mo, yr = batch(x, y, meta, idx)
                with torch.autocast("cuda", dtype=torch.bfloat16): out.append(torch.sigmoid(model(xb, sb, mo, yr).float())[:, 0])
        model.train(); return torch.cat(out)

    yev_t = (yev >= 0).to(dev); yev_pos = (yev == 1).float().to(dev)[yev_t]
    for seed in seeds:
        t_seed = time.time(); torch.manual_seed(seed); enc.load_state_dict(enc_init)
        head = OlmoUNet(len(names), Dm, mc.get("proj_ch", 16), mc["width"], mc["depth"], mc["dropout"])
        if init is not None: head.load_state_dict(init)
        model = OlmoUNetE2E(head, enc, norm, oc.get("unfreeze_blocks", 2), oc.get("patch_size", 8), oc.get("input_res", 10)).to(dev).train()
        groups = [{"params": list(head.parameters()), "lr": lr}]
        if model.encoder_params(): groups.append({"params": model.encoder_params(), "lr": oc.get("encoder_lr", 1e-5)})
        opt = torch.optim.AdamW(groups, weight_decay=tc["weight_decay"]); sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(tc["warmup_steps"], total))
        g = torch.Generator(device=dev).manual_seed(seed); aug_rng = np.random.default_rng(seed)
        best, best_ep, best_state, bad, hist, ep = -1.0, -1, None, 0, [], -1
        n_enc = sum(p.numel() for p in model.encoder_params()); n_head = sum(p.numel() for p in head.parameters())
        print(f"[{cfg['name']} s{seed}] trainable: head {n_head/1e6:.2f}M encoder {n_enc/1e6:.2f}M", flush=True)
        t_train = time.time()
        for ep in range(tc["max_epochs"]):
            tl = 0.0
            for _ in range(spe):
                idx = torch.randint(0, N, (bs,), device=dev, generator=g).to(xtr.device)
                xb, yb, sb, mo, yr = batch(xtr, ytr, mtr, idx, int(aug_rng.integers(8)) if tc["augment"] else None)
                with torch.autocast("cuda", dtype=torch.bfloat16): logits = model(xb, sb, mo, yr)
                loss = loss_fn(logits, yb, tc["loss"], tc["pos_weight"])
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_([p for gr in groups for p in gr["params"]], 1.0); opt.step(); sched.step(); tl += loss.detach().item()
            if (ep + 1) % tc["eval_every"] == 0:
                pev = predict(model, xev, yev, mev); ev = float(auc_pr_torch(pev[yev_t], yev_pos)); tl /= spe
                hist.append({"epoch": ep, "train_loss": tl, "eval_auc_pr": ev})
                if ev > best: best, best_ep, bad = ev, ep, 0; best_state = {k: v.detach().clone() for k, v in model.named_parameters() if v.requires_grad}
                else: bad += 1
                print(f"[{cfg['name']} s{seed}] ep {ep:3d} loss {tl:.4f} eval_auc_pr {ev:.4f} best {best:.4f}@{best_ep} ({time.time()-t_train:.0f}s)", flush=True)
                if bad >= tc["patience"]: break
        model.load_state_dict(best_state, strict=False)
        probs = {"eval": predict(model, xev, yev, mev).cpu().numpy(), "test": predict(model, xte, yte, mte).cpu().numpy()}
        ys = {"eval": yev.cpu().numpy(), "test": yte.cpu().numpy()}; prevs = {sp: (data[sp][0][:, -1] > 0.5).float().cpu().numpy() for sp in ("eval", "test")}
        extra = {"n_trainable_encoder": n_enc, "n_trainable_head": n_head, "init_from": cfg.get("init_from") or "", "wall_s_seed": time.time() - t_seed}
        finish(cfg, seed, 1, names, hist, best, best_ep, ep, probs, ys, prevs, T0, t_train, extra, best_state)
        del model, opt; torch.cuda.empty_cache()
    print(f"[{cfg['name']}] {len(seeds)} seeds done in {time.time()-T0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--seeds", default="0-9"); ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config))
    for kv in a.set: k, v = kv.split("=", 1); set_nested(cfg, k, v)
    torch.backends.cudnn.benchmark = True
    (stage2 if int(cfg.get("stage", 1)) == 2 else stage1)(cfg, parse_seeds(a.seeds), "cuda")


if __name__ == "__main__":
    main()
