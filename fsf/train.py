"""One process per arm; all seeds of the arm train simultaneously as vmapped model copies on GPU-resident data.
Usage: python -m fsf.train configs/x.yaml --seeds 0-9 [--set key=value ...]
Per seed: one row (config + final metrics) appended to <results_dir>/runs.parquet (file-locked), plus
<results_dir>/detail/<run_id>.json with the curve and reliability diagram. Best weights kept in memory, written once."""
import argparse, json, os, time, uuid, subprocess, math, fcntl, copy
import numpy as np, torch, torch.nn.functional as Fn, yaml, pandas as pd
from torch.func import stack_module_state, functional_call, vmap
from . import data as D, metrics as M
from .unet import UNet


def git_sha():
    try: return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception: return "nogit"


def set_nested(cfg, key, val):
    d = cfg; ks = key.split(".")
    for k in ks[:-1]: d = d.setdefault(k, {})
    d[ks[-1]] = yaml.safe_load(val)


def parse_seeds(s):
    if "-" in s: a, b = s.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def loss_fn(logits, y, kind, pos_weight):
    """Per-sample-batch loss. y int8 in {-1,0,1}: -1 (uncertain) excluded. logits (B,1,H,W)."""
    logits = logits[:, 0].float(); valid = (y >= 0).float(); t = y.clamp(min=0).float()
    if kind in ("bce", "bce_dice"):
        w = valid * (1 + (pos_weight - 1) * t)
        bce = (Fn.binary_cross_entropy_with_logits(logits, t, reduction="none") * w).sum() / w.sum()
        if kind == "bce": return bce
    p = torch.sigmoid(logits) * valid
    inter = (p * t).sum((1, 2)); dice = (1 - (2 * inter + 1) / (p.sum((1, 2)) + (t * valid).sum((1, 2)) + 1)).mean()
    return dice if kind == "dice" else bce + dice


def auc_pr_torch(prob, y):
    """Average precision on GPU (sklearn definition: sum over positives of precision at that rank). prob,y flat; y in {0,1}."""
    o = torch.argsort(prob, descending=True); ys = y[o].float()
    tp = ys.cumsum(0); prec = tp / torch.arange(1, len(ys) + 1, device=ys.device)
    return (prec * ys).sum() / ys.sum().clamp(min=1)


def append_parquet(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        df = pd.DataFrame([row])
        if os.path.exists(path): df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
        df.to_parquet(path + ".tmp", index=False); os.replace(path + ".tmp", path)


class Ensemble:
    """S independent UNets held as stacked parameters; forward/loss vmapped over the seed axis."""
    def __init__(self, seeds, in_ch, mc, dev):
        models = []
        for s in seeds:
            torch.manual_seed(s); models.append(UNet(in_ch, mc["width"], mc["depth"], mc["dropout"]).to(dev))
        self.params, self.buffers = stack_module_state(models)
        self.base = copy.deepcopy(models[0]).to("meta"); self.S = len(seeds)
        self.n_params = sum(p[0].numel() for p in self.params.values())

    def _f(self, params, buffers, x): return functional_call(self.base, (params, buffers), (x,))

    def forward_train(self, x):   # x (S,B,C,H,W) -> logits (S,B,1,H,W); dropout differs per model
        return vmap(self._f, in_dims=(0, 0, 0), randomness="different")(self.params, self.buffers, x)

    @torch.no_grad()
    def predict(self, x, bs=256):  # shared x (N,C,H,W) -> probs (S,N,H,W) fp32; chunk keeps S*bs activations small
        self.base.eval(); out = []
        for i in range(0, len(x), bs):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(torch.sigmoid(vmap(self._f, in_dims=(0, 0, None))(self.params, self.buffers, x[i:i + bs].float()).float())[:, :, 0])
        self.base.train(); return torch.cat(out, 1)

    def state(self, i): return {k: v[i].detach().clone() for k, v in self.params.items()}
    def load_state(self, i, st):
        with torch.no_grad():
            for k, v in st.items(): self.params[k][i].copy_(v)


def clip_per_model(params, max_norm):
    gs = [p.grad for p in params.values() if p.grad is not None]
    norm = torch.sqrt(sum(g.flatten(1).pow(2).sum(1) for g in gs))            # (S,)
    scale = (max_norm / (norm + 1e-6)).clamp(max=1.0)
    for g in gs: g.mul_(scale.view(-1, *[1] * (g.dim() - 1)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--seeds", default="0-9"); ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config))
    for kv in a.set: k, v = kv.split("=", 1); set_nested(cfg, k, v)
    seeds = parse_seeds(a.seeds); S = len(seeds); dev = "cuda"; T0 = time.time()
    tc, mc = cfg["train"], cfg["model"]; torch.backends.cudnn.benchmark = True
    data, names, vec_idx, _ = D.load(cfg["data_root"], cfg["features"], dev)
    xtr, ytr = data["train"]; xev, yev = data["eval"]; xte, yte = data["test"]; N = len(xtr)
    ens = Ensemble(seeds, len(names), mc, dev)
    print(f"[{cfg['name']}] C={len(names)} train={N} eval={len(xev)} test={len(xte)} seeds={seeds} params/model={ens.n_params/1e6:.2f}M "
          f"gpu_mem={torch.cuda.memory_allocated()/1e9:.1f}GB loaded in {time.time()-T0:.0f}s", flush=True)
    bs = tc["batch_size"]; lr = tc["base_lr"] * bs / tc["base_batch"]; spe = N // bs; total = spe * tc["max_epochs"]; warm = tc["warmup_steps"]
    opt = torch.optim.AdamW(list(ens.params.values()), lr=lr, weight_decay=tc["weight_decay"])  # elementwise => independent per model
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    gens = []
    for s in seeds: g = torch.Generator(device=dev); g.manual_seed(s); gens.append(g)
    aug_rng = [np.random.default_rng(s) for s in seeds]

    def step(xb, yb):
        with torch.autocast("cuda", dtype=torch.bfloat16): logits = ens.forward_train(xb)
        losses = vmap(loss_fn, in_dims=(0, 0, None, None))(logits, yb, tc["loss"], tc["pos_weight"])
        return losses
    step_c = torch.compile(step, mode=tc.get("compile_mode", "reduce-overhead")) if tc.get("compile", True) else step

    yev_t = (yev >= 0); yev_pos = (yev == 1).float()[yev_t]
    best = torch.full((S,), -1.0); best_ep = [-1] * S; best_state = [None] * S; bad = [0] * S; done = [False] * S; hist = [[] for _ in seeds]
    t_train = time.time(); ep = -1
    for ep in range(tc["max_epochs"]):
        tl = torch.zeros(S, device=dev)
        for _ in range(spe):
            idx = torch.stack([torch.randint(0, N, (bs,), device=dev, generator=g) for g in gens])   # (S,B) per-seed sampling
            xb, yb = xtr[idx].float(), ytr[idx]                                                         # (S,B,C,H,W), (S,B,H,W); fp16 cache -> fp32
            if tc["augment"]:
                for i in range(S): xb[i], yb[i] = D.dihedral(xb[i], yb[i], int(aug_rng[i].integers(8)), vec_idx)
            losses = step_c(xb, yb)
            opt.zero_grad(set_to_none=True); losses.sum().backward(); clip_per_model(ens.params, 1.0); opt.step(); sched.step()
            tl += losses.detach()
        if (ep + 1) % tc["eval_every"] == 0:
            pev = ens.predict(xev)                                                   # (S,Nev,H,W)
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
    prev_idx = names.index("prev_fire"); probs = {sp: ens.predict(x).cpu().numpy() for sp, (x, _) in [("eval", (xev, yev)), ("test", (xte, yte))]}
    ys = {"eval": yev.cpu().numpy(), "test": yte.cpu().numpy()}; prevs = {"eval": (xev[:, prev_idx] > 0.5).float().cpu().numpy(), "test": (xte[:, prev_idx] > 0.5).float().cpu().numpy()}
    rd = cfg["results_dir"]; os.makedirs(f"{rd}/detail", exist_ok=True)
    for i, seed in enumerate(seeds):
        run_id = f"{cfg['name']}_s{seed}_{uuid.uuid4().hex[:6]}"
        res = {"run_id": run_id, "name": cfg["name"], "seed": seed, "features": cfg["features"], "n_channels": len(names), "git_sha": git_sha(),
               "wall_s_arm": time.time() - T0, "epochs_run": ep + 1, "best_epoch": best_ep[i], "best_eval_auc_pr": float(best[i]), "lr": lr,
               "n_seeds_in_process": S, "config": json.dumps(cfg)}
        res.update({f"model.{k}": v for k, v in mc.items()}); res.update({f"train.{k}": v for k, v in tc.items()})
        detail = {"history": hist[i]}
        for sp in ("eval", "test"):
            m = M.evaluate(probs[sp][i], ys[sp], prevs[sp]); detail[sp] = m; res.update({f"{sp}.{k}": v for k, v in m.items() if k != "reliability"})
        append_parquet(f"{rd}/runs.parquet", res); json.dump({**res, **detail}, open(f"{rd}/detail/{run_id}.json", "w"), indent=1)
        if cfg.get("save_model"): torch.save({"state_dict": best_state[i], "config": cfg, "names": names}, f"{rd}/detail/{run_id}.pt")
        print(f"[{run_id}] TEST auc_pr={res['test.auc_pr']:.4f} persistence={res['test.auc_pr_persistence']:.4f} ece={res['test.ece']:.4f} "
              f"growth_auc_pr={res['test.auc_pr_growth_region']:.4f} best_ep={best_ep[i]}", flush=True)
    print(f"[{cfg['name']}] {S} seeds done in {time.time()-T0:.0f}s (train {time.time()-t_train:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
