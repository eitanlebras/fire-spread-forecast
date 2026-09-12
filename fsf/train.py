"""One process per arm. Loads the dataset onto the GPU once, compiles the UNet once, then loops over seeds.
Usage: python -m fsf.train configs/x.yaml --seeds 0-9 [--set key=value ...]
Each seed appends one row (config + final metrics) to <results_dir>/runs.parquet (file-locked) and writes
<results_dir>/detail/<run_id>.json with the training curve and reliability diagram."""
import argparse, json, os, time, uuid, subprocess, math, copy, fcntl
import numpy as np, torch, torch.nn.functional as Fn, yaml, pandas as pd
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
    """y in {-1,0,1}; -1 (uncertain) pixels excluded from the loss. logits (B,1,H,W)."""
    logits = logits[:, 0]; valid = (y >= 0).float(); t = y.clamp(min=0)
    if kind in ("bce", "bce_dice"):
        w = valid * torch.where(t > 0, torch.full_like(t, pos_weight), torch.ones_like(t))
        bce = (Fn.binary_cross_entropy_with_logits(logits, t, reduction="none") * w).sum() / w.sum()
        if kind == "bce": return bce
    p = torch.sigmoid(logits) * valid
    inter = (p * t).sum((1, 2)); dice = (1 - (2 * inter + 1) / (p.sum((1, 2)) + (t * valid).sum((1, 2)) + 1)).mean()
    return dice if kind == "dice" else bce + dice


@torch.no_grad()
def predict(model, x, bs=1024):
    model.eval(); out = []
    for i in range(0, len(x), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16): out.append(torch.sigmoid(model(x[i:i + bs]).float())[:, 0])
    model.train(); return torch.cat(out)


def append_parquet(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        df = pd.DataFrame([row])
        if os.path.exists(path): df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
        df.to_parquet(path + ".tmp", index=False); os.replace(path + ".tmp", path)


def reset_weights(m):
    if hasattr(m, "reset_parameters"): m.reset_parameters()


def run_seed(cfg, seed, data, names, vec_idx, model, compiled, dev, t_start):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed); gen = torch.Generator(device=dev); gen.manual_seed(seed)
    model.apply(reset_weights)                     # fresh init in place: compiled graph + CUDA-graph buffers stay valid
    xtr, ytr = data["train"]; xev, yev = data["eval"]; xte, yte = data["test"]
    tc = cfg["train"]; bs = tc["batch_size"]; lr = tc["base_lr"] * bs / tc["base_batch"]
    steps_per_epoch = len(xtr) // bs             # drop_last: static shapes for CUDA graphs
    total = steps_per_epoch * tc["max_epochs"]; warm = tc["warmup_steps"]
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    run_id = f"{cfg['name']}_s{seed}_{uuid.uuid4().hex[:6]}"; t0 = time.time()
    best, best_state, best_ep, hist, bad = -1.0, None, -1, [], 0
    for ep in range(tc["max_epochs"]):
        perm = torch.randperm(len(xtr), device=dev, generator=gen); tl = 0.0
        for i in range(steps_per_epoch):
            idx = perm[i * bs:(i + 1) * bs]; xb, yb = xtr[idx], ytr[idx]
            if tc["augment"]: xb, yb = D.dihedral(xb, yb, int(rng.integers(8)), vec_idx)
            with torch.autocast("cuda", dtype=torch.bfloat16): logits = compiled(xb)
            loss = loss_fn(logits.float(), yb, tc["loss"], tc["pos_weight"])
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
            tl += loss.detach()
        if (ep + 1) % tc["eval_every"] == 0:
            ev = M.auc_pr(predict(model, xev).cpu().numpy(), yev.cpu().numpy()); tl = float(tl) / steps_per_epoch
            hist.append({"epoch": ep, "train_loss": tl, "eval_auc_pr": ev})
            print(f"  [{cfg['name']} s{seed}] ep {ep:3d} loss {tl:.4f} eval_auc_pr {ev:.4f} best {best:.4f}@{best_ep} ({time.time()-t0:.0f}s)", flush=True)
            if ev > best: best, best_ep, bad = ev, ep, 0; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= tc["patience"]: break
    model.load_state_dict(best_state)
    res = {"run_id": run_id, "name": cfg["name"], "seed": seed, "features": cfg["features"], "n_channels": len(names), "git_sha": git_sha(),
           "wall_s": time.time() - t0, "epochs_run": ep + 1, "best_epoch": best_ep, "best_eval_auc_pr": best, "lr": lr, "config": json.dumps(cfg)}
    for k, v in cfg["model"].items(): res[f"model.{k}"] = v
    for k, v in tc.items(): res[f"train.{k}"] = v
    detail = {"history": hist}; prev_idx = names.index("prev_fire")
    for split, (x, y) in [("eval", (xev, yev)), ("test", (xte, yte))]:
        prob = predict(model, x).cpu().numpy(); yn = y.cpu().numpy(); prev = (x[:, prev_idx] > 0.5).float().cpu().numpy()
        m = M.evaluate(prob, yn, prev); detail[split] = m
        res.update({f"{split}.{k}": v for k, v in m.items() if k != "reliability"})
    rd = cfg["results_dir"]; os.makedirs(f"{rd}/detail", exist_ok=True)
    append_parquet(f"{rd}/runs.parquet", res)
    json.dump({**res, **detail}, open(f"{rd}/detail/{run_id}.json", "w"), indent=1)
    if cfg.get("save_model"): torch.save({"state_dict": best_state, "config": cfg, "names": names}, f"{rd}/detail/{run_id}.pt")
    print(f"[{run_id}] TEST auc_pr={res['test.auc_pr']:.4f} persistence={res['test.auc_pr_persistence']:.4f} ece={res['test.ece']:.4f} "
          f"growth_auc_pr={res['test.auc_pr_growth_region']:.4f} epochs={ep+1} wall={res['wall_s']:.0f}s total={time.time()-t_start:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--seeds", default="0"); ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config))
    for kv in a.set: k, v = kv.split("=", 1); set_nested(cfg, k, v)
    dev = "cuda"; t_start = time.time(); torch.backends.cudnn.benchmark = True
    data, names, vec_idx, _ = D.load(cfg["data_root"], cfg["features"], dev)
    print(f"[{cfg['name']}] features={cfg['features']} C={len(names)} train={len(data['train'][0])} eval={len(data['eval'][0])} "
          f"test={len(data['test'][0])} loaded in {time.time()-t_start:.0f}s", flush=True)
    mc = cfg["model"]
    model = UNet(len(names), mc["width"], mc["depth"], mc["dropout"]).to(dev)
    compiled = torch.compile(model, mode=cfg["train"].get("compile_mode", "reduce-overhead")) if cfg["train"].get("compile", True) else model
    for seed in parse_seeds(a.seeds):
        run_seed(cfg, seed, data, names, vec_idx, model, compiled, dev, t_start)
    print(f"[{cfg['name']}] all seeds done in {time.time()-t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
