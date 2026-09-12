# fire-spread-forecast

Next-day wildfire spread as per-pixel segmentation: mask(t) + observations(t) -> P(burning at t+1).
Model: **fire-spread-forecast-v1-small**, a derivative of **allenai/OlmoEarth-v1_2-Small** (see [MODEL_CARD.md](MODEL_CARD.md);
base_model: allenai/OlmoEarth-v1_2-Small). Encoder weights fall under the [OlmoEarth Artifact License](LICENSE-OlmoEarth.txt),
whose no-military / no-extractive use restrictions apply to everything derived here.

## Pipeline (all on the cloud box; nothing is downloaded locally)

1. `scripts/bootstrap_pod.sh` — NDWS + WildfireSpreadTS fires (HTTP range reads from Zenodo), venv, OlmoEarth package.
2. `python fsf/s2_pull.py --workers 8` — Sentinel-2 monthly composites per fire event, OlmoEarth band order (`/root/data/s2`).
3. `python -m fsf.wfts cache /root/data/wfts/tif /root/data/wfts/cache/wfts.pt` — fp16 tile cache.
4. `python -m fsf.olmo cache /root/data/s2 /root/data/wfts/cache/olmo_v1_2_small.pt` — frozen-encoder embeddings per tile.
5. Arms (5 seeds each, one process per arm, `scripts/launch_arms.sh`-style detached runs):
   - `configs/wfts_only.yaml` — no OlmoEarth (the ablation)
   - `configs/wfts_olmo_frozen.yaml` — stage 1, frozen encoder + learned 1x1 projection
   - `configs/wfts_olmo_ft.yaml` — stage 2, last 2 encoder blocks unfrozen, head lr 1e-3 / encoder lr 1e-5
   NDWS baselines: `configs/base.yaml`, `derived.yaml`, `raw_th.yaml` via `python -m fsf.train`.

Every seed appends a row (config + metrics) to `/root/fsf/results/runs.parquet`; `fsf/report.py` reads it. Metrics are
always paired with the persistence baseline on the same pixels; never report full-mask IoU.
