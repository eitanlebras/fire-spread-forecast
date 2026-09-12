# Checkpoints

`fire-spread-forecast-v1-small_seed0.pt` — Stage 2 (last 2 OlmoEarth encoder blocks + head) weights of run `wfts_olmo_ft_s0_258814`,
derivative of allenai/OlmoEarth-v1_2-Small (OlmoEarth Artifact License, see ../LICENSE-OlmoEarth.txt; no military /
no extractive use, carried downstream). Metrics (test 2021, 3 seeds run, this is seed 0): AUC-PR 0.5417
vs persistence 0.2731, ECE 0.0054, growth-region AUC-PR 0.2380.
Full metrics + reliability diagram: `fire-spread-forecast-v1-small_seed0.json`.

Loading (the file holds only the trainable parameters; the frozen encoder comes from Hugging Face):
```python
from fsf.olmo import load_encoder, S2Norm, OlmoUNet, OlmoUNetE2E
import torch
enc, D = load_encoder("OLMOEARTH_V1_2_SMALL", "cuda"); ck = torch.load("checkpoints/fire-spread-forecast-v1-small_seed0.pt")
model = OlmoUNetE2E(OlmoUNet(40, D, 16, 32, 3, 0.1), enc, S2Norm("cuda"), unfreeze_blocks=2).cuda()
model.load_state_dict(ck["state_dict"], strict=False)   # x (B,40,128,128) WFTS channels, s2 (B,4,12,128,128) DN, months, years
```
