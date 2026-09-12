# Demo: 2021/fire_25547988 (near Redding, CA), WildfireSpreadTS test split

Real fire geometry (VIIRS active fire), real HIFLD transmission lines and substations (voltages, owners), real model output.
Model: `wfts_only` baseline checkpoint (`results/detail/wfts_only_ckpt_s0_a95917.pt`, test AUC-PR 0.543 vs persistence 0.273).
Swap in the OlmoEarth post-trained checkpoint with `python -m fsf.predict CKPT FIRE_DIR` and rerun `viz.demo_map`.

- `demo_map.html`: day 8 = 2021-09-24 (largest active-fire day, 202 px), forecast made from 2021-09-23. Layers: perimeter, 24 h burn probability, HIFLD assets coloured by exposure, ranked pins. Fuel and wind layers are still placeholders.
- `exposure_ranked.txt/.csv`: the exposure queue for the 2021-09-24 forecast. Dollar values and vulnerability coefficients are the flagged assumptions in `viz/exposure.py`, not customer data.
- `replay.gif`: forecast (left) vs observed (right), 17 daily frames 2021-09-17 .. 2021-10-03.
- `forecast_dates.json`: date layout of `probs.npy`.

Model field sanity (from `fsf.predict`): on active days mean P on burned pixels 0.43-0.70 vs 0.003-0.007 on unburned; a re-ignition on 2021-09-23 (296 px) was missed (mean P 0.008).
