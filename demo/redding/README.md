# Demo: 2021/fire_25547988 (near Redding, CA), WildfireSpreadTS test split

Real fire geometry (VIIRS active fire), real HIFLD transmission lines and substations (voltages, owners), real model output.
Model: `wfts_only` baseline checkpoint (`results/detail/wfts_only_ckpt_s0_a95917.pt`, test AUC-PR 0.543 vs persistence 0.273).
Swap in the OlmoEarth post-trained checkpoint with `python -m fsf.predict CKPT FIRE_DIR` and rerun `viz.demo_map`.

- `demo_map.html`: today = day 7 = 2021-09-23 (296 active px). Layers: dark grey = burning today (VIIRS); blue = the model's P(burn in 24 h) on the growth region only (pixels not burning today), shown within a feathered 10 km buffer of today's fire, bilinearly upsampled; orange = new fire actually observed on 2021-09-24 (71 px). HIFLD assets coloured by exposure, ranked pins. Fuel and wind layers are still placeholders.
  Direction check for this day (P mass within 6 km of the fire): model 334°, observed new fire 327°, wind blowing toward 303°. Mean P on the pixels that burned next day 0.48 vs 0.07 on other pixels within 6 km.
- `exposure_ranked.txt/.csv`: the exposure queue for the 2021-09-24 forecast. Dollar values and vulnerability coefficients are the flagged assumptions in `viz/exposure.py`, not customer data.
- `replay.gif`: forecast (left) vs observed (right), 17 daily frames 2021-09-17 .. 2021-10-03.
- `forecast_dates.json`: date layout of `probs.npy`.

Model field sanity (from `fsf.predict`): on active days mean P on burned pixels 0.43-0.70 vs 0.003-0.007 on unburned; a re-ignition on 2021-09-23 (296 px) was missed (mean P 0.008).

Day sweep (`python -m fsf.day_sweep`), days with >= 30 new-fire px: day 7 (09-23 -> 09-24) AUC-PR 0.318, overlap 0.65, dir 7°;
day 8 (09-24 -> 09-25) 0.246, 0.71, 26°. Day 7 is the map above. The other candidate test fires: see `demo/chelan`;
fire_25086466 (ND) and fire_25411896 (MT) have no forecastable growth day (sporadic <= 41 px detections, overlap 0).
