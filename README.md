<p align="center">
  <img src="assets/logo.svg" alt="Fireline — wildfire spread forecasting" width="440">
</p>

Fireline predicts, per pixel, where an active wildfire will burn in the next 24 hours as a calibrated probability map rather than a single deterministic perimeter.

**Live demo map:** https://eitanlebras.github.io/fire-spread-forecast/

<p align="center">
  <img src="assets/fire_25294746_day12.png" alt="Fire 25294746, day 12: model forecast (blue) vs. observed next-day fire (orange)" width="720">
</p>

<p align="center"><sub>Fire 25294746, day 12 (2021-07-20). Grey is burning today, blue is the model's probability of burning in the next 24 hours, orange outlines the fire actually observed the next day.</sub></p>

## What's here

- `fsf/` — model, data pipeline, and training code. Fuses the Next Day Wildfire Spread benchmark (weather, terrain, fuel moisture, previous fire mask) with Sentinel-2 composites encoded by Ai2's OlmoEarth foundation model, and post-trains the last encoder blocks jointly with a UNet decoder.
- `configs/` — training arms (`base`, `derived`, `raw_th`).
- `scripts/` — RunPod bootstrap and arm launch scripts.
- `viz/` — demo map: fire perimeter, 24h probability raster, HRRR wind, terrain, HIFLD transmission lines and substations, and ranked asset-exposure pins.

Built at the Frontier Cascadia hackathon.
