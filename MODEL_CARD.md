---
base_model: allenai/OlmoEarth-v1_2-Small
license: other
license_name: olmoearth-artifact-license
license_link: LICENSE-OlmoEarth.txt
pipeline_tag: image-segmentation
tags: [wildfire, fire-spread, sentinel-2, olmoearth, wildfirespreadts]
---

# fire-spread-forecast-v1-small

Per-pixel probability that a pixel is actively burning on day t+1, given the WildfireSpreadTS observation stack on
day t and OlmoEarth features of the months before the fire. A derivative of **allenai/OlmoEarth-v1_2-Small**
(ViT-Small, 36M parameter encoder, OlmoEarth v1.2).

## Architecture

- **Base model:** allenai/OlmoEarth-v1_2-Small encoder (`ModelID.OLMOEARTH_V1_2_SMALL` in olmoearth_pretrain_minimal),
  input modality `sentinel2_l2a` only, patch size 8, tokens averaged over time and band sets.
- **Sentinel-2 input:** four monthly median composites (Earth Search sentinel-2-l2a, cloud cover < 40, SCL cloud and
  shadow masked) of the four calendar months before the fire start, on the event's own 375 m grid, 12 bands in
  OlmoEarth's `sentinel2_l2a` order: B02, B03, B04, B08, B05, B06, B07, B8A, B11, B12, B01, B09.
- **Head:** learned 1x1 convolution 384 -> 16 channels, bilinear upsampling to the pixel grid, concatenation onto the
  40 WildfireSpreadTS channels (23 bands after the authors' preprocessing with landcover one-hot, plus the binary
  active-fire mask), then a UNet (width 32, depth 3, GroupNorm, dropout 0.1).
- **Training stages:** stage 1 keeps the encoder frozen (embeddings cached in fp16); stage 2 unfreezes only the last
  two encoder blocks (patch embedding and encodings stay frozen) with AdamW, head lr 1e-3 and encoder lr 1e-5.

## Data

WildfireSpreadTS (Gerard et al. 2023, Zenodo 8006177): 128 x 128 tiles at 375 m, train 2018 + 2019, validation 2020,
test 2021 (the authors' fold 0). Target: next-day active fire. Padded and unobserved pixels are excluded from every loss
and metric.

## Evaluation protocol

Every number is reported per seed next to the persistence baseline (yesterday's fire mask as the prediction) on the
same pixels: AUC-PR, ECE with a reliability diagram, and AUC-PR restricted to the growth region (pixels not burning on
day t). Full-mask IoU is never reported. Four arms are compared: WFTS only; WFTS + frozen OlmoEarth; WFTS + post-trained
OlmoEarth; the no-OlmoEarth arm is the ablation every OlmoEarth number is read against.

## License and use restrictions

The encoder weights are derived from OlmoEarth and are released under the
[OlmoEarth Artifact License](LICENSE-OlmoEarth.txt) (Ai2). Section 2 of that license prohibits military and
defense-related applications and extractive-industry applications (oil, gas and mineral extraction), and requires that
these restrictions be carried into all downstream distribution. This model and any derivative of it inherit those
restrictions. Please cite Ai2's OlmoEarth (https://github.com/allenai/olmoearth_pretrain) when using this model.
