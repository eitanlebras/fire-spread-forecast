# Interior fill vs true advance over the whole WFTS test split (30 fires, 599 fire-days with >= 5 new px)

Baseline `wfts_only` checkpoint, predictions for every 2021 fire (`fsf.predict`), split with `fsf.split_sweep` against
yesterday's drawn perimeter (smoothed outline, holes filled). Full output: `split_test.txt`; ranked frames: `split_test_frames.json`.

Pooled (pixels concatenated across all fire-days):

| subregion | new-fire px | AUC-PR | overlap in >0.4 zone | prevalence | lift |
|---|---|---|---|---|---|
| interior gaps | 24,351 | 0.451 | 0.83 | 0.240 | 1.9x |
| true advance | 51,478 | 0.065 | 0.19 | 0.0012 | 54x |
| advance, strict split (fill_holes of raw mask) | 75,429 | 0.218 | 0.39 | 0.0018 | 123x |

On the two fires used earlier (Redding, Chelan) advance overlap was 6%; over the test set it is 19%, and on the large
long-running fires (30-44 days) 20-30%. Those two were among the weakest advance fires in the set.

Best advance frames (advance new px >= 30):

| rank | fire (centre) | day | today -> for | advance px | advance AUC-PR | advance overlap | interior px / AUC-PR |
|---|---|---|---|---|---|---|---|
| 1 | 25295941 (45.64, -113.77) SW Montana | 8 | 07-12 -> 07-13 | 86 | 0.605 | 0.85 | 72 / 0.836 |
| 2 | 25294746 (38.72, -119.73) Sierra, CA/NV | 12 | 07-20 -> 07-21 | 160 | 0.541 | 0.68 | 162 / 0.755 |
| 3 | 25295023 (45.87, -117.59) NE Oregon | 1 | 07-10 -> 07-11 | 203 | 0.429 | 0.89 | 123 / 0.709 |
| 4 | 25410444 (41.56, -121.88) N California | 6 | 08-05 -> 08-06 | 297 | 0.567 | 0.48 | 103 / 0.727 |
| 5 | 25295831 (48.36, -118.49) NE Washington | 4 | 07-13 -> 07-14 | 253 | 0.367 | 0.70 | 41 / 0.505 |
| 6 | 25410444 (41.56, -121.88) N California | 13 | 08-12 -> 08-13 | 239 | 0.469 | 0.51 | 68 / 0.864 |

Rank 1 is the demo frame for outward prediction: the next-day fire is an 86-px band beyond the perimeter on the E/SE flank
and the model's P is darkest on that flank, light on the W. Rank 3 has the largest advance with 89% overlap.
