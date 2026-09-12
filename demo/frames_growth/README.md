# Outward-spread frames (fsf.growth_sweep), headers carry growth-region numbers only

Ranked by outside_frac x growth-region AUC-PR over both viable test fires, n_new >= 50, zone = P > 0.4. Top 8:

| fire | day | today | burning | new | outside | growth AUC-PR | overlap | max reach km | detached spots (hits) |
|---|---|---|---|---|---|---|---|---|---|
| 25410845 | 11 | 2021-08-18 | 154 | 63 | 1.00 | 0.649 | 0.71 | 1.7 | 0 |
| 25410845 | 30 | 2021-09-06 | 189 | 72 | 1.00 | 0.625 | 0.72 | 2.1 | 0 |
| 25410845 | 9 | 2021-08-16 | 171 | 139 | 0.99 | 0.553 | 0.26 | 53.2 | 1 (0 hit, 14 px, a separate ignition) |
| 25410845 | 28 | 2021-09-04 | 136 | 135 | 0.99 | 0.506 | 0.26 | 2.9 | 0 |
| 25410845 | 23 | 2021-08-30 | 136 | 62 | 1.00 | 0.474 | 0.47 | 1.5 | 0 |
| 25410845 | 31 | 2021-09-07 | 208 | 57 | 1.00 | 0.400 | 0.89 | 1.4 | 0 |
| 25547988 | 7 | 2021-09-23 | 296 | 71 | 0.99 | 0.318 | 0.65 | 1.4 | 0 |
| 25410845 | 10 | 2021-08-17 | 243 | 51 | 0.98 | 0.308 | 0.45 | 6.2 | 2 (0 hit, 13 px) |

Reading: with "outside" defined against the hole-filled boundary of today's speckled VIIRS mask, essentially all new fire is
outside (>= 0.98) on every day, so the criterion does not discriminate; the earlier "infill" reading came from a smoothed
perimeter that swallows ~1 km of edge growth. What does discriminate is reach: the model's good days are 1.4-2.9 km edge
advances; every jump beyond that (53 km on day 9, 6 km on day 10) is a detached component the model put no P > 0.4 near.
Detached-spot hits are 0 across both fires: the baseline never predicts spotting.

Day 29 blob check (today 2021-09-05 -> 09-06): the two blue spots SE of the fire are two separate P > 0.4 lobes (1 px at
P 0.47 toward 164°, 4 px at P 0.68 toward 149°), both ATTACHED to today's fire (8-connected), and both have observed new
fire within 2 px (5 and 4 px). Two verified edge-growth hits on the SE flank, not spot-fire predictions.
