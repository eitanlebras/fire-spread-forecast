# Demo candidate: 2021/fire_25410845 (central Washington, near Lake Chelan / Entiat), WildfireSpreadTS test split, 41 days

Best forecast day of any test fire we swept. Today = day 30 = 2021-09-06 (189 active px), forecast for 2021-09-07 (72 new px):
AUC-PR 0.625 on the growth region, 72% of the new fire inside the >40% zone, model direction 4° from the observed one.
Day 11 (2021-08-18 -> 08-19) is nearly as good: AUC-PR 0.649, overlap 0.71, 15°.

Caveat for the utility story: on this day the fire is in wilderness. Nothing of value sits in the >40% zone within 10 km
(top pin: a $1k tap); the nearest real substation, Western (115 kV), is at P 9%. Use Redding (`demo/redding`) for the
exposure story, this fire for the "the model knows where it goes" replay.

Files: `demo_map.html` (same layers as Redding), `exposure_ranked.*` (09-07 forecast, unbuffered raster), `replay.gif` (all 40 forecast days, forecast panel masked to the growth region so nothing is shown inside yesterday's fire,
panels cropped to today's fire + new fire with 20% margin, water drawn light blue and verified P=0 on it, 6 s hold on the 2021-09-07 frame
with growth-region AUC-PR 0.625, 72% overlap and 2.1 km max reach in the header; every frame's stats line shows growth-region AUC-PR against persistence's floor, never the full-mask number), `replay_hold_day30_forecast_2021-09-07.png` (that frame),
`forecast_dates.json`. Model: `wfts_only` baseline checkpoint; swap via `fsf.predict`.

Note: the baseline never predicts detached spot fires (0 hits on detached new-fire components across every day of both fires, see demo/frames_growth); do not frame any blue lobe as a spot-fire call.
