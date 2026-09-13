# Per-day frames, headless (viz.frames), water-masked predictions (fsf.predict zeroes P on land-cover class 17)

One PNG per today `t` for the two viable test fires. Title carries: day, dates, burning-today px, new-fire px, AUC-PR of
the forecast on the growth region, share of the observed new fire inside the >0.4 zone, and the direction error between
the model's P-mass (within 6 km) and the observed new fire. Layers: hillshade (elevation band), grey-green = water,
dark grey = burning today, blue = model P(burn in 24 h) on the growth region within the feathered 10 km buffer,
orange = new fire observed the next day.

Shortlist (new fire >= 30 px):
- fire_25547988 (Redding): day07 AUC-PR 0.318 / overlap 0.65 / 7°; day08 0.246 / 0.71 / 26°.
- fire_25410845 (Lake Chelan area): day30 0.625 / 0.72 / 4°; day11 0.649 / 0.71 / 15°; day26 0.503 / 0.68 / 28°; day27 0.580 / 0.58 / 64°; day31 0.400 / 0.89 / 68°.
The raster of fire_25547988 also contains a second, separate fire in its NW corner; its pixels count in the new-fire totals.

Outside-perimeter share (`day_sweep.txt` per fire, column `outside`, `*` = >=50% of new fire beyond yesterday's smoothed perimeter):
the model's strongest days are mostly infill days (outside 0.25-0.45). On the flagged directional days it is weak:
Chelan day09 (outside 0.67) AUC-PR 0.553 but overlap 0.26 and direction error 138°; day28 (0.63) overlap 0.26, 136°;
day21 (0.84) overlap 0.19, 59°; Redding day08 (0.50) overlap 0.71, 26° is the one borderline directional success.
Days with outside = 1.00 and AUC-PR ~0 are new ignitions / spotting far from the fire, which the model cannot see.
