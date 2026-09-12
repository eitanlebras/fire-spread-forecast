# Interior fill vs true advance (fsf.split_sweep), baseline wfts_only checkpoint, both viable test fires

Growth region (~today) split against yesterday's DRAWN perimeter (the smoothed outline the replay shows, holes filled):
interior = unburned pixels inside it (gaps between VIIRS detections), advance = pixels beyond it. Full table: split_sweep.txt.

Pooled over all days (pixels concatenated):

| subregion | new-fire px | pixels | AUC-PR | overlap (>0.4) | prevalence | lift over prevalence |
|---|---|---|---|---|---|---|
| interior gaps | 693 | 2,856 | 0.432 | 0.77 | 0.243 | 1.8x |
| true advance | 1,397 | 2,646,092 | 0.022 | 0.06 | 0.0005 | 41x |

Read: the model is very good at filling gaps inside the perimeter (77% of that new fire lands in the >0.4 zone) and weak at
advance: only 6% of the fire that crossed the boundary was in the >0.4 zone. The strict split (fill_holes of the raw mask)
puts almost everything in "advance" (AUC-PR 0.161, overlap 0.29) because the speckled mask encloses nothing; that number
mixes near-perimeter growth with real advance and is the one the earlier "growth-region AUC-PR" figures reported.

Best advance frames (advance new px >= 30, by advance AUC-PR): Chelan day 9 (2021-08-16 -> 08-17): 0.483 but 0% overlap
(ranking only, no pixel above 0.4); day 28: 0.196 / 9%; day 21: 0.149 / 4%; Redding day 8: 0.077 / 52% (31 px).
Day 30 (the held replay frame): interior 54 px AUC-PR 0.764 vs advance 18 px AUC-PR 0.102, 11% overlap: an interior-fill win.

There is no frame where the baseline wins on advance in the overlap sense. Rendered here: days 9, 28, 30 with the split in the header.
