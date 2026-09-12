"""Asset exposure engine: P(burn in 24 h) raster + a customer's asset layer -> ranked queue of what is at risk and what it is worth.

  python -m viz.exposure --demo [--out out/exposure/demo]                      # synthetic fire + synthetic assets
  python -m viz.exposure --raster p.tif --lines lines.geojson --substations subs.geojson --stands stands.geojson
  python -m viz.exposure --probs p.npy --bbox W S E N --fetch-hifld            # raster on a WGS84 bbox, assets from HIFLD
  options: --threshold 0.4  --segment-m 1000  --buffer-m 200  --out DIR

Raster: a GeoTIFF in the fire event's CRS (WFTS events are 375 m UTM grids), or a .npy on a WGS84 bbox. Assets: GeoJSON
(or a shapefile if fiona is installed), assumed WGS84 unless the file says otherwise; reprojected to the raster CRS and
clipped to its bounds. Lengths, buffers and areas are computed in a metric CRS (the raster's if projected, else the local
UTM zone).

Circuits are never collapsed to a mean: each line is cut into ~1 km segments (shapely.ops.substring), each segment is
buffered 200 m, zonal-stat'd (mean, max, count) and the circuit reports exposed miles + worst segment. Stands report
acres above threshold + salvage window. exposure = P(burn) x value x vulnerability x quantity, summed per segment/pixel.

Outputs in --out: ranked.txt (the table), ranked.csv, and WGS84 GeoJSONs with per-segment / per-asset exposure attached:
segments.geojson, substations.geojson, stands.geojson (for the map layer).
"""
import argparse, csv, json, math, os, sys, numpy as np
import shapely
from shapely.geometry import shape, mapping, box, LineString, MultiLineString, Point
from shapely.ops import substring, transform as shp_transform
from pyproj import CRS, Transformer

# ---------------------------------------------------------------- assumptions (a real pilot gets these from the customer)

VALUE = {                          # replacement value
    "transmission_500kv": 2_500_000,   # per mile
    "transmission_230kv": 1_200_000,
    "transmission_115kv":   600_000,
    "transmission_69kv":    300_000,   # sub-transmission / distribution feeder
    "substation":        18_000_000,   # each
    "timber_stand":          12_000,   # per acre, merchantable (overridden per species below)
}
VULNERABILITY = {                  # P(loss | in burn area)
    "wood_pole":     0.9,
    "steel_lattice": 0.3,
    "substation":    0.6,
    "timber_mature": 0.85,
    "timber_young":  0.2,
}
STRUCTURE_BY_VOLTAGE = lambda kv: "steel_lattice" if kv >= 230 else "wood_pole"     # HIFLD has no structure field; assumed
TIMBER = {                         # merchantable value per acre and salvage window (months of usable value after a burn)
    "douglas_fir":    {"value_per_acre": 14_000, "salvage_months": 12},
    "ponderosa_pine": {"value_per_acre":  9_000, "salvage_months": 9},
    "lodgepole_pine": {"value_per_acre":  5_000, "salvage_months": 18},
    "western_hemlock":{"value_per_acre": 11_000, "salvage_months": 6},
}
MATURE_AGE_YEARS = 40
THRESHOLD, SEGMENT_M, BUFFER_M = 0.40, 1000.0, 200.0
M_PER_MILE, M2_PER_ACRE = 1609.344, 4046.856

HIFLD = {   # ArcGIS REST query endpoints, verified 2026-09-12. Lines: official HIFLD org. Substations: the public HIFLD mirror
    # (services6/OO2s4OoyCZkYJ6oE, ORNL-sourced); the HIFLD org's own substations layer was not found under an obvious name.
    "lines": "https://services1.arcgis.com/Hp6G80Pky0om7QvQ/arcgis/rest/services/Electric_Power_Transmission_Lines/FeatureServer/0/query",
    "substations": "https://services6.arcgis.com/OO2s4OoyCZkYJ6oE/arcgis/rest/services/Substations/FeatureServer/0/query",
}

# ---------------------------------------------------------------- raster

class Raster:
    """North-up probability raster: prob (H, W), origin (x0, y0) of the top-left corner, pixel size (dx, dy>0), pyproj CRS."""
    def __init__(self, prob, x0, y0, dx, dy, crs):
        self.prob = np.asarray(prob, np.float32); self.H, self.W = self.prob.shape
        self.x0, self.y0, self.dx, self.dy, self.crs = float(x0), float(y0), float(dx), float(dy), CRS.from_user_input(crs)
        self.metric = self.crs if self.crs.is_projected else utm_for(*self.center_lonlat())
        self.to_metric = Transformer.from_crs(self.crs, self.metric, always_xy=True) if self.crs != self.metric else None
        self.to_wgs = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        self.from_wgs = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        # pixel area in m2: exact for projected; at the centre latitude for geographic grids
        if self.crs.is_projected: self.px_area_m2 = self.dx * self.dy
        else:
            lat = self.center_lonlat()[1]; self.px_area_m2 = (self.dx * 111_320 * math.cos(math.radians(lat))) * (self.dy * 111_320)

    @classmethod
    def from_geotiff(cls, path, band=1, prob=None):
        import rasterio
        with rasterio.open(path) as ds:
            t = ds.transform; assert t.b == 0 and t.d == 0, "rotated rasters not supported"
            data = ds.read(band).astype(np.float32) if prob is None else prob
            return cls(data, t.c, t.f, t.a, -t.e, ds.crs.to_wkt())

    @classmethod
    def from_bbox(cls, prob, bbox_wgs84):
        w, s, e, n = bbox_wgs84; H, W = np.asarray(prob).shape
        return cls(prob, w, n, (e - w) / W, (n - s) / H, "EPSG:4326")

    @property
    def bounds(self): return (self.x0, self.y0 - self.H * self.dy, self.x0 + self.W * self.dx, self.y0)
    def center_lonlat(self):
        cx, cy = self.x0 + self.W * self.dx / 2, self.y0 - self.H * self.dy / 2
        return Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True).transform(cx, cy)
    def bounds_wgs84(self):
        xs, ys = self.to_wgs.transform([self.bounds[0], self.bounds[2], self.bounds[0], self.bounds[2]], [self.bounds[1], self.bounds[1], self.bounds[3], self.bounds[3]])
        return (min(xs), min(ys), max(xs), max(ys))

    def pixels_in(self, geom):
        """Probabilities of the pixels whose centres fall inside geom (geom in raster CRS)."""
        gx0, gy0, gx1, gy1 = geom.bounds
        c0, c1 = max(int((gx0 - self.x0) / self.dx), 0), min(int((gx1 - self.x0) / self.dx) + 1, self.W)
        r0, r1 = max(int((self.y0 - gy1) / self.dy), 0), min(int((self.y0 - gy0) / self.dy) + 1, self.H)
        if c1 <= c0 or r1 <= r0: return np.zeros(0, np.float32)
        cols, rows = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
        xs, ys = self.x0 + (cols + 0.5) * self.dx, self.y0 - (rows + 0.5) * self.dy
        inside = shapely.contains_xy(geom, xs.ravel(), ys.ravel())
        return self.prob[rows.ravel()[inside], cols.ravel()[inside]]

    def zonal(self, geom):
        p = self.pixels_in(geom)
        return {"mean": float(p.mean()) if p.size else 0.0, "max": float(p.max()) if p.size else 0.0, "count": int(p.size)}


def utm_for(lon, lat):
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1)


def reproject(geom, transformer):
    return shp_transform(transformer.transform, geom) if transformer else geom

# ---------------------------------------------------------------- assets in

def load_features(path):
    """GeoJSON (json) or shapefile (fiona) -> list of features. Returns (features, crs or None)."""
    if path.lower().endswith((".geojson", ".json")):
        fc = json.load(open(path)); crs = (fc.get("crs") or {}).get("properties", {}).get("name")
        return fc["features"], crs
    import fiona
    with fiona.open(path) as src: return [dict(f) for f in src], src.crs_wkt or None


def to_raster_crs(features, src_crs, raster):
    """Reproject features to the raster CRS and clip to its bounds; drops features that fall outside."""
    tr = Transformer.from_crs(CRS.from_user_input(src_crs or "EPSG:4326"), raster.crs, always_xy=True)
    clip = box(*raster.bounds); out = []
    for f in features:
        g = reproject(shape(f["geometry"]), tr)
        if not g.intersects(clip): continue
        g = g.intersection(clip)
        if g.is_empty: continue
        out.append({"properties": dict(f.get("properties") or {}), "geom": g})
    return out


def fetch_hifld(layer, bbox_wgs84, page=2000):
    """All features of a HIFLD layer intersecting a WGS84 bbox, paged through resultOffset. Returns GeoJSON features."""
    import urllib.request, urllib.parse
    feats, off = [], 0
    while True:
        q = {"where": "1=1", "geometry": ",".join(f"{b:.6f}" for b in bbox_wgs84), "geometryType": "esriGeometryEnvelope", "inSR": "4326",
             "spatialRel": "esriSpatialRelIntersects", "outFields": "*", "outSR": "4326", "f": "geojson", "resultOffset": off, "resultRecordCount": page}
        with urllib.request.urlopen(HIFLD[layer] + "?" + urllib.parse.urlencode(q), timeout=120) as r: fc = json.load(r)
        feats += fc.get("features", [])
        if not fc.get("properties", {}).get("exceededTransferLimit") or not fc.get("features"): return feats
        off += page

# ---------------------------------------------------------------- assessment

def voltage_class(kv):
    kv = kv or 0
    return "transmission_500kv" if kv >= 345 else "transmission_230kv" if kv >= 200 else "transmission_115kv" if kv >= 100 else "transmission_69kv"


def _prop(p, *keys, default=None):
    for k in keys:
        for kk in (k, k.lower(), k.upper()):
            if kk in p and p[kk] not in (None, "", -999999, "NOT AVAILABLE"): return p[kk]
    return default


def line_segments(geom_m, seg_len):
    parts = list(geom_m.geoms) if isinstance(geom_m, MultiLineString) else [geom_m]
    for part in parts:
        n = max(1, int(math.ceil(part.length / seg_len)))
        for i in range(n): yield substring(part, i * part.length / n, (i + 1) * part.length / n)


def assess_lines(assets, raster, threshold=THRESHOLD, seg_len=SEGMENT_M, buffer_m=BUFFER_M):
    """assets: [{properties, geom (raster CRS)}] -> (circuit rows, segment features in raster CRS)."""
    to_m, from_m = raster.to_metric, (Transformer.from_crs(raster.metric, raster.crs, always_xy=True) if raster.to_metric else None)
    rows, segs = [], []
    for i, a in enumerate(assets):
        p = a["properties"]; kv = float(_prop(p, "VOLTAGE", "voltage", "kv", default=0) or 0); vclass = voltage_class(kv)
        name = _prop(p, "NAME", "name") or f"{_prop(p, 'OWNER', 'owner', default='line')} {kv:.0f} kV #{_prop(p, 'ID', 'id', default=i)}"
        struct = STRUCTURE_BY_VOLTAGE(kv); vuln, val_mi = VULNERABILITY[struct], VALUE[vclass]
        gm = reproject(a["geom"], to_m); miles_tot = miles_above = expo = worst = 0.0; k = 0
        for seg in line_segments(gm, seg_len):
            miles = seg.length / M_PER_MILE
            z = raster.zonal(reproject(seg.buffer(buffer_m, cap_style="flat"), from_m))
            e = z["mean"] * val_mi * vuln * miles
            miles_tot += miles; expo += e; worst = max(worst, z["max"])
            if z["mean"] >= threshold: miles_above += miles
            segs.append({"geom": reproject(seg, from_m), "properties": {"circuit": name, "voltage_kv": kv, "segment": k, "miles": round(miles, 3),
                                                                        "p_mean": round(z["mean"], 4), "p_max": round(z["max"], 4), "n_px": z["count"],
                                                                        "exposure_usd": round(e), "structure": struct, "kind": "line"}}); k += 1
        rows.append({"name": name, "kind": "line", "voltage_kv": kv, "quantity": f"{miles_above:.1f} of {miles_tot:.0f} mi above {threshold:.0%}",
                     "worst": worst, "exposure_usd": expo, "miles_total": miles_tot, "miles_above": miles_above, "unit": "mi"})
    return rows, segs


def assess_substations(assets, raster, threshold=THRESHOLD, buffer_m=BUFFER_M):
    to_m, from_m = raster.to_metric, (Transformer.from_crs(raster.metric, raster.crs, always_xy=True) if raster.to_metric else None)
    rows, feats = [], []
    for i, a in enumerate(assets):
        p = a["properties"]; kv = float(_prop(p, "MAX_VOLT", "VOLTAGE", "voltage", default=0) or 0)
        name = _prop(p, "NAME", "name") or f"substation #{_prop(p, 'ID', 'id', default=i)}"
        z = raster.zonal(reproject(reproject(a["geom"], to_m).buffer(buffer_m), from_m))
        e = z["max"] * VALUE["substation"] * VULNERABILITY["substation"]
        feats.append({"geom": a["geom"], "properties": {"name": name, "voltage_kv": kv, "p_mean": round(z["mean"], 4), "p_max": round(z["max"], 4), "n_px": z["count"],
                                                        "exposure_usd": round(e), "kind": "substation"}})
        rows.append({"name": name, "kind": "substation", "voltage_kv": kv, "quantity": f"{kv:.0f} kV substation, P {z['max']:.0%}", "worst": z["max"],
                     "exposure_usd": e, "unit": "site"})
    return rows, feats


def stand_attrs(p, i):
    """Species / age from the parcel's properties if present, else synthetic but deterministic per parcel (labelled as such)."""
    species = _prop(p, "species", "SPECIES"); age = _prop(p, "age", "AGE", "stand_age")
    synthetic = species is None or age is None
    keys = list(TIMBER)
    species = species if species in TIMBER else keys[i % len(keys)]
    age = float(age) if age is not None else float(15 + (i * 37) % 90)
    return species, age, synthetic


def assess_stands(assets, raster, threshold=THRESHOLD):
    rows, feats = [], []
    for i, a in enumerate(assets):
        p = a["properties"]; species, age, synth = stand_attrs(p, i)
        name = _prop(p, "NAME", "name", "STAND", "UNIT") or f"Stand {i + 1:02d}-{'NSEW'[i % 4]}"
        vk = "timber_mature" if age >= MATURE_AGE_YEARS else "timber_young"; vuln, val_ac = VULNERABILITY[vk], TIMBER[species]["value_per_acre"]
        px = raster.pixels_in(a["geom"]); ac_px = raster.px_area_m2 / M2_PER_ACRE
        acres = reproject(a["geom"], raster.to_metric).area / M2_PER_ACRE if raster.to_metric else a["geom"].area / M2_PER_ACRE
        acres_above = float((px >= threshold).sum()) * ac_px; e = float((px * val_ac * vuln * ac_px).sum())
        worst = float(px.max()) if px.size else 0.0; salvage = TIMBER[species]["salvage_months"]
        feats.append({"geom": a["geom"], "properties": {"name": name, "species": species, "age_years": age, "acres": round(acres), "acres_above": round(acres_above),
                                                        "p_mean": round(float(px.mean()) if px.size else 0.0, 4), "p_max": round(worst, 4), "n_px": int(px.size),
                                                        "exposure_usd": round(e), "salvage_months": salvage, "values_synthetic": synth, "kind": "stand"}})
        rows.append({"name": name, "kind": "stand", "species": species, "age_years": age, "quantity": f"{acres_above:,.0f} of {acres:,.0f} ac above {threshold:.0%}",
                     "worst": worst, "exposure_usd": e, "acres": acres, "acres_above": acres_above, "salvage_months": salvage, "unit": "ac"})
    return rows, feats


def rank(rows):
    rows = sorted(rows, key=lambda r: (-r["exposure_usd"], -r["worst"]))
    for i, r in enumerate(rows): r["rank"] = i + 1
    return rows

# ---------------------------------------------------------------- out

def usd(x):
    return f"${x / 1e6:.1f}M" if x >= 1e6 else f"${x / 1e3:.0f}k" if x >= 1e3 else f"${x:.0f}"


def format_table(rows):
    lines = []
    for r in rows:
        tail = f"worst segment {r['worst']:.0%}" if r["kind"] == "line" else f"salvage window {r['salvage_months']} mo" if r["kind"] == "stand" else f"P {r['worst']:.0%}"
        lines.append(f"{r['rank']:>2}. {r['name']:<28} {r['quantity']:<34} {tail:<24} {usd(r['exposure_usd']):>8}")
    return "\n".join(lines)


def to_geojson(feats, raster):
    out = []
    for f in feats:
        out.append({"type": "Feature", "properties": f["properties"], "geometry": mapping(reproject(f["geom"], raster.to_wgs))})
    return {"type": "FeatureCollection", "features": out}


def write_outputs(out_dir, rows, raster, segs=(), subs=(), stands=()):
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "ranked.txt"), "w").write(format_table(rows) + "\n")
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(out_dir, "ranked.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); [w.writerow({k: r.get(k, "") for k in keys}) for r in rows]
    for name, feats in (("segments", segs), ("substations", subs), ("stands", stands)):
        if feats: json.dump(to_geojson(feats, raster), open(os.path.join(out_dir, f"{name}.geojson"), "w"))


def assess(raster, lines=None, substations=None, stands=None, src_crs=None, threshold=THRESHOLD, seg_len=SEGMENT_M, buffer_m=BUFFER_M):
    """Run everything that was supplied. Returns dict(rows=ranked, segs=..., subs=..., stands=...). Features in the raster CRS."""
    out = {"segs": [], "subs": [], "stands": []}; rows = []
    if lines:
        r, out["segs"] = assess_lines(to_raster_crs(lines, src_crs, raster), raster, threshold, seg_len, buffer_m); rows += r
    if substations:
        r, out["subs"] = assess_substations(to_raster_crs(substations, src_crs, raster), raster, threshold, buffer_m); rows += r
    if stands:
        r, out["stands"] = assess_stands(to_raster_crs(stands, src_crs, raster), raster, threshold); rows += r
    out["rows"] = rank(rows); return out

# ---------------------------------------------------------------- placeholder territory

def synthetic_assets(bbox, seed=0):
    """Placeholder asset layer for a territory bbox (WGS84): four circuits of different voltage, substations on them, eight stands.
    Geometry is hand-laid so a fire in the territory hits some assets hard, some lightly and some not at all."""
    w, s, e, n = bbox; W, Hh = e - w, n - s; rng = np.random.default_rng(seed)
    def wobble(pts, amp=0.004): return [[x + rng.normal(0, amp), y + rng.normal(0, amp)] for x, y in pts]
    circuits = [("Cascade–Bend 500 kV", 500, [(w, s + 0.86 * Hh), (w + 0.5 * W, s + 0.80 * Hh), (e, s + 0.88 * Hh)]),
                ("Santiam 230 kV", 230, [(w, s + 0.55 * Hh), (w + 0.4 * W, s + 0.50 * Hh), (w + 0.7 * W, s + 0.58 * Hh), (e, s + 0.52 * Hh)]),
                ("Redding 115 kV", 115, [(w + 0.15 * W, s), (w + 0.35 * W, s + 0.35 * Hh), (w + 0.55 * W, s + 0.62 * Hh), (w + 0.62 * W, n)]),
                ("Foothills 69 kV feeder", 69, [(w, s + 0.18 * Hh), (w + 0.5 * W, s + 0.14 * Hh), (e, s + 0.20 * Hh)])]
    lines, subs = [], []
    for name, kv, pts in circuits:
        dense = []
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            dense += [[x0 + (x1 - x0) * t, y0 + (y1 - y0) * t] for t in np.linspace(0, 1, 8)[:-1]]
        dense.append(list(pts[-1])); dense = wobble(dense, 0.002)
        lines.append({"type": "Feature", "properties": {"NAME": name, "VOLTAGE": kv, "OWNER": "placeholder", "ID": str(kv)}, "geometry": {"type": "LineString", "coordinates": dense}})
        for j in (2, len(dense) // 2, len(dense) - 3):
            subs.append({"type": "Feature", "properties": {"NAME": f"{name.split()[0]} sub {j}", "MAX_VOLT": kv, "STATUS": "IN SERVICE"}, "geometry": {"type": "Point", "coordinates": dense[j]}})
    stands, species = [], list(TIMBER)
    for i, (fx, fy, fw, fh) in enumerate([(0.05, 0.60, 0.16, 0.18), (0.24, 0.66, 0.14, 0.14), (0.42, 0.30, 0.15, 0.20), (0.60, 0.62, 0.18, 0.16),
                                          (0.72, 0.30, 0.16, 0.14), (0.30, 0.08, 0.20, 0.14), (0.80, 0.78, 0.14, 0.14), (0.55, 0.05, 0.16, 0.12)]):
        x0, y0 = w + fx * W, s + fy * Hh; ring = [[x0, y0], [x0 + fw * W, y0], [x0 + fw * W, y0 + fh * Hh], [x0, y0 + fh * Hh], [x0, y0]]
        stands.append({"type": "Feature", "properties": {"NAME": f"Stand {70 + 3 * i}-{'NSEW'[i % 4]}", "species": species[i % 4], "age": int(12 + (i * 29) % 80), "owner": "placeholder parcel"},
                       "geometry": {"type": "Polygon", "coordinates": [ring]}})
    return lines, subs, stands


def demo_raster(bbox=None, day=4):
    from viz.map import synthetic_fire, DEMO_BBOX
    from viz.replay import placeholder_probs
    masks, dates, _ = synthetic_fire(T=8); return Raster.from_bbox(placeholder_probs(masks)[day], bbox or DEMO_BBOX), masks[day]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raster", help="GeoTIFF of P(burn 24h) in the event CRS"); ap.add_argument("--probs", help=".npy (H,W) on --bbox")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    ap.add_argument("--lines"); ap.add_argument("--substations"); ap.add_argument("--stands"); ap.add_argument("--fetch-hifld", action="store_true")
    ap.add_argument("--threshold", type=float, default=THRESHOLD); ap.add_argument("--segment-m", type=float, default=SEGMENT_M); ap.add_argument("--buffer-m", type=float, default=BUFFER_M)
    ap.add_argument("--demo", action="store_true"); ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.demo:
        raster, _ = demo_raster(); lines, subs, stands = synthetic_assets(raster.bounds_wgs84()); src = "EPSG:4326"
        print("demo: synthetic fire, synthetic assets (geometry and values are placeholders)")
    else:
        if a.raster: raster = Raster.from_geotiff(a.raster)
        elif a.probs and a.bbox: raster = Raster.from_bbox(np.load(a.probs), a.bbox)
        else: sys.exit("need --raster, or --probs with --bbox, or --demo")
        lines = subs = stands = None; src = None
        if a.fetch_hifld:
            bb = raster.bounds_wgs84(); lines, subs = fetch_hifld("lines", bb), fetch_hifld("substations", bb); src = "EPSG:4326"
            print(f"HIFLD: {len(lines)} lines, {len(subs)} substations intersect {tuple(round(b, 3) for b in bb)}")
        if a.lines: lines, src = load_features(a.lines)
        if a.substations: subs, src = load_features(a.substations)
        if a.stands: stands, src = load_features(a.stands)
    res = assess(raster, lines, subs, stands, src, a.threshold, a.segment_m, a.buffer_m)
    print(format_table(res["rows"]))
    out = a.out or os.path.join("out", "exposure", "demo" if a.demo else "run")
    write_outputs(out, res["rows"], raster, res["segs"], res["subs"], res["stands"]); print("wrote", out)


if __name__ == "__main__":
    main()
