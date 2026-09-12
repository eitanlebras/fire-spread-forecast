"""Interactive fire map -> one standalone HTML file (folium / Leaflet). Layers, each toggleable in the layer control:

  fire perimeter (today)      polygon outline of the current active-fire mask
  24h spread probability      our model's P(fire tomorrow) as a colored raster overlay
  HRRR 10 m wind              arrow field (u, v) from a CSV: lat,lon,u,v  [m/s]
  terrain                     OpenTopoMap base layer + Esri hillshade overlay
  transmission lines          HIFLD GeoJSON, colored by max P along the line
  substations                 HIFLD GeoJSON, one pin per substation
  asset exposure (ranked)     numbered pins, ranked by P within ~1 km of each substation, popup with the numbers

  python -m viz.map                                                  # fully synthetic demo (fire, wind, grid assets)
  python -m viz.map FIRE_DIR --day 6 [--probs P.npy]                 # WFTS fire dir; bbox read from the GeoTIFF
  python -m viz.map masks.npz --bbox W S E N --probs P.npy           # masks from .npz (or key bbox_wgs84 in the npz)
  python -m viz.map FIRE_DIR --wind wind.csv --transmission lines.geojson --substations subs.geojson
  python -m viz.map FIRE_DIR --fetch-hifld                           # query HIFLD ArcGIS services for the bbox (needs network)
  --out out/map/<fire>.html  --exposure-km 1.0

Anything not supplied is synthesised and labelled "(placeholder)" in the layer name, so the page works before the model,
the HRRR pull, or the HIFLD download exist. Same day convention as viz.replay: --day k shows day k's perimeter and the
forecast for day k+1 (probs[k]). Pixel -> lat/lon is a linear map over the WGS84 bbox; fine at fire scale.
"""
import argparse, json, os, sys, base64, io, math, numpy as np
import matplotlib
matplotlib.use("Agg")
import contourpy, folium
from folium import plugins
from PIL import Image
from viz import palette as P
from viz.replay import load_fire, placeholder_probs, align_probs

HIFLD = {   # ArcGIS REST endpoints (HIFLD Open); bbox query with f=geojson. Unverified from this machine: pass GeoJSON files if they 404.
    "transmission": "https://services1.arcgis.com/Hp6G80Pky0om7QvQ/arcgis/rest/services/Electric_Power_Transmission_Lines/FeatureServer/0/query",
    "substations": "https://services1.arcgis.com/Hp6G80Pky0om7QvQ/arcgis/rest/services/Electric_Substations/FeatureServer/0/query",
}
DEMO_BBOX = (-121.95, 44.30, -121.55, 44.55)     # W S E N; central Oregon Cascades foothills, for the synthetic demo

# ---------------------------------------------------------------- geometry helpers

class Grid:
    """Linear pixel <-> lon/lat map over a WGS84 bbox for an (H, W) raster (row 0 = north)."""
    def __init__(self, bbox, H, W):
        self.w, self.s, self.e, self.n = bbox; self.H, self.W = H, W
        self.dlon, self.dlat = (self.e - self.w) / W, (self.n - self.s) / H
        self.km_per_px = 111.32 * self.dlat                                            # N-S pixel size, km
    def to_ll(self, r, c): return self.n - (r + 0.5) * self.dlat, self.w + (c + 0.5) * self.dlon
    def to_rc(self, lat, lon): return int((self.n - lat) / self.dlat), int((lon - self.w) / self.dlon)
    def inside(self, r, c): return 0 <= r < self.H and 0 <= c < self.W
    @property
    def bounds(self): return [[self.s, self.w], [self.n, self.e]]
    @property
    def center(self): return (self.s + self.n) / 2, (self.w + self.e) / 2


def fire_bbox(src, masks_file_bbox=None):
    """WGS84 bbox for a WFTS fire dir (from the first GeoTIFF) or an explicit / npz-provided bbox."""
    if masks_file_bbox is not None: return tuple(float(b) for b in masks_file_bbox)
    if os.path.isdir(src):
        import glob, rasterio
        from rasterio.warp import transform_bounds
        with rasterio.open(sorted(glob.glob(os.path.join(src, "*.tif")))[0]) as ds: return tuple(transform_bounds(ds.crs, "EPSG:4326", *ds.bounds))
    if src.endswith(".npz"):
        z = np.load(src, allow_pickle=True)
        if "bbox_wgs84" in z: return tuple(float(b) for b in z["bbox_wgs84"])
    sys.exit("no bbox: pass --bbox W S E N (or store bbox_wgs84 in the npz)")


def mask_polygons(mask, grid, smooth=1.0):
    """Perimeter polygons (lon/lat rings) of a binary mask, from a contour of the lightly smoothed mask."""
    from viz.replay import _blur
    z = _blur(mask, smooth) if smooth else mask.astype(float)
    rings = []
    for line in contourpy.contour_generator(z=z).lines(0.3):
        if len(line) < 4: continue
        rings.append([[float(grid.w + x * grid.dlon), float(grid.n - y * grid.dlat)] for x, y in line])   # contour x = col, y = row
    return rings


def prob_png(prob, alpha_max=0.85):
    """Probability raster -> RGBA PNG bytes (colormap from the palette, transparent near zero)."""
    rgba = (P.PROB_CMAP(np.clip(prob, 0, 1)) * 255).astype(np.uint8)
    rgba[..., 3] = (np.clip(prob, 0, 1) ** 0.6 * alpha_max * 255).astype(np.uint8)
    buf = io.BytesIO(); Image.fromarray(rgba).save(buf, "PNG"); return buf.getvalue()

# ---------------------------------------------------------------- placeholder data

def synthetic_fire(H=160, W=220, T=8, seed=0):
    """Same shape of fire as the replay test: a crescent front marching east, speckled like VIIRS detections."""
    rng = np.random.default_rng(seed); yy, xx = np.mgrid[:H, :W]; cy, cx = H // 2, W // 4; burned = np.zeros((H, W), bool); ms = []
    for t in range(T):
        r = 8 + 9 * t; ang = np.arctan2(yy - cy, xx - cx)
        front = ((yy - cy) ** 2 + (xx - cx - 6 * t) ** 2) < (r * (1 + 0.35 * np.cos(ang - 0.4))) ** 2
        ms.append(front & ~burned & (rng.random((H, W)) > 0.15)); burned |= front
    return np.stack(ms), [f"day {t}" for t in range(T)], "synthetic_fire"


def synthetic_wind(grid, n=9, seed=0):
    """Placeholder for HRRR: a prevailing WSW wind (8 m/s) with mild spatial variation. Rows: lat, lon, u, v."""
    rng = np.random.default_rng(seed); rows = []
    for lat in np.linspace(grid.s, grid.n, n)[1:-1]:
        for lon in np.linspace(grid.w, grid.e, n)[1:-1]:
            rows.append((float(lat), float(lon), 6.5 + rng.normal(0, 1.0), 3.0 + rng.normal(0, 1.0)))
    return rows


def synthetic_grid_assets(grid, seed=0):
    """Placeholder for HIFLD: three transmission lines crossing the bbox and substations along them (GeoJSON dicts)."""
    rng = np.random.default_rng(seed); lines, subs = [], []
    specs = [("Cascade–Bend 230 kV", 230, 0.72), ("Santiam 115 kV", 115, 0.45), ("Foothills tap 69 kV", 69, 0.28)]
    for i, (name, kv, fy) in enumerate(specs):
        lat0 = grid.s + fy * (grid.n - grid.s); pts = []
        for j, lon in enumerate(np.linspace(grid.w - 0.02, grid.e + 0.02, 7)):
            pts.append([float(lon), float(lat0 + 0.02 * math.sin(j + i) + rng.normal(0, 0.003))])
        lines.append({"type": "Feature", "properties": {"NAME": name, "VOLTAGE": kv, "OWNER": "placeholder"}, "geometry": {"type": "LineString", "coordinates": pts}})
        for j in (1, 3, 5):
            subs.append({"type": "Feature", "properties": {"NAME": f"{name.split()[0]} sub {j}", "MAX_VOLT": kv, "STATUS": "IN SERVICE"},
                         "geometry": {"type": "Point", "coordinates": pts[j]}})
    return {"type": "FeatureCollection", "features": lines}, {"type": "FeatureCollection", "features": subs}

# ---------------------------------------------------------------- real-data loaders

def load_wind_csv(path):
    """CSV with header containing lat, lon and u, v (m/s) columns; e.g. HRRR 10 m UGRD/VGRD sampled to a grid."""
    import csv
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            k = {c.lower().strip(): c for c in r}
            u = r[k.get("u") or k.get("u10") or k.get("ugrd")]; v = r[k.get("v") or k.get("v10") or k.get("vgrd")]
            rows.append((float(r[k["lat"]]), float(r[k["lon"]]), float(u), float(v)))
    return rows


def fetch_hifld(layer, bbox):
    import urllib.request, urllib.parse
    q = {"where": "1=1", "geometry": ",".join(map(str, bbox)), "geometryType": "esriGeometryEnvelope", "inSR": "4326", "spatialRel": "esriSpatialRelIntersects",
         "outFields": "*", "outSR": "4326", "f": "geojson"}
    url = HIFLD[layer] + "?" + urllib.parse.urlencode(q)
    with urllib.request.urlopen(url, timeout=60) as r: return json.load(r)

# ---------------------------------------------------------------- exposure

def sample_max(prob, grid, lon, lat, radius_px):
    r0, c0 = grid.to_rc(lat, lon)
    if not grid.inside(r0, c0): return 0.0
    win = prob[max(r0 - radius_px, 0):r0 + radius_px + 1, max(c0 - radius_px, 0):c0 + radius_px + 1]
    return float(win.max()) if win.size else 0.0


def dist_to_fire_km(mask, grid, lon, lat):
    rr, cc = np.nonzero(mask)
    if len(rr) == 0: return float("inf")
    r0, c0 = grid.to_rc(lat, lon)
    return float(np.sqrt(((rr - r0) * grid.km_per_px) ** 2 + ((cc - c0) * grid.km_per_px * math.cos(math.radians(lat))) ** 2).min())


def rank_exposure(subs, prob, mask, grid, radius_km):
    rpx = max(1, int(round(radius_km / grid.km_per_px)))
    out = []
    for f in subs["features"]:
        lon, lat = f["geometry"]["coordinates"][:2]; p = f.get("properties", {})
        out.append({"name": p.get("NAME") or p.get("name") or "substation", "kv": p.get("MAX_VOLT") or p.get("VOLTAGE") or p.get("voltage"),
                    "lat": lat, "lon": lon, "p_max": sample_max(prob, grid, lon, lat, rpx), "dist_km": dist_to_fire_km(mask, grid, lon, lat)})
    out.sort(key=lambda d: (-d["p_max"], d["dist_km"]))
    for i, d in enumerate(out): d["rank"] = i + 1
    return out


def line_exposure(lines, prob, grid, radius_km):
    rpx = max(1, int(round(radius_km / grid.km_per_px)))
    for f in lines["features"]:
        g = f["geometry"]; coords = g["coordinates"] if g["type"] == "LineString" else [c for part in g["coordinates"] for c in part]
        f["properties"]["p_max"] = max((sample_max(prob, grid, lon, lat, rpx) for lon, lat, *_ in coords), default=0.0)
    return lines

# ---------------------------------------------------------------- map

def build_map(masks, probs, dates, name, bbox, day, wind, lines, subs, radius_km, placeholder):
    T, H, W = masks.shape; grid = Grid(bbox, H, W); mask, prob = masks[day], probs[day]
    tag = lambda k: "  (placeholder)" if placeholder.get(k) else ""
    m = folium.Map(location=grid.center, zoom_start=12, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", name="Terrain (OpenTopoMap)", attr="© OpenStreetMap contributors, SRTM | © OpenTopoMap (CC-BY-SA)", max_zoom=17).add_to(m)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", name="Satellite (Esri)", attr="Esri, Maxar, Earthstar Geographics").add_to(m)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}", name="Hillshade overlay (Esri)",
                     attr="Esri", overlay=True, control=True, opacity=0.45, show=False).add_to(m)

    # probability raster
    fg_prob = folium.FeatureGroup(name=f"24h spread probability{tag('probs')}", show=True)
    folium.raster_layers.ImageOverlay(image="data:image/png;base64," + base64.b64encode(prob_png(prob)).decode(), bounds=grid.bounds, opacity=1.0, zindex=2,
                                      interactive=False).add_to(fg_prob)
    fg_prob.add_to(m)

    # perimeter
    fg_fire = folium.FeatureGroup(name=f"Fire perimeter, {dates[day]}{tag('fire')}", show=True)
    rings = mask_polygons(mask, grid)
    if rings:
        folium.GeoJson({"type": "Feature", "properties": {}, "geometry": {"type": "MultiLineString", "coordinates": rings}},
                       style_function=lambda _: {"color": P.ORANGE, "weight": 2.5, "opacity": 0.95},
                       tooltip=f"active fire, {dates[day]}: {int(mask.sum()):,} px").add_to(fg_fire)
    fg_fire.add_to(m)

    # wind
    fg_wind = folium.FeatureGroup(name=f"HRRR 10 m wind{tag('wind')}", show=True)
    scale = 0.004                                                    # degrees per m/s for the arrow length
    for lat, lon, u, v in wind:
        spd = math.hypot(u, v); deg = (math.degrees(math.atan2(u, v))) % 360        # direction the wind blows TOWARD, clockwise from north
        lat2, lon2 = lat + v * scale, lon + u * scale / max(math.cos(math.radians(lat)), 1e-6)
        tip = f"wind {spd:.1f} m/s toward {deg:.0f}°"
        folium.PolyLine([(lat, lon), (lat2, lon2)], color=P.INK_2, weight=2, opacity=0.8, tooltip=tip).add_to(fg_wind)
        folium.RegularPolygonMarker((lat2, lon2), number_of_sides=3, radius=5, rotation=deg - 90, color=P.INK_2, fill_color=P.INK_2, fill_opacity=0.9, weight=1, tooltip=tip).add_to(fg_wind)
    fg_wind.add_to(m)

    # grid assets
    lines = line_exposure(lines, prob, grid, radius_km)
    fg_lines = folium.FeatureGroup(name=f"Transmission lines (HIFLD){tag('grid')}", show=True)
    folium.GeoJson(lines, style_function=lambda f: {"color": P.PROB_CMAP(0.35 + 0.65 * f["properties"].get("p_max", 0)) and matplotlib.colors.to_hex(P.PROB_CMAP(0.35 + 0.65 * f["properties"].get("p_max", 0))),
                                                    "weight": 3, "opacity": 0.9},
                   tooltip=folium.GeoJsonTooltip(fields=[k for k in ("NAME", "VOLTAGE", "p_max") if all(k in f["properties"] for f in lines["features"])],
                                                 aliases=["line", "kV", "max P(24h) within reach"][:3])).add_to(fg_lines)
    fg_lines.add_to(m)
    fg_subs = folium.FeatureGroup(name=f"Substations (HIFLD){tag('grid')}", show=True)
    for f in subs["features"]:
        lon, lat = f["geometry"]["coordinates"][:2]; p = f.get("properties", {})
        folium.CircleMarker((lat, lon), radius=4, color=P.SURFACE, weight=1, fill_color=P.INK, fill_opacity=0.9, tooltip=f"{p.get('NAME', 'substation')} · {p.get('MAX_VOLT', p.get('VOLTAGE', '?'))} kV").add_to(fg_subs)
    fg_subs.add_to(m)

    # ranked exposure pins
    ranked = rank_exposure(subs, prob, mask, grid, radius_km)
    fg_exp = folium.FeatureGroup(name=f"Asset exposure, ranked{tag('grid') or tag('probs')}", show=True)
    for d in ranked:
        hot = d["p_max"] >= 0.5; bg = "#d03b3b" if hot else ("#ec835a" if d["p_max"] >= 0.2 else P.INK_2)
        icon = folium.DivIcon(html=f'<div style="background:{bg};color:#fff;border:2px solid #fff;border-radius:50%;width:24px;height:24px;line-height:24px;'
                                   f'text-align:center;font:600 12px system-ui,sans-serif;box-shadow:0 1px 3px rgba(0,0,0,.4)">{d["rank"]}</div>', icon_size=(24, 24), icon_anchor=(12, 12))
        dist = "inside fire" if d["dist_km"] == 0 else (f"{d['dist_km']:.1f} km from fire" if math.isfinite(d["dist_km"]) else "no fire today")
        popup = (f"<div style='font:13px system-ui,sans-serif;min-width:200px'><b>#{d['rank']} {d['name']}</b><br>{d['kv'] or '?'} kV substation<br>"
                 f"max P(fire in 24 h) within {radius_km:g} km: <b>{d['p_max']:.2f}</b><br>{dist}<br>"
                 f"<span style='color:#898781'>{'placeholder forecast' if placeholder.get('probs') else 'model forecast'} for {dates[min(day + 1, T - 1)]}</span></div>")
        folium.Marker((d["lat"], d["lon"]), icon=icon, popup=folium.Popup(popup, max_width=280), tooltip=f"#{d['rank']} {d['name']} · P {d['p_max']:.2f}").add_to(fg_exp)
    fg_exp.add_to(m)

    # legend / title box
    top = "".join(f"<tr><td style='padding:1px 6px 1px 0'>#{d['rank']}</td><td style='padding:1px 6px 1px 0'>{d['name']}</td><td><b>{d['p_max']:.2f}</b></td></tr>" for d in ranked[:5])
    legend = (f"<div style='position:fixed;top:12px;left:56px;z-index:1000;background:rgba(252,252,251,.94);padding:10px 12px;border-radius:6px;"
              f"box-shadow:0 1px 4px rgba(0,0,0,.25);font:12px system-ui,sans-serif;color:#0b0b0b;max-width:320px'>"
              f"<div style='font:600 14px Georgia,serif;margin-bottom:2px'>{name} · {dates[day]}</div>"
              f"<div style='color:#52514e;margin-bottom:6px'>forecast for {dates[min(day + 1, T - 1)]} · fire-spread-forecast-v1-small"
              f"{' · <b>placeholder data</b>' if any(placeholder.values()) else ''}</div>"
              f"<div style='display:flex;align-items:center;gap:6px'><span style='display:inline-block;width:90px;height:10px;background:linear-gradient(90deg,{P.SURFACE},{P.SEQ_BLUE[3]},{P.SEQ_BLUE[-1]})'></span>"
              f"<span style='color:#52514e'>P(fire in 24 h) 0 → 1</span></div>"
              f"<div style='margin-top:6px;color:#52514e'>Top exposure (substations)</div><table style='border-collapse:collapse'>{top}</table></div>")
    m.get_root().html.add_child(folium.Element(legend))
    folium.LayerControl(collapsed=False).add_to(m)
    plugins.Fullscreen().add_to(m); plugins.MousePosition(position="bottomright", num_digits=4).add_to(m)
    m.fit_bounds(grid.bounds)
    return m, ranked


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fire", nargs="?", help="WFTS fire dir or masks .npy/.npz; omit for a synthetic demo")
    ap.add_argument("--probs"); ap.add_argument("--day", type=int, help="day index shown (default: last day with a forecast)")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    ap.add_argument("--wind", help="CSV lat,lon,u,v"); ap.add_argument("--transmission", help="HIFLD lines GeoJSON"); ap.add_argument("--substations", help="HIFLD substations GeoJSON")
    ap.add_argument("--fetch-hifld", action="store_true", help="query HIFLD ArcGIS services for the bbox instead of files")
    ap.add_argument("--exposure-km", type=float, default=1.0); ap.add_argument("--out")
    a = ap.parse_args(argv); placeholder = {}
    if a.fire: masks, dates, name = load_fire(a.fire); bbox = tuple(a.bbox) if a.bbox else fire_bbox(a.fire)
    else: masks, dates, name = synthetic_fire(); bbox = tuple(a.bbox) if a.bbox else DEMO_BBOX; placeholder["fire"] = True
    T, H, W = masks.shape
    if a.probs: probs = align_probs(np.load(a.probs), T, (H, W))
    else: probs, placeholder["probs"] = placeholder_probs(masks), True
    day = a.day if a.day is not None else T - 2                                    # last day that has a next-day forecast
    if not 0 <= day <= T - 2: sys.exit(f"--day must be in 0..{T - 2} (day {T - 1} has no forecast)")
    grid = Grid(bbox, H, W)
    if a.wind: wind = load_wind_csv(a.wind)
    else: wind, placeholder["wind"] = synthetic_wind(grid), True
    if a.fetch_hifld:
        lines, subs = fetch_hifld("transmission", bbox), fetch_hifld("substations", bbox)
        print(f"HIFLD: {len(lines['features'])} lines, {len(subs['features'])} substations in bbox")
    elif a.transmission and a.substations: lines, subs = json.load(open(a.transmission)), json.load(open(a.substations))
    else: (lines, subs), placeholder["grid"] = synthetic_grid_assets(grid), True
    m, ranked = build_map(masks, probs[day - 0: day + 1][0:1].reshape(1, H, W).repeat(T - 1, 0) if False else probs, dates, name, bbox, day, wind, lines, subs, a.exposure_km, placeholder)
    out = a.out or os.path.join("out", "map", f"{name.replace('/', '_')}_{dates[day].replace(' ', '')}.html")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True); m.save(out)
    print(f"{name} day {day} ({dates[day]}): {H}x{W} px, bbox {tuple(round(b, 4) for b in bbox)}; placeholder: {sorted(placeholder) or 'none'}")
    for d in ranked[:5]: print(f"  #{d['rank']} {d['name']:<24} P={d['p_max']:.2f}  {d['dist_km']:.1f} km")
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
