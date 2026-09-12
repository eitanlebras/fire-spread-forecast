"""The demo: one standalone HTML map, every layer toggleable, a ranked "tonight" panel, and a customer switch.

  python -m viz.demo_map                                             # placeholder territory: 3 synthetic fires, synthetic assets
  python -m viz.demo_map --event "Bear Creek=/root/data/wfts/2021/fire_123:6" --probs "Bear Creek=p.npy" --fetch-hifld
  python -m viz.demo_map --event A=DIR --event B=DIR --lines l.geojson --substations s.geojson --stands st.geojson --wind w.csv
  options: --fuel NAME=rgb.png (false-colour composite for that event's bbox)  --n-active 14  --territory "..."  --out FILE

Layers in draw order: basemap (satellite / terrain / OSM) -> Esri hillshade -> fuel state (OlmoEarth composite, false colour)
-> HRRR 10 m wind -> fire perimeter today -> 24 h burn probability (the model) -> assets coloured by exposure -> ranked pins 1-5.
Customer switch (Utility / Timber / Volunteer fire district) swaps only the asset layer and the panel; probability stays.

Real events are WFTS fire dirs (bbox + CRS read from the GeoTIFF; the probability field is warped to WGS84 for the overlay)
or masks .npz with --bbox. Anything not supplied is synthesised and labelled "(placeholder)". Asset values and
vulnerabilities are the assumptions at the top of viz/exposure.py, never customer data.
"""
import argparse, base64, io, json, math, os, sys, numpy as np
import matplotlib
matplotlib.use("Agg")
import folium
from folium import plugins
from PIL import Image
from viz import palette as P, exposure as X
from viz.map import Grid, prob_png, mask_polygons, synthetic_wind, exposure_color, load_wind_csv, synthetic_fire
from viz.replay import load_fire, placeholder_probs, align_probs, _blur

TERRITORY = "A utility service territory in Oregon"
DEMO_BBOX = (-122.35, 44.05, -121.25, 44.85)                       # ~88 x 89 km
DEMO_FIRES = [   # name, centre (lon, lat), half-size (deg), peak P, plume direction (deg, blowing toward), day index for the ring
    ("Bear Creek", (-121.72, 44.53), 0.13, 0.78, 55, 4),      # sits on the 115 kV and 230 kV circuits
    ("Gulch",      (-121.50, 44.22), 0.11, 0.52, 80, 4),      # reaches the 69 kV feeder

    ("Pine",       (-122.12, 44.70), 0.09, 0.19, 40, 3),
]
N_ACTIVE_DEMO = 14
CUSTOMERS = {"utility": "Utility", "timber": "Timber", "volunteer": "Volunteer fire district"}

# ---------------------------------------------------------------- placeholder events

def synthetic_event(name, center, half, peak, plume_deg, day, seed):
    """A ring-shaped active fire plus a smooth, peaked, downwind-skewed P field with light speckle. Distinct per fire."""
    rng = np.random.default_rng(seed); H = W = 160
    masks, _, _ = synthetic_fire(H, W, T=day + 2, seed=seed); mask = masks[day]
    dx, dy = math.sin(math.radians(plume_deg)), math.cos(math.radians(plume_deg))
    shifted = _shift(mask.astype(np.float32), -int(10 * dy), int(10 * dx))                   # push the field downwind (no wrap-around)
    field = _blur(shifted.astype(np.float32), 9.0); field = field / max(field.max(), 1e-6)
    field = field * peak * (1 + 0.08 * rng.standard_normal((H, W))); field[field < 0.06] = 0        # cut the gaussian tail so the raster box has no edge
    prob = np.clip(np.where(mask, np.maximum(field, 0.9 * peak), field), 0, peak).astype(np.float32)
    lon, lat = center; bbox = (lon - half, lat - half * 0.8, lon + half, lat + half * 0.8)
    return {"name": name, "mask": mask, "prob": prob, "bbox": bbox, "raster": X.Raster.from_bbox(prob, bbox), "placeholder": True, "date": "tonight"}


def _shift(a, dr, dc):
    out = np.zeros_like(a); H, W = a.shape
    out[max(dr, 0):H + min(dr, 0), max(dc, 0):W + min(dc, 0)] = a[max(-dr, 0):H + min(-dr, 0), max(-dc, 0):W + min(-dc, 0)]
    return out


def synthetic_fuel_png(H, W, seed):
    """Placeholder for the OlmoEarth false-colour composite: smooth fields -> RGB where red = dry/cured fuel, green = live vegetation."""
    rng = np.random.default_rng(seed)
    a, b = _blur(rng.random((H, W)), 5), _blur(rng.random((H, W)), 9)
    a, b = (a - a.min()) / np.ptp(a), (b - b.min()) / np.ptp(b)
    rgb = np.stack([60 + 150 * a, 70 + 120 * b, 40 + 50 * (1 - a)], -1).astype(np.uint8)
    buf = io.BytesIO(); Image.fromarray(rgb).save(buf, "PNG"); return buf.getvalue()

# ---------------------------------------------------------------- real events

BUFFER_KM, FEATHER_KM = 10.0, 3.0      # forecast shown only this far from today's fire; edge fades over FEATHER_KM


def growth_field(prob, today, pixel_km, buffer_km=BUFFER_KM, feather_km=FEATHER_KM, seam_sigma=0.7):
    """What the model adds: P(burn tomorrow) on pixels NOT burning today, confined to a feathered buffer around today's
    fire, lightly smoothed so pixel/tile seams don't read as artifacts. Returns (display field, exposure field, fade)."""
    from scipy import ndimage as ndi
    d = ndi.distance_transform_edt(~today) * pixel_km
    fade = np.clip((buffer_km - d) / feather_km, 0, 1).astype(np.float32)
    shown = ndi.gaussian_filter(prob * fade * (~today), seam_sigma).astype(np.float32)
    return shown, (prob * fade).astype(np.float32), fade


def load_event(name, src, day, probs_path=None):
    """day = TODAY (0..T-2): today's fire is the grey fill, the model's forecast FOR TOMORROW (probs[day+1], made from
    today) is the blue growth field, and tomorrow's observed new fire (if the record has it) is the orange outline."""
    masks, dates, _ = load_fire(src); T = len(masks)
    probs = align_probs(np.load(probs_path), T, masks.shape[1:]) if probs_path else placeholder_probs(masks)
    day = T - 2 if day is None else day
    if not 0 <= day <= T - 2: sys.exit(f"{name}: --day must be in 0..{T - 2}")
    today = masks[day].astype(bool); new_next = masks[day + 1].astype(bool) & ~today
    ev = {"name": name, "date": dates[day], "date_next": dates[day + 1], "placeholder": probs_path is None}
    if os.path.isdir(src):
        import glob, rasterio
        from rasterio.warp import reproject, Resampling, calculate_default_transform
        tif = sorted(glob.glob(os.path.join(src, "*.tif")))[0]
        with rasterio.open(tif) as ds:
            pixel_km = abs(ds.transform.a) / 1000.0 if ds.crs.to_epsg() != 4326 else abs(ds.transform.a) * 111.0
            shown, expo, _ = growth_field(probs[day + 1], today, pixel_km)
            ev["raster"] = X.Raster.from_geotiff(tif, prob=expo)                                  # event CRS, for the exposure engine
            if ds.crs.to_epsg() == 4326: ev["prob"], ev["mask"], ev["new_next"], ev["bbox"] = shown, today, new_next, tuple(ds.bounds)
            else:                                                                                   # warp to a WGS84 grid for the overlay
                tr, w, h = calculate_default_transform(ds.crs, "EPSG:4326", ds.width, ds.height, *ds.bounds)
                out = []
                for arr in (shown, today.astype(np.float32), new_next.astype(np.float32)):
                    dst = np.zeros((h, w), np.float32)
                    reproject(arr, dst, src_transform=ds.transform, src_crs=ds.crs, dst_transform=tr, dst_crs="EPSG:4326", resampling=Resampling.bilinear); out.append(dst)
                ev["prob"], ev["mask"], ev["new_next"] = out[0], out[1] > 0.5, out[2] > 0.5
                ev["bbox"] = (tr.c, tr.f + h * tr.e, tr.c + w * tr.a, tr.f)
    else:
        sys.exit(f"{name}: masks files need a bbox; use --event NAME=DIR for GeoTIFFs or supply bbox_wgs84 in the npz") if "bbox" not in np.load(src) else None
        ev["bbox"] = tuple(float(b) for b in np.load(src)["bbox"]); pixel_km = (ev["bbox"][2] - ev["bbox"][0]) * 111.0 * math.cos(math.radians((ev["bbox"][1] + ev["bbox"][3]) / 2)) / masks.shape[2]
        shown, expo, _ = growth_field(probs[day + 1], today, pixel_km)
        ev["prob"], ev["mask"], ev["new_next"] = shown, today, new_next
        ev["raster"] = X.Raster.from_bbox(expo, ev["bbox"])
    ev["n_today"], ev["n_new_next"] = int(today.sum()), int(new_next.sum())
    return ev

# ---------------------------------------------------------------- panel text

def fire_headline(rows, customer, threshold):
    """Best row for the panel line of one fire, for one customer. Returns (exposure, html fragment)."""
    kinds = {"utility": ("line", "substation"), "timber": ("stand",)}[customer]
    cand = [r for r in rows if r["kind"] in kinds and r["exposure_usd"] > 0]
    if not cand: return 0.0, None
    hot = [r for r in cand if r["worst"] >= threshold]                      # prefer the costliest asset that is actually above threshold
    r = max(hot or cand, key=lambda r: r["exposure_usd"])
    if not hot: return r["exposure_usd"], f"{r['worst']:.0%} near {r['name']} · <span style='color:#52514e'>monitor</span>"
    if r["kind"] == "line": over = f"{r['worst']:.0%} over {r['miles_above']:.1f} mi, {r['name']}"
    elif r["kind"] == "substation": over = f"{r['worst']:.0%} at {r['name']}"
    else: over = f"{r['worst']:.0%} over {r['name']}, {r['acres_above']:,.0f} ac"
    return r["exposure_usd"], f"{over} · <b>{X.usd(r['exposure_usd'])}</b>"


def build_panels(events, threshold, n_active):
    """{customer: html} for the top-left panel: 'Tonight — k of n active fires' + the ranked fire lines."""
    panels = {}
    for c in CUSTOMERS:
        lines = []
        for ev in events:
            if c == "volunteer":
                ac = float((ev["prob"] >= threshold).sum()) * ev["raster"].px_area_m2 / X.M2_PER_ACRE
                lines.append((float(ev["prob"].max()), f"{ev['prob'].max():.0%} peak · {ac:,.0f} ac forecast above {threshold:.0%}"))
            else:
                e, frag = fire_headline(ev["rows"], c, threshold)
                lines.append((e, frag or f"{ev['prob'].max():.0%} peak, no {'grid asset' if c == 'utility' else 'stand'} in reach · <span style='color:#52514e'>monitor</span>"))
        order = sorted(zip(lines, events), key=lambda t: -t[0][0])
        body = "".join(f"<div style='margin:3px 0'><b>{i + 1}. {ev['name']}</b> · {frag}</div>" for i, ((_, frag), ev) in enumerate(order))
        head = f"<div style='font:600 15px Georgia,serif;margin-bottom:6px'>Tonight — {len(events)} of {n_active} active fires</div>"
        note = {"utility": "exposure = P(burn) × replacement value × vulnerability × exposed miles", "timber": "exposure = P(burn) × merchantable value × vulnerability × acres",
                "volunteer": "<b>free</b> · probability field only, no asset integration"}[c]
        panels[c] = head + body + f"<div style='color:#52514e;margin-top:6px;font-size:11px'>{note}</div>"
    return panels

# ---------------------------------------------------------------- map

def add_assets(m, events, customer, threshold, network):
    """Asset layer + top-5 pins for one customer. The whole territory network is drawn neutral underneath; assessed pieces
    (inside a fire raster) are coloured by exposure on top. Groups stay out of the layer control; the customer switch drives them."""
    fg = folium.FeatureGroup(name=f"assets:{customer}", control=False, show=(customer == "utility")); pins = folium.FeatureGroup(name=f"pins:{customer}", control=False, show=(customer == "utility"))
    lines, subs, stands = network; neutral = {"color": P.MUTED, "weight": 2, "opacity": 0.75, "fillColor": P.MUTED, "fillOpacity": 0.08}
    base = {"utility": [f for f in (lines or []) + (subs or []) if f["geometry"]["type"] != "Point"], "timber": stands or []}[customer]
    if base: folium.GeoJson({"type": "FeatureCollection", "features": base}, style_function=lambda _: neutral, interactive=False).add_to(fg)
    for f in (subs or []) if customer == "utility" else []:
        lon, lat = f["geometry"]["coordinates"][:2]; folium.CircleMarker((lat, lon), radius=3, color=P.MUTED, weight=1, fill_color=P.MUTED, fill_opacity=0.8, interactive=False).add_to(fg)
    cand = []
    for ev in events:
        R = ev["raster"]
        if customer == "utility":
            segs = X.to_geojson(ev["segs"], R); subs = X.to_geojson(ev["subs"], R)
            if segs["features"]:
                folium.GeoJson(segs, style_function=lambda f: {"color": exposure_color(f["properties"]["p_mean"]), "weight": 2 + f["properties"]["voltage_kv"] / 150, "opacity": 0.9},
                               tooltip=folium.GeoJsonTooltip(fields=["circuit", "voltage_kv", "miles", "p_mean", "p_max", "exposure_usd"],
                                                             aliases=["circuit", "kV", "segment mi", "mean P", "max P", "exposure $"])).add_to(fg)
            for f in subs["features"]:
                lon, lat = f["geometry"]["coordinates"]; p = f["properties"]
                folium.CircleMarker((lat, lon), radius=5, color=P.SURFACE, weight=1, fill_color=exposure_color(p["p_max"]), fill_opacity=0.95,
                                    tooltip=f"{p['name']} · {p['voltage_kv']:.0f} kV · P {p['p_max']:.0%} · {X.usd(p['exposure_usd'])}").add_to(fg)
            for r in ev["rows"]:
                if r["kind"] not in ("line", "substation") or r["exposure_usd"] <= 0: continue
                if r["kind"] == "line":
                    worst = max((s for s in ev["segs"] if s["properties"]["circuit"] == r["name"]), key=lambda s: s["properties"]["p_max"])
                    pt = worst["geom"].interpolate(0.5, normalized=True)
                else: pt = next(s["geom"] for s in ev["subs"] if s["properties"]["name"] == r["name"])
                lon, lat = R.to_wgs.transform(pt.x, pt.y)
                cand.append((r["exposure_usd"], lat, lon, f"<b>{r['name']}</b><br>{ev['name']} · {r['quantity']}<br>{'worst segment' if r['kind'] == 'line' else 'P'} {r['worst']:.0%} · <b>{X.usd(r['exposure_usd'])}</b>", r["name"]))
        elif customer == "timber":
            st = X.to_geojson(ev["stands"], R)
            if st["features"]:
                folium.GeoJson(st, style_function=lambda f: {"color": exposure_color(f["properties"]["p_max"]), "weight": 1.5, "fillColor": exposure_color(f["properties"]["p_mean"]),
                                                             "fillOpacity": 0.25 if f["properties"]["p_mean"] < 0.2 else 0.45},
                               tooltip=folium.GeoJsonTooltip(fields=["name", "species", "age_years", "acres", "acres_above", "p_max", "salvage_months", "exposure_usd"],
                                                             aliases=["stand", "species", "age (yr)", "acres", f"acres above {threshold:.0%}", "max P", "salvage window (mo)", "exposure $"])).add_to(fg)
            for r in ev["rows"]:
                if r["kind"] != "stand" or r["exposure_usd"] <= 0: continue
                g = next(s["geom"] for s in ev["stands"] if s["properties"]["name"] == r["name"]); c = g.representative_point()
                lon, lat = R.to_wgs.transform(c.x, c.y)
                cand.append((r["exposure_usd"], lat, lon, f"<b>{r['name']}</b><br>{ev['name']} · {r['quantity']}<br>{r['species'].replace('_', ' ')}, {r['age_years']:.0f} yr · salvage window {r['salvage_months']} mo · <b>{X.usd(r['exposure_usd'])}</b>", r["name"]))
    seen, top = set(), []
    for c in sorted(cand, key=lambda t: -t[0]):
        if c[4] in seen: continue
        seen.add(c[4]); top.append(c)
        if len(top) == 5: break
    for i, (e, lat, lon, html, _) in reversed(list(enumerate(top, 1))):                 # #1 added last so it sits on top
        bg = "#d03b3b" if i <= 2 else "#ec835a" if i <= 4 else P.INK_2
        icon = folium.DivIcon(html=f'<div style="background:{bg};color:#fff;border:2px solid #fff;border-radius:50%;width:26px;height:26px;line-height:26px;text-align:center;'
                                   f'font:700 13px system-ui,sans-serif;box-shadow:0 1px 3px rgba(0,0,0,.4)">{i}</div>', icon_size=(26, 26), icon_anchor=(13, 13))
        folium.Marker((lat, lon), icon=icon, popup=folium.Popup(f"<div style='font:13px system-ui,sans-serif;min-width:220px'>#{i} {html}</div>", max_width=300)).add_to(pins)
    fg.add_to(m); pins.add_to(m); return fg, pins


def build_map(events, wind, fuel, panels, placeholder, territory, threshold, network):
    m = folium.Map(location=(np.mean([e["bbox"][1] + e["bbox"][3] for e in events]) / 2, np.mean([e["bbox"][0] + e["bbox"][2] for e in events]) / 2), zoom_start=10, tiles=None, control_scale=True)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", name="Satellite (Esri)", attr="Esri, Maxar, Earthstar Geographics").add_to(m)
    folium.TileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", name="Terrain (OpenTopoMap)", attr="© OpenStreetMap contributors, SRTM | © OpenTopoMap (CC-BY-SA)", max_zoom=17).add_to(m)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}", name="Terrain hillshade (Esri)", attr="Esri", overlay=True, opacity=0.5, show=False).add_to(m)
    tag = lambda k: "  (placeholder)" if placeholder.get(k) else ""
    fg_fuel = folium.FeatureGroup(name=f"Fuel state — OlmoEarth composite, false colour{tag('fuel')}", show=False)
    for ev in events:
        png = fuel.get(ev["name"])
        if png: folium.raster_layers.ImageOverlay(image="data:image/png;base64," + base64.b64encode(png).decode(), bounds=Grid(ev["bbox"], 1, 1).bounds, opacity=0.75, zindex=1).add_to(fg_fuel)
    fg_fuel.add_to(m)
    fg_wind = folium.FeatureGroup(name=f"HRRR 10 m wind{tag('wind')}", show=False); scale = 0.004
    for lat, lon, u, v in wind:
        spd = math.hypot(u, v); deg = math.degrees(math.atan2(u, v)) % 360; lat2, lon2 = lat + v * scale, lon + u * scale / max(math.cos(math.radians(lat)), 1e-6)
        folium.PolyLine([(lat, lon), (lat2, lon2)], color=P.INK_2, weight=2, opacity=0.8, tooltip=f"wind {spd:.1f} m/s toward {deg:.0f}°").add_to(fg_wind)
        folium.RegularPolygonMarker((lat2, lon2), number_of_sides=3, radius=5, rotation=deg - 90, color=P.INK_2, fill_color=P.INK_2, fill_opacity=0.9, weight=1).add_to(fg_wind)
    fg_wind.add_to(m)
    fg_fire = folium.FeatureGroup(name=f"Burning today — VIIRS detections (grey){tag('fire')}", show=True)
    for ev in events:
        g = Grid(ev["bbox"], *ev["mask"].shape); rings = mask_polygons(ev["mask"], g)
        if rings: folium.GeoJson({"type": "Feature", "properties": {}, "geometry": {"type": "MultiPolygon", "coordinates": [[r] for r in rings]}},
                                 style_function=lambda _: {"color": P.INK, "weight": 1.5, "opacity": 0.95, "fillColor": "#3a3a37", "fillOpacity": 0.7},
                                 tooltip=f"{ev['name']} · burning {ev['date']} ({ev.get('n_today', '?')} px)").add_to(fg_fire)
    fg_fire.add_to(m)
    fg_prob = folium.FeatureGroup(name=f"Where it spreads in 24 h — the model, growth region only{tag('probs')}", show=True)
    for ev in events:
        folium.raster_layers.ImageOverlay(image="data:image/png;base64," + base64.b64encode(prob_png(ev["prob"])).decode(), bounds=Grid(ev["bbox"], 1, 1).bounds, opacity=1.0, zindex=3, interactive=False).add_to(fg_prob)
    fg_prob.add_to(m)
    fg_next = folium.FeatureGroup(name="What actually burned next day — new fire (orange)", show=True)
    for ev in events:
        nn = ev.get("new_next")
        if nn is None or not nn.any(): continue
        g = Grid(ev["bbox"], *nn.shape); rings = mask_polygons(nn, g)
        if rings: folium.GeoJson({"type": "Feature", "properties": {}, "geometry": {"type": "MultiLineString", "coordinates": rings}},
                                 style_function=lambda _: {"color": P.ORANGE, "weight": 2.5, "opacity": 0.95},
                                 tooltip=f"{ev['name']} · new fire observed {ev.get('date_next', '')} ({ev.get('n_new_next', '?')} px)").add_to(fg_next)
    fg_next.add_to(m)
    groups = {c: add_assets(m, events, c, threshold, network) if c != "volunteer" else () for c in CUSTOMERS}
    folium.LayerControl(collapsed=False).add_to(m); plugins.Fullscreen().add_to(m)
    m.fit_bounds([[min(e["bbox"][1] for e in events) - 0.05, min(e["bbox"][0] for e in events) - 0.05], [max(e["bbox"][3] for e in events) + 0.05, max(e["bbox"][2] for e in events) + 0.05]])

    # panel + customer switch + footer
    btn = "".join(f"<button data-c='{k}' onclick=\"setCustomer('{k}')\" style='font:12px system-ui,sans-serif;padding:4px 9px;border:1px solid #c3c2b7;background:#fff;border-radius:4px;cursor:pointer'>{v}</button>" for k, v in CUSTOMERS.items())
    html = (f"<div id='fsf-panel' style='position:fixed;top:12px;left:56px;z-index:1000;background:rgba(252,252,251,.95);padding:12px 14px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.25);"
            f"font:13px system-ui,sans-serif;color:#0b0b0b;max-width:420px'>"
            f"<div style='display:flex;gap:6px;margin-bottom:10px' id='fsf-switch'>{btn}</div><div id='fsf-body'></div>"
            f"<div style='color:#898781;margin-top:8px;font-size:11px'>{territory} · fire-spread-forecast-v1-small"
            f"{' · <b>placeholder fires and wind</b>' if placeholder.get('fire') else ''}</div></div>"
            f"<div style='position:fixed;bottom:22px;left:12px;z-index:1000;background:rgba(252,252,251,.9);padding:5px 9px;border-radius:4px;font:11px system-ui,sans-serif;color:#52514e'>"
            f"public infrastructure data (HIFLD{', placeholder geometry' if placeholder.get('grid') else ''}) · illustrative asset values and vulnerabilities · "
            f"<span style='display:inline-block;width:70px;height:9px;vertical-align:middle;background:linear-gradient(90deg,{P.SURFACE},{P.SEQ_BLUE[3]},{P.SEQ_BLUE[-1]})'></span> P(burn in 24 h) 0 → 1 · shown within {BUFFER_KM:g} km of today's fire, growth region only</div>")
    m.get_root().html.add_child(folium.Element(html))
    js_groups = {c: [g.get_name() for g in grp] for c, grp in groups.items()}
    js = f"""
    document.addEventListener("DOMContentLoaded", function () {{      // folium's own scripts run inline first; the groups exist by now
    var fsfGroups = {{{", ".join(f"{c}: [{', '.join(names)}]" for c, names in js_groups.items())}}};
    var fsfPanels = {json.dumps(panels)};
    function setCustomer(c) {{
      for (var k in fsfGroups) fsfGroups[k].forEach(function (g) {{ if ({m.get_name()}.hasLayer(g)) {m.get_name()}.removeLayer(g); }});
      fsfGroups[c].forEach(function (g) {{ {m.get_name()}.addLayer(g); }});
      document.getElementById('fsf-body').innerHTML = fsfPanels[c];
      document.querySelectorAll('#fsf-switch button').forEach(function (b) {{ var on = b.dataset.c === c; b.style.background = on ? '#0b0b0b' : '#fff'; b.style.color = on ? '#fff' : '#0b0b0b'; }});
    }}
    window.setCustomer = setCustomer; var c0 = (location.hash || "#utility").slice(1); setCustomer(c0 in fsfGroups ? c0 : "utility");   // #timber opens on Timber
    }});
    """
    m.get_root().script.add_child(folium.Element(js))
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", action="append", default=[], help="NAME=FIRE_DIR[:DAY] (repeatable)"); ap.add_argument("--probs", action="append", default=[], help="NAME=P.npy")
    ap.add_argument("--fuel", action="append", default=[], help="NAME=rgb.png false-colour composite on the event bbox")
    ap.add_argument("--lines"); ap.add_argument("--substations"); ap.add_argument("--stands"); ap.add_argument("--fetch-hifld", action="store_true"); ap.add_argument("--wind")
    ap.add_argument("--threshold", type=float, default=X.THRESHOLD); ap.add_argument("--n-active", type=int); ap.add_argument("--territory", default=TERRITORY); ap.add_argument("--out")
    a = ap.parse_args(argv); placeholder = {}
    probs = dict(s.split("=", 1) for s in a.probs); fuel_paths = dict(s.split("=", 1) for s in a.fuel)
    if a.event:
        events = []
        for spec in a.event:
            name, rest = spec.split("=", 1); src, _, day = rest.partition(":"); events.append(load_event(name, src, int(day) if day else None, probs.get(name)))
        placeholder["probs"] = any(e["placeholder"] for e in events)
    else:
        events = [synthetic_event(*spec, seed=i) for i, spec in enumerate(DEMO_FIRES)]; placeholder["fire"] = placeholder["probs"] = True
    bbox = (min(e["bbox"][0] for e in events) - 0.1, min(e["bbox"][1] for e in events) - 0.1, max(e["bbox"][2] for e in events) + 0.1, max(e["bbox"][3] for e in events) + 0.1)
    if not a.event: bbox = DEMO_BBOX
    grid = Grid(bbox, 1, 1)
    wind = load_wind_csv(a.wind) if a.wind else synthetic_wind(grid, n=13); placeholder["wind"] = not a.wind
    fuel = {}
    for ev in events:
        if ev["name"] in fuel_paths: fuel[ev["name"]] = open(fuel_paths[ev["name"]], "rb").read()
        else: fuel[ev["name"]] = synthetic_fuel_png(*ev["prob"].shape, seed=hash(ev["name"]) % 1000); placeholder["fuel"] = True
    if a.fetch_hifld: lines, subs = X.fetch_hifld("lines", bbox), X.fetch_hifld("substations", bbox); stands = X.load_features(a.stands)[0] if a.stands else None
    elif a.lines or a.substations or a.stands:
        lines = X.load_features(a.lines)[0] if a.lines else None; subs = X.load_features(a.substations)[0] if a.substations else None; stands = X.load_features(a.stands)[0] if a.stands else None
    else: lines, subs, stands = X.synthetic_assets(bbox); placeholder["grid"] = True
    for ev in events:
        res = X.assess(ev["raster"], lines, subs, stands, "EPSG:4326", a.threshold); ev.update(res)
        print(f"{ev['name']}: peak P {ev['prob'].max():.2f}; top: " + ("; ".join(f"{r['name']} {X.usd(r['exposure_usd'])}" for r in res["rows"][:3]) or "nothing in reach"))
    panels = build_panels(events, a.threshold, a.n_active or (N_ACTIVE_DEMO if not a.event else len(events)))
    m = build_map(events, wind, fuel, panels, placeholder, a.territory, a.threshold, (lines, subs, stands))
    out = a.out or os.path.join("out", "demo_map", "demo.html" if not a.event else "map.html")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True); m.save(out)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB); placeholder: {sorted(k for k, v in placeholder.items() if v) or 'none'}")


if __name__ == "__main__":
    main()
