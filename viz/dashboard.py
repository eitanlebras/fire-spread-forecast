"""Fireline dashboard: header strip, map (left ~65%), sidebar (right ~35%) with the queue, the model's reasons, the track
record, and what NGFS sees burning right now.

  python -m viz.dashboard --event "Redding=/root/data/wfts/tif/2021/fire_25547988:7" --probs Redding=/root/data/preds/2021/fire_25547988/probs.npy \
      --fetch-hifld --territory "Shasta County, CA" --active-root /root/data/wfts/tif/2021 --out demo/redding/dashboard.html
  options: --lines/--substations/--stands FILES  --threshold 0.4  --no-live (skip the NGFS pull)  --live-state CA

Same event loading and exposure engine as viz.demo_map (today's fire = grey, the model's forecast for tomorrow = blue growth
field, tomorrow's observed new fire = orange). Differences: no placeholder layers at all (fuel and wind are dropped unless
real data is supplied), plain-language layer names, collapsed layer control, water from the WFTS land-cover band, and a
sidebar that explains the ranking from the WFTS weather bands of the day (GridMET wind / ERC / PDSI, GFS forecast wind).

Live: NGFS (NOAA Next Generation Fire System, CIMSS/SSEC, GOES-18/19) scene detections, pulled at build time from the
RealEarth viewer API with a fallback to the public mirror (github.com/Deasus/firestorm-ngfs-data, ~5 min cadence), and
refreshed in the browser from that mirror every 5 minutes. The model is NOT run on live fires here: fire-spread-forecast
needs the 23-band WFTS input stack (VIIRS reflectances, GridMET, GFS, SRTM, MODIS land cover) which has no live pipeline yet.
"""
import argparse, base64, glob, gzip, hashlib, html as H, json, math, os, secrets, sys, time, urllib.parse, urllib.request, http.cookiejar, numpy as np
import matplotlib
matplotlib.use("Agg")
import folium
from folium import plugins
from viz import palette as P, exposure as X
from viz.map import Grid, prob_png, mask_polygons, exposure_color, load_wind_csv
from viz.demo_map import load_event, add_assets, BUFFER_KM, CUSTOMERS
from viz.replay import load_fire, align_probs

MODEL = "fire-spread-forecast-v1-small"
TRACK_RECORD = {   # WildfireSpreadTS 2021 test split, 30 held-out fires, wfts_only baseline (results/, README)
    "fires": 30, "advance_share": 0.19, "advance_x_chance": 54, "interior_aucpr": 0.451, "interior_chance": 0.240,
    "aucpr": 0.552, "persistence": 0.273, "ece": 0.0044, "spot_hits": 0,
}
NGFS_MIRROR = "https://raw.githubusercontent.com/Deasus/firestorm-ngfs-data/main/data/ngfs.json"
RE_BASE = "https://re-ngfs-pub.ssec.wisc.edu"
LC_WATER = 17

# ---------------------------------------------------------------- the model's reasons, from the day's WFTS bands

def compass(deg):
    return ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"][int(((deg % 360) + 11.25) // 22.5) % 16]


def circ_mean(deg):
    r = np.radians(np.asarray(deg, float)); return float(np.degrees(np.arctan2(np.nanmean(np.sin(r)), np.nanmean(np.cos(r)))) % 360)


def explain(src, day, probs, masks, threshold):
    """Read the raw WFTS GeoTIFF of `day` and summarise the drivers over today's fire + a 5 km halo, plus what the forecast
    field says (direction, reach, peak) and, if tomorrow is on record, how it verified."""
    import rasterio
    from scipy import ndimage as ndi
    tif = sorted(glob.glob(os.path.join(src, "*.tif")))[day]
    with rasterio.open(tif) as ds:
        a = ds.read().astype(np.float32); names = list(ds.descriptions); px_km = abs(ds.transform.a) / 1000.0
    b = lambda n: a[names.index(n)]
    today = masks[day].astype(bool); halo = ndi.binary_dilation(today, iterations=max(1, int(5 / px_km))) & ~today
    zone = today | halo
    def mean(n, where=zone):
        v = b(n)[where]; v = v[np.isfinite(v)]; return float(v.mean()) if v.size else float("nan")
    wind_from = circ_mean(b("wind direction")[zone]); wind_ms = mean("wind speed")
    fc_from = circ_mean(b("forecast wind direction")[zone]); fc_ms = mean("forecast wind speed")
    aspect = circ_mean(b("aspect")[halo]); slope = mean("slope", halo)
    upslope = (aspect + 180) % 360; toward = (wind_from + 180) % 360
    align = abs(((toward - upslope) + 180) % 360 - 180)                    # 0 = wind pushes straight upslope
    out = {"wind_mph": wind_ms * 2.237, "wind_from": wind_from, "wind_toward": toward, "fc_wind_mph": fc_ms * 2.237, "fc_from": fc_from, "fc_toward": (fc_from + 180) % 360,
           "veer": abs(((fc_from - wind_from) + 180) % 360 - 180), "erc": mean("energy release component"), "pdsi": mean("pdsi"), "precip_mm": mean("total precipitation"),
           "tmax_c": mean("maximum temperature") - 273.15, "q": mean("specific humidity"), "slope": slope, "aspect": aspect, "upslope": upslope, "align": align,
           "ndvi": mean("NDVI_last") / 10000.0, "water": (b("LC_Type1") == LC_WATER)}
    # forecast field geometry: P for tomorrow on the growth region
    p = probs[day + 1].astype(np.float32) * (~today); rr, cc = np.nonzero(today); cy, cx = rr.mean(), cc.mean()
    d = ndi.distance_transform_edt(~today) * px_km
    hot = p >= threshold
    out["peak"] = float(p.max()); out["reach_km"] = float(d[hot].max()) if hot.any() else 0.0
    out["area_above_ac"] = float(hot.sum()) * (px_km * 1000) ** 2 / X.M2_PER_ACRE
    w = p * (d <= 6.0)
    if w.sum() > 0:
        yy, xx = np.nonzero(w); wy, wx = np.average(yy, weights=w[yy, xx]), np.average(xx, weights=w[yy, xx])
        out["dir"] = float(np.degrees(np.arctan2(wx - cx, -(wy - cy))) % 360)
        sect = np.degrees(np.arctan2(xx - cx, -(yy - cy))) % 360; diff = np.abs(((sect - out["dir"]) + 180) % 360 - 180)
        out["dir_share"] = float(w[yy, xx][diff <= 45].sum() / w.sum())    # share of the forecast mass inside a 90° cone
    else: out["dir"], out["dir_share"] = float("nan"), 0.0
    out["confidence"] = "high" if out["peak"] >= 0.6 and out["dir_share"] >= 0.6 else "medium" if out["peak"] >= threshold else "low"
    if day + 1 < len(masks):
        nn = masks[day + 1].astype(bool) & ~today; out["verified_px"] = int(nn.sum())
        out["verified_share"] = float(hot[nn].mean()) if nn.any() else float("nan")
        if nn.any():
            yy, xx = np.nonzero(nn); out["verified_dir"] = float(np.degrees(np.arctan2(xx.mean() - cx, -(yy.mean() - cy))) % 360)
    return out


def pdsi_word(v):
    return "extreme drought" if v <= -4 else "severe drought" if v <= -3 else "moderate drought" if v <= -2 else "near normal" if v < 2 else "wet"


def why_html(ev, w, threshold):
    if not w: return "<div class='muted'>no WFTS bands for this event</div>"
    align = "aligned upslope" if w["align"] <= 45 else "cross-slope" if w["align"] <= 135 else "pushing downslope"
    rain = "no rain" if w["precip_mm"] < 0.5 else f"{w['precip_mm']:.0f} mm rain"
    rows = [
        f"Wind <b>{w['wind_mph']:.0f} mph</b> from the {compass(w['wind_from'])}, {align} ({w['slope']:.0f}° slopes facing {compass(w['aspect'])})",
        f"Fuel dryness: ERC <b>{w['erc']:.0f}</b> · PDSI {w['pdsi']:.1f} ({pdsi_word(w['pdsi'])}) · {rain} · max {w['tmax_c']:.0f} °C",
        f"Tomorrow's wind (GFS): <b>{w['fc_wind_mph']:.0f} mph</b> from the {compass(w['fc_from'])}" + (f", veering {w['veer']:.0f}°" if w["veer"] >= 30 else ", steady"),
        f"Model: spread toward the <b>{compass(w['dir'])}</b>, up to <b>{w['reach_km']:.1f} km</b> above {threshold:.0%} · peak P {w['peak']:.0%} · {w['area_above_ac']:,.0f} ac above {threshold:.0%} · <b>{w['confidence']} confidence</b>"
        + f" ({w['dir_share']:.0%} of the forecast mass in one 90° cone)",
    ]
    if "verified_px" in w and w["verified_px"]:
        rows.append(f"Verified {ev['date_next']}: <b>{w['verified_share']:.0%}</b> of the {w['verified_px']} new-fire pixels fell inside the >{threshold:.0%} zone; observed direction {compass(w['verified_dir'])} vs forecast {compass(w['dir'])}")
    return "".join(f"<div class='why'>{r}</div>" for r in rows)

# ---------------------------------------------------------------- queue

def action_line(r, w, threshold):
    p = r["worst"]; kind = r["kind"]
    when = "tonight" if p >= 0.5 else "before tomorrow's burn period"
    wind = f" Wind {w['fc_wind_mph']:.0f} mph from the {compass(w['fc_from'])} tomorrow{', veering ' + str(int(w['veer'])) + '°' if w['veer'] >= 30 else ''}." if w and p >= threshold else ""
    if kind == "substation":
        act = f"De-energize and clear defensible space {when}." if p >= threshold else "Stage a crew; re-check at the next forecast." if p >= 0.2 else "Monitor."
    elif kind == "line":
        act = f"Patrol the {r['miles_above']:.1f} mi above {threshold:.0%}; pre-position a switching crew {when}." if p >= threshold else "Patrol at first light." if p >= 0.2 else "Monitor."
    else:
        act = f"Move equipment out, pre-sell salvage ({r['salvage_months']} mo window)." if p >= threshold else "Monitor."
    return act + wind


def queue_html(events, threshold, why, n=5):
    rows = sorted(((r, ev) for ev in events for r in ev["rows"] if r["exposure_usd"] > 0), key=lambda t: -t[0]["exposure_usd"])[:n]
    if not rows: return "<div class='muted'>nothing of value in reach of the forecast</div>"
    out = []
    for i, (r, ev) in enumerate(rows, 1):
        head = f"<b>{i}. {H.escape(r['name'])}</b> <span class='muted'>{'substation' if r['kind'] == 'substation' else 'line' if r['kind'] == 'line' else 'stand'}, {r['voltage_kv']:.0f} kV</span>" if r["kind"] != "stand" else f"<b>{i}. {H.escape(r['name'])}</b>"
        ll = asset_latlon(ev, r); at = f" data-lat='{ll[0]:.5f}' data-lng='{ll[1]:.5f}'" if ll else ""
        out.append(f"<div class='q'{at} title='show on map'><div class='qh'>{head}<span class='qn'>{r['worst']:.0%} · <b>{X.usd(r['exposure_usd'])}</b></span></div>"
                   f"<div class='qa'>→ {action_line(r, why.get(ev['name']), threshold)}</div></div>")
    return "".join(out)


def asset_latlon(ev, r):
    """WGS84 point for a queue row: worst segment's midpoint for a line, the point for a substation, a representative point for a stand."""
    R = ev["raster"]
    try:
        if r["kind"] == "line": pt = max((s for s in ev["segs"] if s["properties"]["circuit"] == r["name"]), key=lambda s: s["properties"]["p_max"])["geom"].interpolate(0.5, normalized=True)
        elif r["kind"] == "substation": pt = next(s["geom"] for s in ev["subs"] if s["properties"]["name"] == r["name"])
        else: pt = next(s["geom"] for s in ev["stands"] if s["properties"]["name"] == r["name"]).representative_point()
    except (StopIteration, ValueError): return None
    lon, lat = R.to_wgs.transform(pt.x, pt.y); return lat, lon

# ---------------------------------------------------------------- live NGFS

def ngfs_realearth(products=("NGFS-SCENE-CONUS-WEST", "NGFS-SCENE-CONUS-EAST"), frames=6):
    """The public RealEarth viewer's stateless handshake (as documented by the firestorm-ngfs-data mirror), then the latest
    frames of each NGFS scene product. Returns (detections, newest_frame_iso)."""
    cj = http.cookiejar.CookieJar(); op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    ua = "Mozilla/5.0 (compatible; fireline-dashboard/1.0; +https://github.com/eitanlebras/fire-spread-forecast)"
    op.addheaders = [("User-Agent", ua), ("Accept", "*/*"), ("Accept-Encoding", "gzip, deflate")]
    op.open(f"{RE_BASE}/?products={products[0]}&view=leaflet", timeout=30).read()
    sid = next((c.value for c in cj if c.name == "PHPSESSID"), None)
    if not sid: raise RuntimeError("no PHPSESSID")
    sh = secrets.token_hex(16)
    with op.open(f"{RE_BASE}/util/session.php?sh={sh}&md5={hashlib.md5(sh.encode()).hexdigest()}", timeout=30) as r:
        raw = r.read(); raw = gzip.decompress(raw) if r.headers.get("content-encoding") == "gzip" else raw
    if raw.decode(errors="replace").strip() != sh: raise RuntimeError("handshake echo mismatch")
    def get(path):
        req = urllib.request.Request(RE_BASE + path, headers={"re-session-hash": sh, "re-session-id": sid, "re-access-key": "", "Accept": "application/json", "Accept-Encoding": "gzip, deflate", "User-Agent": ua, "Referer": RE_BASE + "/"})
        with op.open(req, timeout=90) as r:
            raw = r.read(); return json.loads(gzip.decompress(raw) if r.headers.get("content-encoding") == "gzip" else raw)
    dets, newest = {}, None
    for prod in products:
        times = (get(f"/api/products?products={prod}") or [{}])[0].get("times") or []
        for ts in [t for t in times[-frames:] if "." in t]:
            d, t = ts.split("."); iso = f"{d[:4]}-{d[4:6]}-{d[6:8]}T{t[:2]}:{t[2:4]}:{t[4:6]}Z"; newest = max(newest or "", iso)
            fc = get(f"/api/shapes?products={prod}&date={iso[:10]}&time={urllib.parse.quote(iso[11:19])}&bounds=&merge=none&notifications=false")
            for f in fc.get("features") or []:
                if not f or not f.get("geometry"): continue
                p = f.get("properties") or {}; c = f["geometry"].get("coordinates") or []
                if len(c) < 2 or not all(isinstance(v, (int, float)) for v in c[:2]): continue
                lon, lat = c[:2]
                det = {"lat": round(lat, 5), "lng": round(lon, 5), "frp": p.get("FEATURE_FRP", p.get("FRP")), "tracking_id": p.get("FEATURE_TRACKING_ID"), "acq": p.get("ACQ_DATE_TIME"),
                       "type_desc": p.get("TYPE_DESCRIPTION"), "confidence": p.get("CONFIDENCE"), "sat": "GOES-18" if "WEST" in prod else "GOES-19", "state": p.get("STATE"), "county": p.get("COUNTY"),
                       "incident_name": None if p.get("KNOWN_INCIDENT_NAME") in ("NULL", None, "") else p.get("KNOWN_INCIDENT_NAME"), "fuel": p.get("FUEL"), "land_cover": p.get("LAND_COVER")}
                key = det["tracking_id"] or f"{det['lat']},{det['lng']}"
                if key not in dets or (det["acq"] or "") > (dets[key]["acq"] or ""): dets[key] = det
            time.sleep(0.4)
    return list(dets.values()), newest


def ngfs_mirror():
    with urllib.request.urlopen(NGFS_MIRROR, timeout=60) as r: d = json.load(r)
    return d.get("detections") or [], max((d.get("newest_frame") or {}).values(), default=None)


def fetch_live():
    try:
        dets, newest = ngfs_realearth(); src = "RealEarth API (CIMSS/SSEC)"
    except Exception as e:
        print(f"NGFS RealEarth pull failed ({type(e).__name__}: {e}); using the public mirror", file=sys.stderr)
        try: dets, newest = ngfs_mirror(); src = "public mirror of the CIMSS/SSEC feed"
        except Exception as e2: print(f"NGFS mirror failed too ({e2})", file=sys.stderr); return None
    return {"detections": dets, "newest": newest, "source": src, "pulled_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

# ---------------------------------------------------------------- page

CSS = """
:root{--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;--surface:#fcfcfb;--page:#f9f9f7;--orange:#d85a30;--blue:#2a78d6}
html,body{height:100%;margin:0;background:var(--page);color:var(--ink);font:13px/1.45 -apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif}
#fl-head{position:fixed;top:0;left:0;right:0;height:56px;z-index:1200;background:var(--surface);border-bottom:1px solid var(--line);display:flex;align-items:center;gap:22px;padding:0 18px;box-sizing:border-box}
#fl-head .brand{display:flex;align-items:center;gap:10px;font-weight:600;font-size:17px;letter-spacing:-.2px}
#fl-head .brand svg{width:34px;height:34px}
#fl-head .meta{color:var(--ink2);font-size:13px}
#fl-head .meta b{color:var(--ink);font-weight:600}
#fl-head .live{margin-left:auto;display:flex;align-items:center;gap:8px;color:var(--ink2);font-size:12px}
#fl-head .dot{width:8px;height:8px;border-radius:50%;background:#1baf7a;box-shadow:0 0 0 3px rgba(27,175,122,.18)}
#fl-map-wrap{position:fixed;top:56px;left:0;width:65%;bottom:0}
#fl-side{position:fixed;top:56px;right:0;width:35%;bottom:0;overflow-y:auto;background:var(--page);border-left:1px solid var(--line);padding:14px 18px 24px;box-sizing:border-box}
.fl-panel{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin-bottom:12px}
.fl-panel h2{font:600 11px/1 -apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:0 0 10px}
.fl-panel h2 span{float:right;letter-spacing:0;text-transform:none;font-weight:400}
.muted{color:var(--muted)} .why{padding:5px 0;border-top:1px solid var(--line)} .why:first-child{border-top:0;padding-top:0}
.q{padding:7px 0;border-top:1px solid var(--line)} .q[data-lat]{cursor:pointer} .q[data-lat]:hover{background:#f3f2ee;margin:0 -8px;padding-left:8px;padding-right:8px} .q.on{background:#eef4fc;margin:0 -8px;padding-left:8px;padding-right:8px} .q:first-child{border-top:0;padding-top:0}
.qh{display:flex;justify-content:space-between;gap:8px;align-items:baseline} .qn{white-space:nowrap;color:var(--ink2)}
.qa{color:var(--ink2);margin-top:2px;padding-left:14px;text-indent:-14px}
.tr{display:grid;grid-template-columns:1fr 1fr;gap:8px 14px}
.tr .n{font:600 22px/1.1 Georgia,serif;letter-spacing:-.3px} .tr .l{color:var(--ink2);font-size:12px} .tr .c{color:var(--muted);font-size:11px}
.lv{padding:6px 0;border-top:1px solid var(--line);display:flex;justify-content:space-between;gap:8px} .lv:first-child{border-top:0;padding-top:0}
.lv .nm b{font-weight:600} .lv .fr{white-space:nowrap;color:var(--ink2)}
#fl-foot{position:absolute;left:12px;bottom:12px;z-index:1000;background:rgba(252,252,251,.92);padding:5px 9px;border-radius:4px;font-size:11px;color:var(--ink2)}
.leaflet-control-layers{font-size:12px}
.leaflet-top.leaflet-left{top:0}
@media (max-width:900px){ #fl-map-wrap{position:static;width:100%;height:60vh;margin-top:56px}#fl-side{position:static;width:100%}#fl-head .meta.sec{display:none}}
"""

LOGO = ('<svg viewBox="0 0 120 120" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><path d="M58.5 45.6 A 16 16 0 1 1 58.5 74.4" stroke="#8A8A85" stroke-width="5" stroke-linecap="round"/>'
        '<path d="M54.6 34.7 A 27.5 27.5 0 1 1 54.6 85.3" stroke="#8A8A85" stroke-width="5" stroke-linecap="round"/><path d="M50.6 23.8 A 39 39 0 1 1 50.6 96.2" stroke="#D85A30" stroke-width="8" stroke-linecap="round"/></svg>')


def focus_bounds(events, pad_km=6.0):
    """[[S, W], [N, E]] around today's fire + the forecast footprint (P > 0.05) + the verification outline, not the whole tile."""
    from scipy import ndimage as ndi
    S_, W_, N_, E_ = 90.0, 180.0, -90.0, -180.0
    for ev in events:
        g = Grid(ev["bbox"], *ev["mask"].shape); on = ev["mask"] | (ev["prob"] > 0.05)
        if ev.get("new_next") is not None: on |= ev["new_next"]
        lab, n = ndi.label(ndi.binary_dilation(ev["mask"], iterations=3))          # detached fragments far from the main body are left out of the frame
        if n > 1:
            big = 1 + int(np.argmax(ndi.sum(ev["mask"], lab, range(1, n + 1)))); cy, cx = ndi.center_of_mass(lab == big)
            yy, xx = np.mgrid[:on.shape[0], :on.shape[1]]; on &= np.hypot((yy - cy) * g.km_per_px, (xx - cx) * g.km_per_px * math.cos(math.radians(g.center[0]))) <= 15.0
        rr, cc = np.nonzero(on)
        if not rr.size: continue
        (n, w), (s, e) = g.to_ll(rr.min(), cc.min()), g.to_ll(rr.max(), cc.max())
        S_, W_, N_, E_ = min(S_, s), min(W_, w), max(N_, n), max(E_, e)
    if S_ > N_: return [[min(e["bbox"][1] for e in events), min(e["bbox"][0] for e in events)], [max(e["bbox"][3] for e in events), max(e["bbox"][2] for e in events)]]
    dlat = pad_km / 111.32; dlon = dlat / max(math.cos(math.radians((S_ + N_) / 2)), 1e-6)
    return [[S_ - dlat, W_ - dlon], [N_ + dlat, E_ + dlon]]


def build(events, network, why, threshold, territory, n_active, live, live_state, wind, out):
    m = folium.Map(location=(np.mean([e["bbox"][1] + e["bbox"][3] for e in events]) / 2, np.mean([e["bbox"][0] + e["bbox"][2] for e in events]) / 2), zoom_start=10, tiles=None, control_scale=True)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", name="Satellite", attr="Esri, Maxar, Earthstar Geographics").add_to(m)
    folium.TileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", name="Terrain", attr="© OpenStreetMap contributors, SRTM | © OpenTopoMap (CC-BY-SA)", max_zoom=17).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Streets").add_to(m)
    folium.TileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/Elevation/World_Hillshade/MapServer/tile/{z}/{y}/{x}", name="Hillshade", attr="Esri", overlay=True, opacity=0.5, show=False).add_to(m)
    # water, from the WFTS land-cover band (class 17), in its own colour
    fg_water = folium.FeatureGroup(name="Water", show=True)
    for ev in events:
        wm = ev.get("water")
        if wm is None or not wm.any(): continue
        rings = mask_polygons(wm, Grid(ev["bbox"], *wm.shape), smooth=0.6)
        if rings: folium.GeoJson({"type": "Feature", "properties": {}, "geometry": {"type": "MultiPolygon", "coordinates": [[r] for r in rings]}},
                                 style_function=lambda _: {"color": P.WATER, "weight": 0.8, "opacity": 0.9, "fillColor": P.WATER, "fillOpacity": 0.55}, interactive=False).add_to(fg_water)
    fg_water.add_to(m)
    if wind:
        fg_wind = folium.FeatureGroup(name="Wind — HRRR 10 m", show=False); scale = 0.004
        for lat, lon, u, v in wind:
            spd = math.hypot(u, v); deg = math.degrees(math.atan2(u, v)) % 360; lat2, lon2 = lat + v * scale, lon + u * scale / max(math.cos(math.radians(lat)), 1e-6)
            folium.PolyLine([(lat, lon), (lat2, lon2)], color=P.INK_2, weight=2, opacity=0.8, tooltip=f"wind {spd * 2.237:.0f} mph toward {compass(deg)}").add_to(fg_wind)
            folium.RegularPolygonMarker((lat2, lon2), number_of_sides=3, radius=5, rotation=deg - 90, color=P.INK_2, fill_color=P.INK_2, fill_opacity=0.9, weight=1).add_to(fg_wind)
        fg_wind.add_to(m)
    fg_fire = folium.FeatureGroup(name="Burning now", show=True)
    for ev in events:
        rings = mask_polygons(ev["mask"], Grid(ev["bbox"], *ev["mask"].shape))
        if rings: folium.GeoJson({"type": "Feature", "properties": {}, "geometry": {"type": "MultiPolygon", "coordinates": [[r] for r in rings]}},
                                 style_function=lambda _: {"color": P.INK, "weight": 1.5, "opacity": 0.95, "fillColor": P.BURNING, "fillOpacity": 0.7},
                                 tooltip=f"{ev['name']} · burning {ev['date']} · {ev['n_today']} px (VIIRS)").add_to(fg_fire)
    fg_fire.add_to(m)
    fg_prob = folium.FeatureGroup(name="Forecast — next 24 hours", show=True)
    for ev in events:
        folium.raster_layers.ImageOverlay(image="data:image/png;base64," + base64.b64encode(prob_png(ev["prob"])).decode(), bounds=Grid(ev["bbox"], 1, 1).bounds, opacity=1.0, zindex=3, interactive=False).add_to(fg_prob)
    fg_prob.add_to(m)
    fg_next = folium.FeatureGroup(name="Verification — what actually burned", show=True)
    for ev in events:
        nn = ev.get("new_next")
        if nn is None or not nn.any(): continue
        rings = mask_polygons(nn, Grid(ev["bbox"], *nn.shape))
        if rings: folium.GeoJson({"type": "Feature", "properties": {}, "geometry": {"type": "MultiLineString", "coordinates": rings}},
                                 style_function=lambda _: {"color": P.ORANGE, "weight": 2.5, "opacity": 0.95}, tooltip=f"{ev['name']} · new fire observed {ev['date_next']} · {ev['n_new_next']} px").add_to(fg_next)
    fg_next.add_to(m)
    lines, subs, stands = network
    customers = [c for c in CUSTOMERS if c == "utility" and (lines or subs) or c == "timber" and stands]
    groups = {c: add_assets(m, events, c, threshold, network) for c in customers}
    # live NGFS detections (points; the layer is refreshed in the browser)
    fg_live = folium.FeatureGroup(name="Live — NGFS detections (GOES)", show=True); fg_live.add_to(m)
    folium.LayerControl(collapsed=True, position="topright").add_to(m); plugins.Fullscreen(position="topleft").add_to(m)
    fit = focus_bounds(events); m.fit_bounds(fit)

    # ---- sidebar content
    total = sum(r["exposure_usd"] for ev in events for r in ev["rows"]); n_exposed = sum(1 for ev in events if any(r["exposure_usd"] > 0 and r["worst"] >= threshold for r in ev["rows"]))
    ev0 = events[0]; w0 = why.get(ev0["name"]) or {}
    tr = TRACK_RECORD
    track = (f"<div class='tr'><div><div class='n'>{tr['advance_share']:.0%}</div><div class='l'>of advancing fire inside our >{threshold:.0%} zone</div><div class='c'>{tr['advance_x_chance']}× chance</div></div>"
             f"<div><div class='n'>{tr['ece'] * 100:.1f}%</div><div class='l'>calibration error (ECE)</div><div class='c'>a 40% pixel burns about 40% of the time</div></div>"
             f"<div><div class='n'>{tr['aucpr']:.3f}</div><div class='l'>AUC-PR, next-day fire</div><div class='c'>persistence {tr['persistence']:.3f}</div></div>"
             f"<div><div class='n'>{tr['interior_aucpr']:.3f}</div><div class='l'>AUC-PR, interior fill</div><div class='c'>chance {tr['interior_chance']:.3f}</div></div></div>"
             f"<div class='muted' style='margin-top:8px;font-size:11px'>{tr['fires']} held-out fires, WildfireSpreadTS 2021 test split, 3 seeds. Detached spot fires: {tr['spot_hits']} predicted — never read a blue lobe as a spot-fire call.</div>")
    live_json = json.dumps(live or {"detections": [], "newest": None, "source": None, "pulled_utc": None})
    dates = f"today {ev0['date']} → forecast {ev0['date_next']}"
    side = (f"<div class='fl-panel'><h2>Tonight's queue <span>{dates}</span></h2>{queue_html(events, threshold, why)}"
            f"<div class='muted' style='margin-top:8px;font-size:11px'>exposure = P(burn) × replacement value × vulnerability × exposed miles · HIFLD public infrastructure, illustrative values</div></div>"
            f"<div class='fl-panel'><h2>Why this ranks here <span>{H.escape(ev0['name'])}</span></h2>{why_html(ev0, w0, threshold)}</div>"
            f"<div class='fl-panel'><h2>Burning right now <span id='fl-live-when'>NGFS · loading</span></h2><div id='fl-live'></div>"
            f"<div class='muted' style='margin-top:8px;font-size:11px'>GOES-18/19 scene detections from NOAA's Next Generation Fire System (CIMSS/SSEC), grouped by tracked fire, {live_state} only, refreshed every 5 min. "
            f"The forecast above is a replay of a held-out 2021 fire; running the model on these live fires needs the WFTS input stack (VIIRS, GridMET, GFS, terrain, land cover), not wired yet.</div></div>")
    head = (f"<div id='fl-head'><div class='brand'>{LOGO}Fireline</div>"
            f"<div class='meta'><b>{H.escape(territory)}</b> · <span id='fl-clock'>--:-- UTC</span> · next refresh <span id='fl-next'>--:--</span></div>"
            f"<div class='meta sec'><b>{n_active}</b> active fires in the record that day · <b>{n_exposed}</b> with asset exposure · <b>{X.usd(total)}</b> total in queue</div>"
            f"<div class='live'><span class='dot'></span><span id='fl-live-n'>NGFS live</span></div></div>")
    foot = (f"<div id='fl-foot'>{MODEL} · {dates} · <span style='display:inline-block;width:70px;height:9px;vertical-align:middle;background:{P.PROB_CMAP_CSS}'></span> P(burn in 24 h) 0 → 1, within {BUFFER_KM:g} km of today's fire, growth region only · "
            f"<span style='display:inline-block;width:10px;height:10px;vertical-align:middle;background:{P.WATER}'></span> water · <span style='display:inline-block;width:10px;height:10px;vertical-align:middle;background:{P.BURNING}'></span> burning now · "
            f"<span style='display:inline-block;width:18px;height:3px;vertical-align:middle;background:{P.ORANGE}'></span> burned next day · data: HIFLD, VIIRS (WFTS), GridMET/GFS, NGFS</div>")
    js_groups = {c: [g.get_name() for g in grp] for c, grp in groups.items()}
    js = f"""
    document.addEventListener("DOMContentLoaded", function () {{
      var map = {m.get_name()}, liveGroup = {fg_live.get_name()}, LIVE = {live_json}, STATE = {json.dumps(live_state)};
      var wrap = document.getElementById('fl-map-wrap'); wrap.appendChild(document.getElementById('{m.get_name()}')); wrap.appendChild(document.getElementById('fl-foot'));
      map.invalidateSize(); map.fitBounds({json.dumps(fit)}, {{padding: [20, 20]}});
      var groups = {{{", ".join(f"{c}: [{', '.join(n)}]" for c, n in js_groups.items())}}}; for (var k in groups) groups[k].forEach(function (g) {{ if (k !== 'utility' && map.hasLayer(g)) map.removeLayer(g); }});
      var pins = groups.utility ? groups.utility[1] : null;
      document.querySelectorAll('#fl-side .q[data-lat]').forEach(function (el) {{ el.onclick = function () {{
        var lat = +el.dataset.lat, lng = +el.dataset.lng; document.querySelectorAll('#fl-side .q').forEach(function (e) {{ e.classList.remove('on'); }}); el.classList.add('on');
        map.flyTo([lat, lng], 13, {{duration: 0.8}});
        if (pins) pins.eachLayer(function (l) {{ if (l.getLatLng && Math.abs(l.getLatLng().lat - lat) < 1e-4 && Math.abs(l.getLatLng().lng - lng) < 1e-4) setTimeout(function () {{ l.openPopup(); }}, 900); }});
      }}; }});
      function pad(n) {{ return (n < 10 ? '0' : '') + n; }}
      function clock() {{ var d = new Date(); document.getElementById('fl-clock').textContent = pad(d.getUTCHours()) + ':' + pad(d.getUTCMinutes()) + ' UTC';
        var nx = new Date(Math.ceil((d.getTime() + 1) / 300000) * 300000); document.getElementById('fl-next').textContent = pad(nx.getUTCHours()) + ':' + pad(nx.getUTCMinutes()); }}
      clock(); setInterval(clock, 1000);
      function ago(iso) {{ if (!iso) return ''; var s = (Date.now() - Date.parse(iso)) / 1000; return s < 90 ? 'just now' : s < 5400 ? Math.round(s / 60) + ' min ago' : Math.round(s / 3600) + ' h ago'; }}
      function render(live) {{
        var dets = (live.detections || []).filter(function (d) {{ return !STATE || d.state === STATE; }});
        liveGroup.clearLayers(); var fires = {{}};
        dets.forEach(function (d) {{
          var key = d.incident_name || d.tracking_id || (d.lat + ',' + d.lng); var f = fires[key] || (fires[key] = {{name: d.incident_name, county: d.county, frp: 0, n: 0, acq: '', lat: d.lat, lng: d.lng, type: d.type_desc, fuel: d.fuel}});
          f.frp += (d.frp || 0); f.n += 1; if ((d.acq || '') > f.acq) f.acq = d.acq;
          L.circleMarker([d.lat, d.lng], {{radius: 6, color: '#fff', weight: 1, fillColor: '{P.ORANGE}', fillOpacity: 0.9}}).bindTooltip((d.incident_name || 'untracked fire') + ' · ' + (d.county || '') + ' · FRP ' + Math.round(d.frp || 0) + ' MW · ' + (d.acq || '').slice(11, 16) + ' UTC · ' + d.sat).addTo(liveGroup);
        }});
        var list = Object.keys(fires).map(function (k) {{ return fires[k]; }}).sort(function (a, b) {{ return b.frp - a.frp; }});
        document.getElementById('fl-live-n').textContent = list.length + ' fire' + (list.length === 1 ? '' : 's') + ' burning now in ' + STATE + ' (NGFS)';
        document.getElementById('fl-live-when').textContent = 'NGFS · ' + (live.newest ? live.newest.slice(11, 16) + ' UTC, ' + ago(live.newest) : 'no data');
        document.getElementById('fl-live').innerHTML = list.length ? list.slice(0, 8).map(function (f) {{
          return "<div class='lv'><div class='nm'><b>" + (f.name || 'untracked fire') + "</b> <span class='muted'>" + (f.county || '') + (f.name ? '' : ' · ' + (f.type || '')) + "</span><div class='muted' style='font-size:11px'>" + (f.fuel ? 'fuel ' + f.fuel.split(',').slice(0, 2).join(', ') : '') + "</div></div><div class='fr'>" + Math.round(f.frp) + " MW<br><span class='muted' style='font-size:11px'>" + f.n + " px · " + (f.acq || '').slice(11, 16) + " UTC</span></div></div>"; }}).join('')
          : "<div class='muted'>no GOES detections in " + STATE + " in the last frames</div>";
        var first = list[0]; if (first) {{ document.getElementById('fl-live').querySelectorAll('.lv').forEach(function (el, i) {{ el.style.cursor = 'pointer'; el.onclick = function () {{ map.setView([list[i].lat, list[i].lng], 11); }}; }}); }}
      }}
      render(LIVE);
      function refresh() {{ fetch({json.dumps(NGFS_MIRROR)} + '?t=' + Date.now()).then(function (r) {{ return r.json(); }}).then(function (d) {{ render({{detections: d.detections || [], newest: Object.values(d.newest_frame || {{}}).sort().pop() || null}}); }}).catch(function () {{}}); }}
      refresh(); setInterval(refresh, 300000);
    }});
    """
    m.get_root().header.add_child(folium.Element(f"<title>Fireline · {H.escape(territory)}</title><style>{CSS}</style>"))
    m.get_root().html.add_child(folium.Element(head + "<div id='fl-map-wrap'></div><div id='fl-side'>" + side + "</div>" + foot))
    m.get_root().script.add_child(folium.Element(js))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True); m.save(out)
    return total, n_exposed


def count_active(root, date):
    n = 0
    import rasterio
    for tif in glob.glob(os.path.join(root, "*", f"{date}.tif")):
        with rasterio.open(tif) as ds: n += bool((np.nan_to_num(ds.read(ds.count), nan=0.0) > 0).any())
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event", action="append", required=True, help="NAME=FIRE_DIR[:DAY] (repeatable)"); ap.add_argument("--probs", action="append", default=[], help="NAME=P.npy")
    ap.add_argument("--lines"); ap.add_argument("--substations"); ap.add_argument("--stands"); ap.add_argument("--fetch-hifld", action="store_true"); ap.add_argument("--wind", help="CSV lat,lon,u,v (real HRRR only; no placeholder)")
    ap.add_argument("--threshold", type=float, default=X.THRESHOLD); ap.add_argument("--territory", default="Shasta County, CA"); ap.add_argument("--active-root", help="WFTS year dir to count fires active on the day")
    ap.add_argument("--no-live", action="store_true"); ap.add_argument("--live-state", default="CA"); ap.add_argument("--out", default="out/dashboard/index.html")
    a = ap.parse_args(argv)
    probs = dict(s.split("=", 1) for s in a.probs); events, why = [], {}
    for spec in a.event:
        name, rest = spec.split("=", 1); src, _, day = rest.partition(":"); day = int(day) if day else None
        if name not in probs: sys.exit(f"{name}: --probs is required (no placeholder forecasts on the dashboard)")
        ev = load_event(name, src, day, probs[name]); events.append(ev)
        masks, dates, _ = load_fire(src); T = len(masks); day = T - 2 if day is None else day; pr = align_probs(np.load(probs[name]), T, masks.shape[1:])
        if os.path.isdir(src):
            why[name] = explain(src, day, pr, masks, a.threshold)
            water = why[name].pop("water")
            if water.any():                                                            # warp the water mask like the others
                import rasterio
                from rasterio.warp import reproject, Resampling, calculate_default_transform
                with rasterio.open(sorted(glob.glob(os.path.join(src, "*.tif")))[0]) as ds:
                    if ds.crs.to_epsg() == 4326: ev["water"] = water
                    else:
                        tr, w, h = calculate_default_transform(ds.crs, "EPSG:4326", ds.width, ds.height, *ds.bounds); dst = np.zeros((h, w), np.float32)
                        reproject(water.astype(np.float32), dst, src_transform=ds.transform, src_crs=ds.crs, dst_transform=tr, dst_crs="EPSG:4326", resampling=Resampling.nearest); ev["water"] = dst > 0.5
    bbox = (min(e["bbox"][0] for e in events) - 0.1, min(e["bbox"][1] for e in events) - 0.1, max(e["bbox"][2] for e in events) + 0.1, max(e["bbox"][3] for e in events) + 0.1)
    if a.fetch_hifld: lines, subs = X.fetch_hifld("lines", bbox), X.fetch_hifld("substations", bbox)
    else: lines, subs = (X.load_features(a.lines)[0] if a.lines else None), (X.load_features(a.substations)[0] if a.substations else None)
    stands = X.load_features(a.stands)[0] if a.stands else None
    for ev in events:
        res = X.assess(ev["raster"], lines, subs, stands, "EPSG:4326", a.threshold); ev.update(res)
        w = why.get(ev["name"], {})
        print(f"{ev['name']}: peak P {ev['prob'].max():.2f}; dir {compass(w['dir']) if w else '?'} reach {w.get('reach_km', 0):.1f} km; top: " + ("; ".join(f"{r['name']} {X.usd(r['exposure_usd'])}" for r in res["rows"][:3]) or "nothing in reach"))
    n_active = count_active(a.active_root, events[0]["date"]) if a.active_root else len(events)
    live = None if a.no_live else fetch_live()
    if live: print(f"NGFS: {len(live['detections'])} tracked detections via {live['source']}, newest frame {live['newest']}; {sum(d.get('state') == a.live_state for d in live['detections'])} in {a.live_state}")
    wind = load_wind_csv(a.wind) if a.wind else None
    total, n_exposed = build(events, (lines, subs, stands), why, a.threshold, a.territory, n_active, live, a.live_state, wind, a.out)
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB): {n_active} active fires that day, {n_exposed} with exposure, {X.usd(total)} in queue")


if __name__ == "__main__":
    main()
