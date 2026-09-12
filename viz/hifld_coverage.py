"""Which WFTS fires have HIFLD infrastructure near them? Run on the pod (needs the fire manifest + network).

  python -m viz.hifld_coverage /root/data/wfts/cache/wfts_fires.json [--split test] [--pad-km 5] [--top 15]

Reads the manifest that fsf.wfts cache writes (bbox_wgs84 per fire), pads each bbox, and asks the HIFLD services how many
transmission lines and substations intersect it (returnCountOnly, one cheap request per layer per fire). Prints the fires
ranked by substations then lines, so the demo can be pointed at an event that actually has a grid to expose.
Remote wilderness events return 0/0 and would give an empty asset layer.
"""
import argparse, json, sys, urllib.parse, urllib.request
from viz.exposure import HIFLD


def count(layer, bbox):
    q = {"where": "1=1", "geometry": ",".join(f"{b:.5f}" for b in bbox), "geometryType": "esriGeometryEnvelope", "inSR": "4326",
         "spatialRel": "esriSpatialRelIntersects", "returnCountOnly": "true", "f": "json"}
    with urllib.request.urlopen(HIFLD[layer] + "?" + urllib.parse.urlencode(q), timeout=60) as r: return json.load(r).get("count", -1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest"); ap.add_argument("--split", default="test"); ap.add_argument("--pad-km", type=float, default=5.0); ap.add_argument("--top", type=int, default=15)
    a = ap.parse_args(argv)
    fires = [f for f in json.load(open(a.manifest)) if a.split in ("all", f.get("split"))]
    pad = a.pad_km / 111.0; rows = []
    for f in fires:
        w, s, e, n = f["bbox_wgs84"]; bb = (w - pad, s - pad, e + pad, n + pad)
        try: nl, ns = count("lines", bb), count("substations", bb)
        except Exception as ex: nl = ns = -1; print(f"  {f['year']}/{f['fire']}: {ex}", file=sys.stderr)
        rows.append((ns, nl, f)); print(f"  {f['year']}/{f['fire']:<16} days={len(f['dates']):>2} {f['H']}x{f['W']}  lines={nl:>3} substations={ns:>3}", flush=True)
    rows.sort(key=lambda t: (-t[0], -t[1]))
    print(f"\n{a.split} split: {sum(1 for r in rows if r[0] > 0)}/{len(rows)} fires have a substation within {a.pad_km:g} km of the raster; top {a.top}:")
    for ns, nl, f in rows[:a.top]: print(f"  {f['year']}/{f['fire']:<16} substations={ns:>3} lines={nl:>3}  days={len(f['dates'])}  centre=({(f['bbox_wgs84'][1] + f['bbox_wgs84'][3]) / 2:.2f}, {(f['bbox_wgs84'][0] + f['bbox_wgs84'][2]) / 2:.2f})")


if __name__ == "__main__":
    main()
