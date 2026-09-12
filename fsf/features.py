"""Derived features from NDWS channels. All functions take/return numpy arrays of shape (N, H, W)."""
import numpy as np

PIXEL_M = 1000.0  # NDWS is 1 km / px


def slope_aspect(elev):
    """Slope (rise/run, unitless) and downhill unit vector (ax, ay) in image coords (x=cols, y=rows)."""
    dzdy, dzdx = np.gradient(elev, PIXEL_M, axis=(1, 2))
    slope = np.sqrt(dzdx ** 2 + dzdy ** 2)
    n = np.maximum(slope, 1e-9)
    return slope.astype(np.float32), (-dzdx / n).astype(np.float32), (-dzdy / n).astype(np.float32)


def wind_uv(th_deg, vs):
    """gridMET th = direction wind blows FROM, degrees clockwise from north. Return (u, v) the wind blows TOWARD,
    in image coords: u = +x (east, cols), v = +y (south, rows)."""
    to = np.deg2rad(th_deg + 180.0)
    u = vs * np.sin(to)          # eastward
    v = -vs * np.cos(to)         # northward -> negative rows
    return u.astype(np.float32), v.astype(np.float32)


def wind_slope_alignment(u, v, slope, ax, ay):
    """cos(angle between wind-to vector and UPHILL direction) * slope. Uphill = -(downhill unit vector).
    Positive = wind pushing fire upslope (the blow-up condition)."""
    spd = np.maximum(np.sqrt(u ** 2 + v ** 2), 1e-6)
    cos = (u * (-ax) + v * (-ay)) / spd
    return (cos * slope).astype(np.float32)


def relative_humidity(sph, t_k, elev_m):
    """RH [%] from specific humidity (kg/kg), temperature (K), elevation (m) via standard-atmosphere pressure."""
    p = 1013.25 * (1.0 - 2.25577e-5 * np.clip(elev_m, 0, 8000)) ** 5.25588  # hPa
    q = np.clip(sph, 1e-6, 0.05)
    e = q * p / (0.622 + 0.378 * q)
    tc = np.clip(t_k, 230.0, 330.0) - 273.15   # NDWS has garbage outliers (tmmx up to 1230 K)
    es = 6.112 * np.exp(17.67 * tc / (tc + 243.5))
    return np.clip(100.0 * e / es, 0.0, 100.0).astype(np.float32)


def fosberg_emc(rh, t_k):
    """Fosberg (1978) equilibrium moisture content [%], T in F, RH in %."""
    tf = (np.clip(t_k, 230.0, 330.0) - 273.15) * 9.0 / 5.0 + 32.0
    emc = np.where(rh < 10, 0.03229 + 0.281073 * rh - 0.000578 * rh * tf,
          np.where(rh < 50, 2.22749 + 0.160107 * rh - 0.01478 * tf,
                   21.0606 + 0.005565 * rh ** 2 - 0.00035 * rh * tf - 0.483199 * rh))
    return np.clip(emc, 0.0, 40.0).astype(np.float32)


def dead_fuel_moisture(sph, tmmx, elev):
    """1h and 10h dead fuel moisture [%] from Fosberg EMC at daily max temp (driest case).
    NFDRS-style multipliers: 1h ~ 1.03*EMC, 10h ~ 1.28*EMC."""
    rh = relative_humidity(sph, tmmx, elev)
    emc = fosberg_emc(rh, tmmx)
    return (1.03 * emc).astype(np.float32), (1.28 * emc).astype(np.float32), rh
