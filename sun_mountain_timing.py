#!/usr/bin/env python3
"""Estimate when the sun slips behind the mountains before sunset.

For an observer location and a date the program

  1. computes the sun's azimuth/altitude through the afternoon (NOAA / Meeus
     low-precision solar position, plus atmospheric refraction),
  2. builds the terrain horizon seen from the location by ray-marching a
     digital elevation model (AWS "Terrarium" tiles, ~10 m data in the US)
     across the azimuths the sun covers, including earth curvature and
     refraction of the line of sight,
  3. finds every time the sun's disc crosses that skyline: "goes behind" and
     "reappears" events, and the final loss of direct sunlight.

The terrain is whatever the DEM contains (bare earth: no trees or buildings),
so the skyline is exact for mountains such as Mount Timpanogos and the
Roberts Horn cluster but ignores anything closer, like trees on your lot.

Dependencies: numpy, Pillow (matplotlib only for --plot).  Needs internet on
the first run to download terrain tiles; they are cached afterwards.
"""

import argparse
import math
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image

EARTH_R = 6371008.8  # mean earth radius, m
TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
SUN_SEMIDIAMETER = 0.2666  # degrees, mean
FT_PER_M = 3.280839895

# Named summits (SummitPost / USGS positions; the DEM agrees to a few feet).
# Used only to label the ridge point that blocks the sun - add your own.
PEAKS = [
    # name, lat, lon
    ("Mount Timpanogos", 40.3909, -111.6460),
    ("Roberts Horn", 40.39938, -111.63732),
]
PEAK_NEAR_M = 400    # closer than this: "at <peak>"
PEAK_MATCH_M = 3500  # farther than this: unlabeled


# --------------------------------------------------------------------------
# Sun
# --------------------------------------------------------------------------

def solar_position(unix_t, lat, lon):
    """Return (azimuth, true altitude) in degrees for unix times (array).

    Azimuth is measured clockwise from north.  NOAA solar calculator
    equations (Meeus); accurate to roughly 0.01 degree.
    """
    unix_t = np.asarray(unix_t, dtype=float)
    jd = unix_t / 86400.0 + 2440587.5
    T = (jd - 2451545.0) / 36525.0

    L0 = (280.46646 + T * (36000.76983 + T * 0.0003032)) % 360.0
    M = 357.52911 + T * (35999.05029 - 0.0001537 * T)
    e = 0.016708634 - T * (0.000042037 + 0.0000001267 * T)
    Mr = np.radians(M)
    C = (np.sin(Mr) * (1.914602 - T * (0.004817 + 0.000014 * T))
         + np.sin(2 * Mr) * (0.019993 - 0.000101 * T)
         + np.sin(3 * Mr) * 0.000289)
    true_long = L0 + C
    omega = 125.04 - 1934.136 * T
    lam = np.radians(true_long - 0.00569 - 0.00478 * np.sin(np.radians(omega)))
    eps0 = 23.0 + (26.0 + (21.448 - T * (46.815 + T * (0.00059 - T * 0.001813))) / 60.0) / 60.0
    eps = np.radians(eps0 + 0.00256 * np.cos(np.radians(omega)))
    decl = np.arcsin(np.sin(eps) * np.sin(lam))

    y = np.tan(eps / 2.0) ** 2
    L0r = np.radians(L0)
    eqtime = 4.0 * np.degrees(
        y * np.sin(2 * L0r) - 2 * e * np.sin(Mr)
        + 4 * e * y * np.sin(Mr) * np.cos(2 * L0r)
        - 0.5 * y * y * np.sin(4 * L0r) - 1.25 * e * e * np.sin(2 * Mr))

    utc_min = ((jd + 0.5) % 1.0) * 1440.0
    tst = (utc_min + eqtime + 4.0 * lon) % 1440.0
    ha = np.radians(tst / 4.0 - 180.0)

    latr = math.radians(lat)
    sin_alt = math.sin(latr) * np.sin(decl) + math.cos(latr) * np.cos(decl) * np.cos(ha)
    alt = np.degrees(np.arcsin(np.clip(sin_alt, -1, 1)))
    az = (np.degrees(np.arctan2(np.sin(ha), np.cos(ha) * math.sin(latr)
                                - np.tan(decl) * math.cos(latr))) + 180.0) % 360.0
    return az, alt


def refraction_deg(true_alt, pressure_kpa, temp_c):
    """Atmospheric refraction (degrees) for a true altitude (Saemundsson)."""
    h = np.maximum(true_alt, -1.5)
    r_arcmin = 1.02 / np.tan(np.radians(h + 10.3 / (h + 5.11)))
    r_arcmin = r_arcmin * (pressure_kpa / 101.0) * (283.0 / (273.0 + temp_c))
    return r_arcmin / 60.0


def standard_pressure_kpa(elev_m):
    return 101.325 * (1.0 - 2.25577e-5 * elev_m) ** 5.25588


# --------------------------------------------------------------------------
# Terrain
# --------------------------------------------------------------------------

def lonlat_to_pixel(lat, lon, zoom):
    """Global web-mercator pixel coordinates (float) at a zoom level."""
    n = 256.0 * 2 ** zoom
    x = (np.asarray(lon) + 180.0) / 360.0 * n
    y = (1.0 - np.arcsinh(np.tan(np.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def fetch_tile(zoom, tx, ty, cache_dir):
    path = os.path.join(cache_dir, str(zoom), str(tx), f"{ty}.png")
    if not os.path.exists(path):
        url = TILE_URL.format(z=zoom, x=tx, y=ty)
        last = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    data = r.read()
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path + ".part", "wb") as f:
                    f.write(data)
                os.replace(path + ".part", path)
                break
            except Exception as exc:  # network hiccup: back off and retry
                last = exc
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"could not download terrain tile {url}: {last}")
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    return rgb[..., 0] * 256.0 + rgb[..., 1] + rgb[..., 2] / 256.0 - 32768.0


class Dem:
    """Elevation mosaic built from just the tiles a set of points touches."""

    def __init__(self, zoom, cache_dir):
        self.zoom = zoom
        self.cache_dir = cache_dir
        self.mosaic = None

    def tiles_needed(self, lat, lon):
        x, y = lonlat_to_pixel(lat, lon, self.zoom)
        x0, y0 = np.floor(x - 0.5).astype(np.int64), np.floor(y - 0.5).astype(np.int64)
        tiles = set()
        for dx in (0, 1):
            for dy in (0, 1):
                tx, ty = (x0 + dx) // 256, (y0 + dy) // 256
                tiles.update(zip(*(a.tolist() for a in np.unique(np.stack([tx, ty]), axis=1))))
        return tiles

    def load(self, tiles):
        tiles = sorted(tiles)
        txs, tys = [t[0] for t in tiles], [t[1] for t in tiles]
        self.tx0, self.ty0 = min(txs), min(tys)
        w, h = max(txs) - self.tx0 + 1, max(tys) - self.ty0 + 1
        self.mosaic = np.full((h * 256, w * 256), np.nan, dtype=np.float32)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda t: fetch_tile(self.zoom, t[0], t[1], self.cache_dir), tiles))
        for (tx, ty), arr in zip(tiles, results):
            r, c = (ty - self.ty0) * 256, (tx - self.tx0) * 256
            self.mosaic[r:r + 256, c:c + 256] = arr

    def sample(self, lat, lon):
        """Bilinear elevation (m) at lat/lon arrays."""
        x, y = lonlat_to_pixel(lat, lon, self.zoom)
        px = x - 0.5 - self.tx0 * 256
        py = y - 0.5 - self.ty0 * 256
        ix, iy = np.floor(px).astype(np.int64), np.floor(py).astype(np.int64)
        fx, fy = (px - ix).astype(np.float32), (py - iy).astype(np.float32)
        m = self.mosaic
        top = m[iy, ix] * (1 - fx) + m[iy, ix + 1] * fx
        bot = m[iy + 1, ix] * (1 - fx) + m[iy + 1, ix + 1] * fx
        return top * (1 - fy) + bot * fy


def destination(lat, lon, bearing_deg, dist_m):
    """Great-circle destination points; inputs broadcast, outputs in degrees."""
    d = dist_m / EARTH_R
    th = np.radians(bearing_deg)
    la1, lo1 = math.radians(lat), math.radians(lon)
    la2 = np.arcsin(np.sin(la1) * np.cos(d) + np.cos(la1) * np.sin(d) * np.cos(th))
    lo2 = lo1 + np.arctan2(np.sin(th) * np.sin(d) * np.cos(la1),
                           np.cos(d) - np.sin(la1) * np.sin(la2))
    return np.degrees(la2), np.degrees(lo2)


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_R * math.asin(math.sqrt(a))


def ray_points(lat, lon, azimuths, max_dist_m, step_m, min_dist_m=100.0):
    dist = np.arange(min_dist_m, max_dist_m, step_m)
    lats, lons = destination(lat, lon, azimuths[:, None], dist[None, :])
    return dist, lats, lons


def compute_horizon(dem, lat, lon, eye_elev, azimuths, max_dist_m, k, chunk=64):
    """Skyline elevation angle (deg) per azimuth and the terrain point causing it.

    Line-of-sight refraction is modelled with an enlarged earth radius
    R/(1-k), the usual terrestrial refraction coefficient k ~ 0.13.
    """
    r_eff = EARTH_R / (1.0 - k)
    step = 156543.03392 * math.cos(math.radians(lat)) / 2 ** dem.zoom
    n = len(azimuths)
    hor = np.empty(n)
    b_lat, b_lon = np.empty(n), np.empty(n)
    b_elev, b_dist = np.empty(n), np.empty(n)
    for i in range(0, n, chunk):
        az = azimuths[i:i + chunk]
        dist, lats, lons = ray_points(lat, lon, az, max_dist_m, step)
        h = dem.sample(lats, lons).astype(np.float64)
        delta = dist[None, :] / r_eff
        ang = np.degrees(np.arctan2((r_eff + h) * np.cos(delta) - (r_eff + eye_elev),
                                    (r_eff + h) * np.sin(delta)))
        ang = np.where(np.isnan(ang), -90.0, ang)
        j = np.argmax(ang, axis=1)
        rows = np.arange(len(az))
        hor[i:i + chunk] = ang[rows, j]
        b_lat[i:i + chunk], b_lon[i:i + chunk] = lats[rows, j], lons[rows, j]
        b_elev[i:i + chunk], b_dist[i:i + chunk] = h[rows, j], dist[j]
    return hor, b_lat, b_lon, b_elev, b_dist


def name_peak(lat, lon):
    """Label a ridge point by the nearest named summit, or None."""
    best = None
    for name, plat, plon in PEAKS:
        d = haversine_m(lat, lon, plat, plon)
        if d <= PEAK_MATCH_M and (best is None or d < best[0]):
            best = (d, name)
    if best is None:
        return None
    d, name = best
    return name if d <= PEAK_NEAR_M else f"ridge {d / 1000:.1f} km from {name}"


# --------------------------------------------------------------------------
# Crossings
# --------------------------------------------------------------------------

def crossings(ts, diff):
    """Linear-interpolated times where diff changes sign: [(t, +1 up / -1 down)]."""
    s = np.sign(diff)
    s[s == 0] = 1
    idx = np.nonzero(s[1:] != s[:-1])[0]
    out = []
    for i in idx:
        frac = diff[i] / (diff[i] - diff[i + 1])
        out.append((ts[i] + frac * (ts[i + 1] - ts[i]), int(s[i + 1])))
    return out


def fmt_time(t, tz):
    dt = datetime.fromtimestamp(t, tz)
    return dt.strftime("%H:%M:%S")


def fmt_peak_elev(m):
    return f"{m * FT_PER_M:,.0f} ft ({m:,.0f} m)"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="When does the sun go behind the mountains before sunset?")
    p.add_argument("--lat", type=float, default=40.4137056, help="observer latitude")
    p.add_argument("--lon", type=float, default=-111.5065762, help="observer longitude")
    p.add_argument("--date", help="YYYY-MM-DD (default: today in --tz)")
    p.add_argument("--tz", default="America/Denver", help="IANA time zone for input/output")
    p.add_argument("--elevation", type=float, help="observer ground elevation, m (default: from DEM)")
    p.add_argument("--eye-height", type=float, default=1.7, help="eye height above ground, m")
    p.add_argument("--limb", choices=["upper", "center", "lower"], default="upper",
                   help="which part of the sun must clear the skyline (default upper = "
                        "the last sliver of direct sun)")
    p.add_argument("--max-dist-km", type=float, default=40.0, help="how far to look for terrain")
    p.add_argument("--az-step", type=float, default=0.05, help="skyline azimuth resolution, deg")
    p.add_argument("--zoom", type=int, default=13, help="terrain tile zoom (13 = ~14 m pixels)")
    p.add_argument("--temp", type=float, default=10.0, help="air temperature for refraction, C")
    p.add_argument("--pressure", type=float, help="air pressure for refraction, kPa (default: from elevation)")
    p.add_argument("--k", type=float, default=0.13, help="terrestrial refraction coefficient")
    p.add_argument("--profile", action="store_true", help="print the skyline every 1 degree of azimuth")
    p.add_argument("--plot", metavar="PNG", help="save a skyline + sun-path plot (needs matplotlib)")
    p.add_argument("--cache-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tile_cache"))
    return p.parse_args()


def main():
    a = parse_args()
    tz = ZoneInfo(a.tz)
    if a.date:
        y, m, d = map(int, a.date.split("-"))
    else:
        now = datetime.now(tz)
        y, m, d = now.year, now.month, now.day
    noon = datetime(y, m, d, 12, 0, tzinfo=tz)
    day_label = noon.strftime("%A %Y-%m-%d")

    # ---- DEM and observer elevation -----------------------------------
    dem = Dem(a.zoom, a.cache_dir)

    # ---- sun path from local noon through the evening, 1 s resolution -
    ts = noon.timestamp() + np.arange(0, 10 * 3600 + 1, dtype=float)
    az, true_alt = solar_position(ts, a.lat, a.lon)

    # Observer elevation needs a (small) tile before pressure is known.
    dem.load(dem.tiles_needed(np.array([a.lat]), np.array([a.lon])))
    ground = a.elevation if a.elevation is not None else float(dem.sample(np.array([a.lat]), np.array([a.lon]))[0])
    pressure = a.pressure if a.pressure is not None else standard_pressure_kpa(ground)
    app_alt = true_alt + refraction_deg(true_alt, pressure, a.temp)
    limb_off = {"upper": SUN_SEMIDIAMETER, "center": 0.0, "lower": -SUN_SEMIDIAMETER}[a.limb]
    alt_limb = app_alt + limb_off

    # Part of the evening where terrain could matter: sun below 20 degrees.
    window = (alt_limb < 20.0) & (alt_limb > -1.0) & (np.arange(len(ts)) > 3600)
    if not window.any():
        sys.exit("The sun never sets below 20 degrees on this date/location.")
    az_lo, az_hi = az[window].min() - 1.0, az[window].max() + 1.0
    azimuths = np.arange(az_lo, az_hi + a.az_step, a.az_step)

    # ---- skyline ------------------------------------------------------
    eye = ground + a.eye_height
    max_m = a.max_dist_km * 1000.0
    step = 156543.03392 * math.cos(math.radians(a.lat)) / 2 ** a.zoom
    needed = set()
    for i in range(0, len(azimuths), 64):
        _, lats, lons = ray_points(a.lat, a.lon, azimuths[i:i + 64], max_m, step)
        needed |= dem.tiles_needed(lats.ravel(), lons.ravel())
    print(f"Loading {len(needed)} terrain tiles (cached in {a.cache_dir}) ...", file=sys.stderr)
    dem.load(needed)
    hor, b_lat, b_lon, b_elev, b_dist = compute_horizon(dem, a.lat, a.lon, eye, azimuths, max_m, a.k)

    # ---- crossings ----------------------------------------------------
    def diff_for(offset):
        sky = np.interp(az, azimuths, hor)
        return (app_alt + offset) - sky

    def blocker_at(azim):
        i = int(np.clip(np.searchsorted(azimuths, azim), 0, len(azimuths) - 1))
        return dict(hor=hor[i], lat=b_lat[i], lon=b_lon[i], elev=b_elev[i], dist=b_dist[i],
                    bearing=azimuths[i], name=name_peak(b_lat[i], b_lon[i]))

    def at(t):
        return (float(np.interp(t, ts, az)), float(np.interp(t, ts, alt_limb)))

    after_noon = np.arange(len(ts)) > 3600
    cr_flat = [t for t, dr in crossings(ts[after_noon], alt_limb[after_noon]) if dr < 0]
    flat_sunset = cr_flat[0] if cr_flat else None

    events = [(t, dr) for t, dr in crossings(ts[after_noon], diff_for(limb_off)[after_noon])
              if np.interp(t, ts, alt_limb) > -1.0]

    # ---- report -------------------------------------------------------
    print()
    print(f"Location : {a.lat:.7f}, {a.lon:.7f}   ground {fmt_peak_elev(ground)}"
          f"{'' if a.elevation is not None else ' (from DEM)'}")
    print(f"Date     : {day_label}  ({a.tz}, UTC{noon.strftime('%z')[:3]}:{noon.strftime('%z')[3:]} {noon.tzname()})")
    print(f"Sun limb : {a.limb}   refraction: {pressure:.1f} kPa, {a.temp:.0f} C   line-of-sight k={a.k}")
    if flat_sunset:
        faz, falt = at(flat_sunset)
        print(f"Flat-horizon sunset (no mountains): {fmt_time(flat_sunset, tz)}  az {faz:.1f} deg")
    print()

    if not events:
        print("The sun stays above the terrain skyline until it sets on the flat horizon.")
    print("Events (sun vs. terrain skyline):")
    for t, dr in events:
        saz, salt = at(t)
        b = blocker_at(saz)
        verb = "REAPPEARS from behind" if dr > 0 else "goes BEHIND          "
        print(f"  {fmt_time(t, tz)}  sun {verb} az {saz:6.2f} deg  alt {salt:5.2f} deg  "
              f"skyline {b['hor']:5.2f} deg")
        print(f"            ridge point {b['lat']:.4f}, {b['lon']:.4f}  {fmt_peak_elev(b['elev'])}, "
              f"{b['dist'] / 1000:.1f} km away" + (f"  [{b['name']}]" if b["name"] else ""))

    downs = [t for t, dr in events if dr < 0]
    if downs:
        t_final = downs[-1]
        saz, _ = at(t_final)
        b = blocker_at(saz)
        print()
        print(f"Direct sunlight is lost for the day at {fmt_time(t_final, tz)}"
              + (f", {(flat_sunset - t_final) / 60:.1f} min before flat-horizon sunset" if flat_sunset else "")
              + f" (behind {b['name'] or 'terrain'}).")
        print("Disc detail for that final disappearance:")
        for label, off in (("lower limb touches skyline", -SUN_SEMIDIAMETER),
                           ("center of sun on skyline ", 0.0),
                           ("upper limb gone          ", SUN_SEMIDIAMETER)):
            c = [t for t, dr in crossings(ts[after_noon], diff_for(off)[after_noon])
                 if dr < 0 and np.interp(t, ts, alt_limb) > -1.0]
            if c:
                print(f"  {fmt_time(c[-1], tz)}  {label}")

    if a.profile:
        print()
        print("Skyline (az deg, elevation angle deg, distance km, ridge elevation):")
        for i in range(0, len(azimuths), max(1, int(round(1.0 / a.az_step)))):
            nm = name_peak(b_lat[i], b_lon[i]) or ""
            print(f"  {azimuths[i]:7.2f}  {hor[i]:6.2f}  {b_dist[i] / 1000:5.1f}  {b_elev[i] * FT_PER_M:7.0f} ft  {nm}")

    if a.plot:
        make_plot(a.plot, azimuths, hor, az[window], alt_limb[window], events, ts, az, alt_limb, tz, a)


def make_plot(path, azimuths, hor, az_w, alt_w, events, ts, az, alt_limb, tz, a):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping --plot", file=sys.stderr)
        return
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(azimuths, hor, -2, color="#4a5a4a", label="terrain skyline")
    ax.plot(az_w, alt_w, color="#e69f00", lw=2, label=f"sun ({a.limb} limb)")
    for t, dr in events:
        saz, salt = np.interp(t, ts, az), np.interp(t, ts, alt_limb)
        ax.plot(saz, salt, "o", color="#d55e00" if dr < 0 else "#009e73")
        ax.annotate(fmt_time(t, tz), (saz, salt), textcoords="offset points",
                    xytext=(6, 8), fontsize=9)
    ax.set_xlim(azimuths[0], azimuths[-1])
    ax.set_ylim(-1, max(hor.max() + 3, 12))
    ax.set_xlabel("azimuth (deg from north)")
    ax.set_ylabel("elevation angle (deg)")
    ax.set_title(f"Skyline and sun path from {a.lat:.4f}, {a.lon:.4f}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"Plot saved to {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
