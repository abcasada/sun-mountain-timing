# sun-mountain-timing

Estimates when the sun disappears behind the mountains before sunset, for any
location and date. The terrain skyline is built from elevation data (AWS
Terrarium tiles), so the "mountain" is whatever ridge actually blocks the sun.
Blocking ridge points are labelled with the nearest named summit (edit `PEAKS`
in the script to add more).

```
pip install numpy pillow          # matplotlib optional, for --plot
python3 sun_mountain_timing.py                        # today, default location
python3 sun_mountain_timing.py --date 2026-09-20 --plot skyline.png --profile
python3 sun_mountain_timing.py --lat 40.41 --lon -111.50 --tz America/Denver
```

Options: `--limb upper|center|lower` (default `upper`: the last sliver of direct
sun), `--elevation` (m, default from the DEM), `--max-dist-km`, `--temp`,
`--pressure`, `--k` (line-of-sight refraction). Needs internet on first run;
tiles are cached in `.tile_cache/`.

Accuracy: solar position agrees with `ephem` to ~0.01 deg. The DEM is bare
earth, so trees, buildings and small nearby hills are not modelled, and the
exact ridge crest can differ from the DEM by a few metres (about +/-15 s here).
