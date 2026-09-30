# blackhole

A general-relativistic black-hole accretion-disk animation for **monochrome
terminals**, cheap enough to run as a screensaver. Rust, zero dependencies.

Physics after Florentin Jaffredo's ["Black hole rendering with
SageMath"](https://sagemanifolds.obspm.fr/) notebook: backward ray tracing of
null geodesics in Schwarzschild spacetime, an optically-thin tilted accretion
disk, and exact gravitational + Doppler redshift with relativistic beaming —
the same physics, but restructured so it runs in milliseconds instead of
minutes:

- **Spherical symmetry** (the notebook's key trick): only a 1D fan of
  geodesics indexed by view angle is integrated; every pixel reuses a fan ray
  rotated to its position angle. The Binet equation `u'' = 3u² − u` replaces
  the 4-D coordinate ODE.
- **Fixed geometry, moving matter**: camera and disk orientation never change,
  so every ray/disk crossing — position, incidence, redshift factor g — is
  baked once at startup into a per-cell *lensing map*. Because disk plane and
  geodesic plane both pass through the origin, crossings sit at fixed orbital
  angles φ_c + kπ (an indexed lookup along the polyline, not an intersection
  test), and
  g = √(1−3M/r) / (√(1−2M/r_cam)·(1−Ω·L_z)) is closed-form.
- **Per frame**, only the turbulent emission pattern is advected with
  Keplerian differential rotation Ω = √(M/r³) through the baked map:
  a vectorized, branch-free pass of a handful of flops per disk crossing,
  with phases re-anchored in f64 so the animation stays exact for days.
  Tone mapping is folded into precomputed thresholds, and only the cells
  whose glyph changed are sent — one terminal write per frame.

What you see (all emergent from the ray tracing, not painted): the shadow,
the thin photon ring near b = 3√3 M, the far side of the disk lensed into
arcs above and below the hole, the secondary image, one side of the disk
Doppler-boosted (g³ by default, bolometric g⁴ via `--beaming`) far brighter
than the other, gravitationally dimmed emission near the inner edge, and
background stars smeared into Einstein arcs around the shadow.

## Usage

```
cargo install --path .
blackhole                 # any key exits
blackhole --style braille # 2×4 dots per cell, Bayer-dithered
blackhole --bench         # print precompute + per-frame timings
blackhole --help          # all options
```

Options: `--fps`, `--inclination`, `--rin/--rout`, `--stars on|off`,
`--rotation cw|ccw`, `--beaming`, `--zoom`, `--speed`, `--exposure`,
`--aspect`, `--seed`. Numeric values must be finite; bounds: fps ≤ 240,
rout ≤ 500.

Screensaver-style use: run it fullscreen, e.g. `konsole --fullscreen -e blackhole`.

## Performance

Target: < 5 % of one core at 20 fps in a fullscreen terminal; map rebuild
(startup and on resize) well under a second. `--bench` measures both — frames
are timed an hour into the animation, where the sheared texture is at its
most expensive — and reports the output rate the terminal has to absorb: at
20 fps only ~1–2 % of cells change per frame, so updates carry just those
(plus a full repaint every 10 s). Cost scales with the samples evaluated:
braille takes 8 per cell, ASCII 9 (anti-aliasing supersamples; 4 on very
large terminals).
