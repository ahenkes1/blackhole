//! The baked per-cell lensing map: all general-relativistic work happens
//! here, once per terminal size. Frames then cost O(hits) flops.
//!
//! # Geometry (all planes pass through the origin — everything is exact)
//!
//! World frame: black hole at origin, camera at C = r_cam·ẑ looking down −ẑ.
//! Screen right = +x̂, screen up = +ŷ. Disk axis (unit normal), tilted from
//! the line of sight by the viewing inclination i (90° = edge-on):
//!
//! ```text
//!     n̂ = (0, sin i, cos i)
//! ```
//!
//! A sample cell at offset (sx, sy) from screen center (in "pixel-width"
//! units, sy already scaled by the cell aspect and positive UP) maps to
//!
//! ```text
//!     ρ = √(sx² + sy²),  view angle δ = k·ρ,  position angle β = atan2(sy, sx)
//! ```
//!
//! with k = fov / (h·px_aspect) (vertical field of view across the frame).
//! The ray's orbital plane is spanned by
//!
//! ```text
//!     ê₁ = ẑ,   ê₂ = x̂·cos β + ŷ·sin β,      plane normal k̂ = (−sin β, cos β, 0)
//! ```
//!
//! and a polyline point (r, φ) sits at P = r(ê₁ cos φ + ê₂ sin φ).
//!
//! ## Disk crossings are at fixed φ
//! n̂·P = 0 ⟺ A cos φ + B sin φ = 0 with A = n̂·ê₁ = cos i, B = n̂·ê₂ = sin i sin β.
//! So crossings happen exactly at φ_c + kπ, k = 0, 1, …, where
//!
//! ```text
//!     φ_c = atan2(−A, B), then += π if ≤ 0     (φ_c ∈ (0, π])
//! ```
//!
//! For each k while φ_c + kπ ≤ ray.phi_end(): look up r via `Ray::r_at_phi`;
//! it is a disk hit iff r ∈ [r_in, r_out]. Keep at most `MAX_HITS` (the
//! lowest-order images dominate).
//!
//! ## Per-hit static weight
//!
//! ```text
//!     weight = profile(r) · min(1/|sin θ_inc|, THICK_MAX) · g^beaming
//! ```
//!
//!  - θ_inc: angle between photon direction and disk plane. Photon in-plane
//!    velocity v = (dr/dφ)·ê_r + r·ê_φ with ê_r = ê₁cos φ + ê₂sin φ,
//!    ê_φ = −ê₁sin φ + ê₂cos φ; sin θ_inc = |v̂·n̂| (Euclidean approximation,
//!    fine for a visual thickness factor).
//!  - g: exact redshift/Doppler factor. The photon's angular momentum about
//!    the disk axis is conserved. The *traced* ray runs backward (camera →
//!    disk, dφ > 0), so its angular momentum points along +k̂; the physical
//!    photon (disk → camera) travels the same path in reverse, so
//!    L_z = −b·(k̂·n̂) = −b·sin i·cos β. For emitting matter on a circular
//!    Keplerian orbit with rotation sense `rot` (±1, counter-clockwise about
//!    n̂ — the same sense the texture advects in), 4-velocity
//!    u^t = 1/√(1−3/r), u^φd = rot·Ω·u^t:
//!
//!    ```text
//!        g = E_cam / E_emit = √(1 − 3/r) / ( √(1 − 2/r_cam) · (1 − rot·Ω(r)·L_z) )
//!    ```
//!
//!    Check: rot = +1 moves the left limb (P ∝ −x̂) along n̂ × (−x̂), which
//!    has v_z = +sin i > 0, i.e. toward the camera, and indeed L_z > 0 there
//!    (cos β = −1) gives g > 1: the approaching side is the bright one.
//!    Beaming: multiply intensity by g^beaming (4 = bolometric, default 3).
//!  - disk azimuth of the hit (for the rotating texture), measured in the
//!    disk basis ê_a = x̂, ê_b = n̂ × x̂ = (0, cos i, −sin i) — increasing
//!    azimuth is counter-clockwise about n̂ = ê_a × ê_b:
//!
//!    ```text
//!        φ₀ = atan2(P·ê_b, P·ê_a),   stored with ω = rot·Ω(r)
//!    ```
//!
//! ## Star background
//! If the ray escapes and stars are enabled, its asymptotic 3D direction is
//! d̂ = cos θ_v·ê₁ + sin θ_v·ê₂ (θ_v = `Fate::Escaped::dir_angle`); the cell's
//! `base` intensity is `star_intensity(d̂, seed)`. Nearby pixels share lensed
//! directions near the shadow → stars smear into Einstein arcs for free.
//!
//! # Build
//! `build` integrates the fan (n = 3 rays per sample of half-diagonal,
//! r_escape = max(1.5·r_cam, 2·r_out)) and fills rows in parallel with
//! `std::thread::scope` (chunk rows by available_parallelism).
//! [`LensingMap::downsample`] then folds an ss×ss anti-aliasing box filter
//! into the hit lists, so supersampling costs nothing per frame beyond the
//! extra hits themselves.

use crate::disk::{omega, profile};
use crate::geodesic::{Fan, Fate};

/// Disk crossings kept per sample cell (before [`LensingMap::downsample`]).
pub const MAX_HITS: usize = 4;
/// Cap on the optically-thin 1/sin(incidence) path-length boost. Kept modest:
/// the near-edge-on foreground is grazing (sin θ ≈ 0.1) while the lensed arch
/// above the shadow is hit almost perpendicularly — an uncapped boost buries
/// the arch, the signature feature, under the foreground fan.
pub const THICK_MAX: f64 = 4.0;

/// Scene parameters (geometric units M = 1).
#[derive(Debug, Clone)]
pub struct Scene {
    pub r_cam: f64,
    /// Viewing inclination in degrees; 90 = edge-on, 0 = face-on.
    pub incl_deg: f64,
    pub r_in: f64,
    pub r_out: f64,
    /// Vertical field of view in radians (whole frame height).
    pub fov: f64,
    /// Height/width of one sample cell in pixel-width units:
    /// 2.0 for ASCII terminal cells, 1.0 for braille subdots.
    pub px_aspect: f64,
    pub stars: bool,
    pub seed: u64,
    /// Beaming exponent applied to g (4 = bolometric).
    pub beaming: f64,
    /// Disk rotation sense, ±1.
    pub rot: f64,
}

impl Default for Scene {
    fn default() -> Self {
        Scene {
            r_cam: 60.0,
            incl_deg: 81.0,
            r_in: 8.0,
            r_out: 44.0,
            // Frames the shadow (angular radius ≈ b_crit/r_cam ≈ 0.085 rad)
            // at ~1/3 of the screen height; the outer disk spills past the
            // edges, Interstellar-style.
            fov: 0.52,
            px_aspect: 2.0,
            stars: true,
            seed: 0x5EED_CAFE,
            // g³ by default: g⁴ is the bolometric truth but blows out the
            // approaching side on a 10-level character ramp.
            beaming: 3.0,
            rot: 1.0,
        }
    }
}

/// The baked disk crossings, structure-of-arrays: the per-frame kernel
/// streams each field, which lets it vectorize (see [`crate::disk::Animator`]).
#[derive(Debug, Clone, Default)]
pub struct Hits {
    pub weight: Vec<f32>,
    /// Disk azimuth of the crossing at t = 0.
    pub phi0: Vec<f32>,
    /// Signed pattern angular velocity rot·Ω(r_hit).
    pub omega: Vec<f32>,
}

impl Hits {
    fn with_capacity(n: usize) -> Hits {
        Hits {
            weight: Vec::with_capacity(n),
            phi0: Vec::with_capacity(n),
            omega: Vec::with_capacity(n),
        }
    }

    fn push(&mut self, weight: f32, phi0: f32, omega: f32) {
        self.weight.push(weight);
        self.phi0.push(phi0);
        self.omega.push(omega);
    }

    fn append(&mut self, other: &mut Hits) {
        self.weight.append(&mut other.weight);
        self.phi0.append(&mut other.phi0);
        self.omega.append(&mut other.omega);
    }

    pub fn len(&self) -> usize {
        self.weight.len()
    }

    pub fn is_empty(&self) -> bool {
        self.weight.is_empty()
    }
}

/// CSR-layout lensing map over `w × h` sample cells (row-major).
#[derive(Debug, Clone)]
pub struct LensingMap {
    pub w: usize,
    pub h: usize,
    /// Static background intensity (lensed stars), one per cell.
    pub base: Vec<f32>,
    /// CSR offsets into `hits`, length w·h + 1.
    pub offsets: Vec<u32>,
    pub hits: Hits,
}

impl LensingMap {
    /// Fold an `ss × ss` box filter into the map: output cell (x, y) takes
    /// the mean `base` of its ss² sub-cells and all of their hits with
    /// weight / ss². A frame is linear in the per-hit texture values, so
    /// evaluating the result equals box-averaging a frame evaluated on `self`
    /// (up to f32 summation order) — without the ss²-sized intermediate
    /// frame or a downsample pass. `w` and `h` must be multiples of `ss`;
    /// `ss = 1` returns the map unchanged.
    pub fn downsample(self, ss: usize) -> LensingMap {
        assert!(ss >= 1 && self.w.is_multiple_of(ss) && self.h.is_multiple_of(ss));
        if ss == 1 {
            return self;
        }
        let (w, h) = (self.w / ss, self.h / ss);
        let inv = 1.0 / (ss * ss) as f32;
        let mut base = Vec::with_capacity(w * h);
        let mut offsets = Vec::with_capacity(w * h + 1);
        let mut hits = Hits::with_capacity(self.hits.len());
        offsets.push(0u32);
        for y in 0..h {
            for x in 0..w {
                let mut b = 0.0;
                for sy in 0..ss {
                    for sx in 0..ss {
                        let i = (y * ss + sy) * self.w + x * ss + sx;
                        b += self.base[i];
                        for j in self.offsets[i] as usize..self.offsets[i + 1] as usize {
                            hits.push(
                                self.hits.weight[j] * inv,
                                self.hits.phi0[j],
                                self.hits.omega[j],
                            );
                        }
                    }
                }
                base.push(b * inv);
                offsets.push(hits.len() as u32);
            }
        }
        LensingMap {
            w,
            h,
            base,
            offsets,
            hits,
        }
    }
}

/// Exact combined gravitational + Doppler redshift factor for emission from
/// a circular Keplerian orbit at `r_emit`, photon angular momentum about the
/// disk axis `lz` (signed, already including rotation sense via the caller),
/// observed by a static camera at `r_cam`:
///
/// ```text
///     g = √(1 − 3/r_emit) / ( √(1 − 2/r_cam) · (1 − Ω(r_emit)·lz) )
/// ```
///
/// Physically Ω·lz < 1 whenever the hit radius and impact parameter are
/// physical (Ω·b ≤ 1/√(r−2) ≈ 0.49 for r ≥ 6.1, |lz| ≤ b), so the Doppler
/// denominator stays positive; the floor only bounds API misuse.
pub fn g_factor(r_emit: f64, lz: f64, r_cam: f64) -> f64 {
    let doppler = (1.0 - omega(r_emit) * lz).max(1e-9);
    (1.0 - 3.0 / r_emit).max(0.0).sqrt() / ((1.0 - 2.0 / r_cam).sqrt() * doppler)
}

pub(crate) fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Deterministic sparse procedural star field sampled by unit direction.
/// ≈ 1–3 % of directions land on a star; brightness in (0, 0.9]; the rest 0.
///
/// Stars are the cells of a 3D cubic lattice (6 mrad pitch) where it meets
/// the unit sphere: compact blobs of ≲ 10 mrad in every direction. (A grid in
/// polar coordinates would put its pole on the line of sight and draw
/// radial needles there, the opposite of the tangential Einstein arcs.)
pub fn star_intensity(dir: [f64; 3], seed: u64) -> f32 {
    const CELL: f64 = 0.006;
    let h = dir.iter().fold(splitmix64(seed), |h, &d| {
        splitmix64(h ^ (d / CELL).floor() as i64 as u64)
    });
    if h & 63 != 0 {
        return 0.0;
    }
    let t = (h >> 16 & 0xFF) as f32 / 255.0;
    t * t * t * 0.9
}

/// Rays launched more than ~83° off-axis (only reachable in the far corners
/// of extremely wide terminals) leave the small-angle camera model's domain;
/// render them dark instead of inventing geometry.
const DELTA_CAP: f64 = 1.45;

/// Bake the lensing map for a `w × h` sample grid.
pub fn build(scene: &Scene, w: usize, h: usize) -> LensingMap {
    let inc = scene.incl_deg.to_radians();
    let (sin_i, cos_i) = inc.sin_cos();
    let k = scene.fov / (h.max(1) as f64 * scene.px_aspect);
    let half_diag = (w as f64 / 2.0).hypot(h as f64 / 2.0 * scene.px_aspect);
    let delta_max = (k * (half_diag + 1.0)).min(DELTA_CAP);
    // 3 rays per sample-diagonal: the photon ring lives in a δ-band narrower
    // than one sample, so nearest-ray quantization must stay well below it.
    let n_rays = ((3.0 * half_diag).ceil() as usize).max(64);
    let r_escape = (1.5 * scene.r_cam).max(2.0 * scene.r_out);
    let fan = Fan::build(scene.r_cam, delta_max, n_rays, r_escape);

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(h.max(1));
    let chunk = h.div_ceil(n_threads);
    let mut parts: Vec<(Vec<f32>, Vec<u32>, Hits)> = Vec::with_capacity(n_threads);
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for t in 0..n_threads {
            let (y0, y1) = (t * chunk, ((t + 1) * chunk).min(h));
            if y0 >= y1 {
                break;
            }
            let fan = &fan;
            handles.push(s.spawn(move || build_rows(scene, fan, k, sin_i, cos_i, w, h, y0, y1)));
        }
        for hd in handles {
            parts.push(hd.join().expect("map build thread panicked"));
        }
    });

    let mut base = Vec::with_capacity(w * h);
    let mut offsets = Vec::with_capacity(w * h + 1);
    offsets.push(0u32);
    let mut hits = Hits::with_capacity(parts.iter().map(|p| p.2.len()).sum());
    for (b, counts, mut hs) in parts {
        base.extend(b);
        for c in counts {
            offsets.push(offsets.last().unwrap() + c);
        }
        hits.append(&mut hs);
    }
    LensingMap {
        w,
        h,
        base,
        offsets,
        hits,
    }
}

/// Bake rows y0..y1 (row-major): per-cell base intensity, hit count, hits.
#[allow(clippy::too_many_arguments)]
fn build_rows(
    scene: &Scene,
    fan: &Fan,
    k: f64,
    sin_i: f64,
    cos_i: f64,
    w: usize,
    h: usize,
    y0: usize,
    y1: usize,
) -> (Vec<f32>, Vec<u32>, Hits) {
    use std::f64::consts::PI;
    let n_cells = (y1 - y0) * w;
    let mut base = Vec::with_capacity(n_cells);
    let mut counts = Vec::with_capacity(n_cells);
    let mut hits = Hits::default();
    for y in y0..y1 {
        for x in 0..w {
            let sx = x as f64 - (w as f64 - 1.0) / 2.0;
            let sy = ((h as f64 - 1.0) / 2.0 - y as f64) * scene.px_aspect;
            let delta = k * sx.hypot(sy);
            if delta >= DELTA_CAP {
                base.push(0.0);
                counts.push(0);
                continue;
            }
            let beta = sy.atan2(sx);
            let (sin_b, cos_b) = beta.sin_cos();
            let ray = fan.ray_for(delta);

            // Disk-plane crossing angles: A cos φ + B sin φ = 0.
            let a = cos_i;
            let b_ = sin_i * sin_b;
            let mut phi_c = (-a).atan2(b_);
            if phi_c <= 0.0 {
                phi_c += PI;
            }

            let mut n_cell = 0u32;
            let mut kk = 0;
            // Crossings of one cell's ray come in increasing φ order, so a
            // per-cell carried polyline cursor beats a binary search per hit.
            let mut cursor = 0usize;
            loop {
                let phi_k = phi_c + kk as f64 * PI;
                if phi_k > ray.phi_end() || n_cell as usize >= MAX_HITS {
                    break;
                }
                kk += 1;
                let Some((r, dr_dphi)) = ray.r_at_phi_with_hint(phi_k, &mut cursor) else {
                    continue;
                };
                if r < scene.r_in || r > scene.r_out {
                    continue;
                }
                let (sin_p, cos_p) = phi_k.sin_cos();
                // ê_r·n̂ = 0 at a crossing, so v̂·n̂ reduces to the ê_φ term.
                let ephi_n = -a * sin_p + b_ * cos_p;
                let sin_th = (r * ephi_n / (dr_dphi.hypot(r))).abs();
                let thickness = (1.0 / sin_th.max(1e-3)).min(THICK_MAX);
                // Physical photon's L_z (reverse of the traced ray), see module docs.
                let lz = -scene.rot * ray.b * sin_i * cos_b;
                let g = g_factor(r, lz, scene.r_cam);
                let weight =
                    profile(r, scene.r_in, scene.r_out) * thickness * g.powf(scene.beaming);
                // Disk azimuth: P/r = ê_r; ê_a = x̂, ê_b = n̂ × x̂.
                let p_a = cos_b * sin_p;
                let p_b = sin_b * sin_p * cos_i - cos_p * sin_i;
                hits.push(
                    weight as f32,
                    p_b.atan2(p_a) as f32,
                    (scene.rot * omega(r)) as f32,
                );
                n_cell += 1;
            }
            counts.push(n_cell);

            let mut b_val = 0.0f32;
            if scene.stars {
                if let Fate::Escaped { dir_angle } = ray.fate {
                    let (sin_v, cos_v) = dir_angle.sin_cos();
                    let d = [sin_v * cos_b, sin_v * sin_b, cos_v];
                    b_val = star_intensity(d, scene.seed);
                }
            }
            base.push(b_val);
        }
    }
    (base, counts, hits)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_scene(incl_deg: f64, stars: bool) -> Scene {
        Scene {
            incl_deg,
            stars,
            px_aspect: 1.0,
            // Pinned: test geometry and thresholds (which radii the small
            // grids sample, expected beaming contrast) must not shift when
            // the cosmetic defaults are retuned.
            fov: 1.35,
            beaming: 4.0,
            ..Scene::default()
        }
    }

    fn weight_sum(map: &LensingMap, x: usize, y: usize) -> f64 {
        let i = y * map.w + x;
        map.hits.weight[map.offsets[i] as usize..map.offsets[i + 1] as usize]
            .iter()
            .map(|&w| w as f64)
            .sum()
    }

    /// Total disk weight left and right of the vertical center line.
    fn left_right(map: &LensingMap) -> (f64, f64) {
        let (mut left, mut right) = (0.0, 0.0);
        for y in 0..map.h {
            for x in 0..map.w {
                let s = weight_sum(map, x, y);
                if 2 * x + 1 < map.w {
                    left += s;
                } else if 2 * x + 1 > map.w {
                    right += s;
                }
            }
        }
        (left, right)
    }

    fn hit_count(map: &LensingMap, x: usize, y: usize) -> usize {
        let i = y * map.w + x;
        (map.offsets[i + 1] - map.offsets[i]) as usize
    }

    #[test]
    fn g_factor_face_on_analytic() {
        // No orbital Doppler (lz = 0): g = √(1−3/r) / √(1−2/r_cam).
        let g = g_factor(10.0, 0.0, 60.0);
        let expected = (1.0f64 - 0.3).sqrt() / (1.0f64 - 1.0 / 30.0).sqrt();
        assert!((g - expected).abs() < 1e-12);
        // Approaching matter (Ω·lz > 0) is blueshifted.
        assert!(g_factor(10.0, 3.0, 60.0) > g);
        assert!(g_factor(10.0, -3.0, 60.0) < g);
    }

    #[test]
    fn g_factor_denominator_guard() {
        // API misuse (Ω·lz ≥ 1 is unphysical for real hits) must stay finite.
        let lz_crit = 1.0 / omega(8.0); // Ω·lz exactly 1 → denominator floored
        assert!(g_factor(8.0, lz_crit, 60.0).is_finite());
        assert!(g_factor(8.0, lz_crit + 10.0, 60.0).is_finite());
    }

    #[test]
    fn face_on_map_is_rotationally_symmetric() {
        let map = build(&small_scene(0.0, false), 41, 41);
        let (c, d) = (20usize, 8usize);
        let w0 = weight_sum(&map, c + d, c);
        for (x, y) in [(c - d, c), (c, c + d), (c, c - d)] {
            let wi = weight_sum(&map, x, y);
            assert!(
                (wi - w0).abs() <= 1e-3 * w0.abs().max(1e-12),
                "asymmetric: {wi} vs {w0} at ({x},{y})"
            );
        }
        assert!(w0 > 0.0, "the ring must be visible face-on");
    }

    #[test]
    fn extreme_inclinations_are_finite() {
        // Edge-on (±90°) puts the camera in the disk plane: crossings
        // degenerate to grazing incidence but must stay finite and visible.
        for incl in [-90.0, -30.0, 90.0] {
            let map = build(&small_scene(incl, true), 61, 41);
            assert!(map.hits.weight.iter().all(|w| w.is_finite() && *w >= 0.0));
            assert!(map.base.iter().all(|b| b.is_finite()));
            assert!(!map.hits.is_empty(), "incl {incl}: disk invisible");
        }
    }

    #[test]
    fn center_ray_is_captured_and_dark() {
        let map = build(&small_scene(81.0, true), 41, 41);
        assert_eq!(hit_count(&map, 20, 20), 0);
        assert_eq!(map.base[20 * 41 + 20], 0.0);
    }

    #[test]
    fn secondary_images_exist_near_edge_on() {
        let map = build(&small_scene(81.0, false), 81, 81);
        let multi = (0..81 * 81)
            .filter(|&i| (map.offsets[i + 1] - map.offsets[i]) >= 2)
            .count();
        assert!(multi > 0, "no cell sees a secondary disk image");
    }

    #[test]
    fn doppler_asymmetry_near_edge_on() {
        // Total weight on the approaching side must clearly exceed the
        // receding side (relativistic beaming), left/right of screen center.
        let map = build(&small_scene(81.0, false), 81, 81);
        let (left, right) = left_right(&map);
        let (dim, bright) = if left < right {
            (left, right)
        } else {
            (right, left)
        };
        assert!(
            bright > 2.0 * dim,
            "beaming asymmetry too weak: {left} vs {right}"
        );
    }

    #[test]
    fn approaching_limb_is_brighter() {
        // rot = +1 is counter-clockwise about n̂ (the sense the texture
        // advects in): the left limb, P ∝ −x̂, moves along n̂ × (−x̂) =
        // (0, −cos i, +sin i), i.e. toward the camera at +ẑ. Beaming must
        // brighten that side — and swap sides with the rotation sense.
        let ccw = build(&small_scene(81.0, false), 81, 81);
        let (l, r) = left_right(&ccw);
        assert!(
            l > 2.0 * r,
            "ccw: approaching left limb must be bright: {l} vs {r}"
        );
        let cw = build(
            &Scene {
                rot: -1.0,
                ..small_scene(81.0, false)
            },
            81,
            81,
        );
        let (l, r) = left_right(&cw);
        assert!(
            r > 2.0 * l,
            "cw: approaching right limb must be bright: {l} vs {r}"
        );
    }

    #[test]
    fn downsample_folds_box_filter() {
        let (w, h, ss) = (27usize, 18usize, 3usize);
        let full = build(&small_scene(81.0, true), w, h);
        let ds = full.clone().downsample(ss);
        assert_eq!((ds.w, ds.h), (w / ss, h / ss));
        assert_eq!(ds.hits.len(), full.hits.len(), "folding keeps every hit");
        assert_eq!(*ds.offsets.last().unwrap() as usize, ds.hits.len());
        for y in 0..ds.h {
            for x in 0..ds.w {
                let (mut base, mut wsum) = (0.0f64, 0.0f64);
                for sy in 0..ss {
                    for sx in 0..ss {
                        base += full.base[(y * ss + sy) * w + x * ss + sx] as f64;
                        wsum += weight_sum(&full, x * ss + sx, y * ss + sy);
                    }
                }
                let n = (ss * ss) as f64;
                assert!((ds.base[y * ds.w + x] as f64 - base / n).abs() < 1e-6);
                let got = weight_sum(&ds, x, y);
                assert!(
                    (got - wsum / n).abs() <= 1e-5 * (wsum / n).max(1e-6),
                    "({x},{y})"
                );
            }
        }
        // ss = 1 is the identity.
        let same = full.clone().downsample(1);
        assert_eq!(same.offsets, full.offsets);
        assert_eq!(same.hits.weight, full.hits.weight);
        assert_eq!(same.base, full.base);
    }

    #[test]
    fn stars_are_compact_near_line_of_sight() {
        // Around the view axis −ẑ, a star must not extend much further
        // radially than tangentially: scan radial lines at 1 mrad steps and
        // bound the longest lit run by the lattice cell's diagonal.
        let mut longest = 0;
        let mut lit = 0;
        for j in 0..2000 {
            let beta = j as f64 * 0.0031;
            let mut run = 0;
            for i in 0..250 {
                let th = 0.05 + i as f64 * 1e-3;
                let d = [th.sin() * beta.cos(), th.sin() * beta.sin(), -th.cos()];
                if star_intensity(d, 7) > 0.0 {
                    lit += 1;
                    run += 1;
                    longest = longest.max(run);
                } else {
                    run = 0;
                }
            }
        }
        assert!(lit > 0, "no stars near the line of sight");
        assert!(longest <= 11, "radial star streak of {longest} mrad");
    }

    #[test]
    fn stars_deterministic_and_sparse() {
        let mut lit = 0;
        for i in 0..10_000u64 {
            let z = (i as f64 / 10_000.0) * 1.8 - 0.9;
            let az = i as f64 * 0.618;
            let d = [
                (1.0 - z * z).sqrt() * az.cos(),
                (1.0 - z * z).sqrt() * az.sin(),
                z,
            ];
            let a = star_intensity(d, 42);
            assert_eq!(a, star_intensity(d, 42), "must be deterministic");
            assert!((0.0..=0.9).contains(&a));
            if a > 0.0 {
                lit += 1;
            }
        }
        assert!(
            (10..=500).contains(&lit),
            "star density off: {lit}/10000 samples lit"
        );
    }
}
