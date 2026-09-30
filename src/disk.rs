//! Accretion-disk physics and the animated emission texture.
//!
//! The disk is a thin annulus r ∈ [r_in, r_out] of matter on circular
//! Keplerian orbits, Ω(r) = r^(−3/2) (M = 1). Emission is optically thin;
//! the *static* geometry factors are baked per-hit by [`crate::map`], so the
//! per-frame work here is only:
//!
//! ```text
//!     I(cell) = base + Σ_hits weight · tex(φ₀, ω, t)
//! ```
//!
//! `tex` is the brightness of the co-rotating turbulent pattern seen at the
//! hit's fixed disk azimuth φ₀, with ω = ±Ω(r_hit) baked into the hit.
//! Differential rotation shears the pattern into spirals. Left alone, the
//! shear winds it ever tighter (the phase gradient across radii grows ∝ t)
//! until, after minutes, neighbouring cells decorrelate into flat grain. So
//! the turbulence has a finite lifetime, like real disk turbulence: patterns
//! are born every L/2 ([`LIFETIME`] L), fade in and out with weight
//! sin²(π·age/L), and each is advected only by its own age:
//!
//! ```text
//!     tex(φ₀, ω, t) = 1 + Σ_gen w_gen Σ_k a_k · sin(m_k·(φ₀ − ω·age_gen) + ρ_gen,k + c_k·age_gen)
//! ```
//!
//! Exactly two generations are alive at any t and their weights sum to 1,
//! so Σ a_k < 1 keeps tex > 0 and its mean at 1. ρ_gen,k are random phases
//! per generation (a new pattern each time), c_k slow drifts that make it
//! flicker. [`Animator`] evaluates exactly this sum, restructured for speed.
//!
//! All images of the disk (primary, secondary, …) are evaluated at the same
//! t: the extra light-travel time of the higher-order images (≈ π·r) is
//! ignored — invisible for a random texture.

use crate::map::{LensingMap, splitmix64};
use std::f64::consts::{PI, TAU};

/// Keplerian angular velocity of a circular orbit at radius r (M = 1).
pub fn omega(r: f64) -> f64 {
    r.powf(-1.5)
}

/// Azimuthal harmonics of the turbulence texture: (m, amplitude, drift rad/s).
pub const HARMONICS: [(f32, f32, f32); 2] = [(3.0, 0.55, 0.11), (7.0, 0.30, -0.07)];
const NH: usize = HARMONICS.len();

/// Lifetime L of one turbulence pattern, in sim units (2 min of wall time at
/// default speed). Bounds the shear winding: a pattern is at most L old, so
/// adjacent cells stay correlated forever instead of for a few minutes.
pub const LIFETIME: f64 = 960.0;

/// One of the two turbulence patterns alive at a given time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Generation {
    /// Fade weight sin²(π·age/L); the two alive weights sum to 1.
    pub weight: f64,
    /// Time since the pattern's birth, in [0, L).
    pub age: f64,
    /// Per-harmonic phase ρ_k + c_k·age, wrapped to [0, 2π).
    pub phase: [f64; NH],
}

/// The two pattern generations alive at sim time `t` (any sign): generation
/// g is born at g·L/2 and dies at g·L/2 + L. A pure function of `t`.
pub fn generations(t: f64) -> [Generation; 2] {
    let half = LIFETIME / 2.0;
    let g = (t / half).floor();
    let young_age = t - g * half; // [0, L/2)
    let r#gen = |id: f64, age: f64| {
        let seed = splitmix64(id as i64 as u64);
        Generation {
            weight: (PI * age / LIFETIME).sin().powi(2),
            age,
            phase: std::array::from_fn(|k| {
                let rho = (splitmix64(seed ^ k as u64) >> 11) as f64 * (TAU / (1u64 << 53) as f64);
                (rho + HARMONICS[k].2 as f64 * age).rem_euclid(TAU)
            }),
        }
    };
    [r#gen(g, young_age), r#gen(g - 1.0, young_age + half)]
}

/// Turbulent brightness at disk azimuth `phi0` (angular velocity `omega`),
/// sim time `t`, per the module-level formula, in f64. Strictly positive,
/// 1 on average. Reference definition for [`Animator`].
pub fn tex(phi0: f64, omega: f64, t: f64) -> f64 {
    let mut v = 1.0;
    for g in generations(t) {
        let psi = phi0 - omega * g.age;
        for (k, &(m, a, _)) in HARMONICS.iter().enumerate() {
            v += g.weight * a as f64 * (m as f64 * psi + g.phase[k]).sin();
        }
    }
    v
}

/// Smooth radial emissivity profile on [r_in, r_out]: zero at both edges,
/// single interior maximum near r_in + 0.2·(r_out − r_in), peak value ≈ 1.
/// (Shape after the notebook's ad-hoc bump; exact form free within tests.)
pub fn profile(r: f64, r_in: f64, r_out: f64) -> f64 {
    if r <= r_in || r >= r_out {
        return 0.0;
    }
    // g(x) = x^A (1−x)^B e^{−Cx}: log-derivative A/x − B/(1−x) − C is strictly
    // decreasing on (0,1), so g is unimodal; its argmax solves
    // C x² − (A+B+C) x + A = 0, taken analytically below to normalize peak = 1
    // (lands at x* ≈ 0.194).
    const A: f64 = 0.75;
    const B: f64 = 1.5;
    const C: f64 = 2.0;
    let g = |x: f64| x.powf(A) * (1.0 - x).powf(B) * (-C * x).exp();
    let s = A + B + C;
    let x_star = (s - (s * s - 4.0 * C * A).sqrt()) / (2.0 * C);
    let x = (r - r_in) / (r_out - r_in);
    g(x) / g(x_star)
}

/// Cells per kernel block: the block's per-hit values stay in L1 between
/// the texture pass and the per-cell sums.
const BLOCK: usize = 256;

/// Per-frame evaluator of the animated disk through a [`LensingMap`].
///
/// Precision: everything unbounded in t (generation index, drift phases) is
/// resolved per frame in f64 by [`generations`]; the f32 kernel only sees
/// pattern ages < [`LIFETIME`] and phases wrapped to [0, 2π), so the
/// animation stays exact for days and a frame is a pure function of t.
///
/// Speed: libm `sinf` is an opaque call whose branches mispredict once
/// differential rotation has scrambled the phases. The kernel instead
/// streams the structure-of-arrays hits through a branch-free polynomial
/// sine (`fast_sin`), which vectorizes, then sums each cell's hits.
#[derive(Debug, Clone)]
pub struct Animator {
    n_hits: usize,
    /// Per-hit texture values of one block of cells.
    vals: Vec<f32>,
}

impl Animator {
    pub fn new(map: &LensingMap) -> Animator {
        let n = map.w * map.h;
        let max_block_hits = (0..n)
            .step_by(BLOCK)
            .map(|c0| (map.offsets[(c0 + BLOCK).min(n)] - map.offsets[c0]) as usize)
            .max()
            .unwrap_or(0);
        Animator {
            n_hits: map.hits.len(),
            vals: vec![0.0; max_block_hits],
        }
    }

    /// Evaluate one animation frame at sim time `t`: raw (not tone-mapped)
    /// intensity per cell. `map` must be the one this was created for, and
    /// `out` must have length `map.w * map.h`.
    pub fn evaluate(&mut self, map: &LensingMap, t: f64, out: &mut [f32]) {
        let n = map.w * map.h;
        assert_eq!(out.len(), n);
        assert_eq!(
            self.n_hits,
            map.hits.len(),
            "Animator used with a different map"
        );
        let gens = generations(t);
        let age = gens.map(|g| g.age as f32);
        let amp: [[f32; NH]; 2] =
            gens.map(|g| std::array::from_fn(|k| (g.weight * HARMONICS[k].1 as f64) as f32));
        let phase = gens.map(|g| g.phase.map(|p| p as f32));
        let hits = &map.hits;

        for (block, out) in out.chunks_mut(BLOCK).enumerate() {
            let c0 = block * BLOCK;
            let (h0, h1) = (
                map.offsets[c0] as usize,
                map.offsets[c0 + out.len()] as usize,
            );
            let vals = &mut self.vals[..h1 - h0];
            let (w, phi0, om) = (
                &hits.weight[h0..h1],
                &hits.phi0[h0..h1],
                &hits.omega[h0..h1],
            );
            for (((v, &w), &p), &o) in vals.iter_mut().zip(w).zip(phi0).zip(om) {
                let mut tx = 1.0;
                for g in 0..2 {
                    let psi = p - o * age[g];
                    for k in 0..NH {
                        tx += amp[g][k] * fast_sin(HARMONICS[k].0 * psi + phase[g][k]);
                    }
                }
                *v = w * tx;
            }
            // Cells' hits are consecutive: walk them with one running index.
            let mut j = 0;
            let ends = &map.offsets[c0 + 1..=c0 + out.len()];
            for ((o, &b), &end) in out.iter_mut().zip(&map.base[c0..]).zip(ends) {
                let mut acc = b;
                while j < end as usize - h0 {
                    acc += vals[j];
                    j += 1;
                }
                *o = acc;
            }
        }
    }
}

/// sin(x) for the texture kernel, branch-free so the per-hit loop vectorizes.
/// Reduces x = q·π + r with |r| ≤ π/2 (π split Cody–Waite style so r stays
/// exact for the kernel's |q| ≲ 160), then an odd degree-9 near-minimax
/// polynomial (fitted on [−π/2, π/2]); sin x = (−1)^q · sin r. Absolute
/// error ≲ 3e-7 for |x| ≲ 1e3. Kernel arguments: |m·ψ| ≤ 7·(π + Ω(6.1)·L)
/// ≈ 470, plus a phase < 2π.
#[inline(always)]
fn fast_sin(x: f32) -> f32 {
    const PI_HI: f32 = 3.140625; // 8 significant bits: q·PI_HI is exact
    const PI_LO: f32 = 9.676_536e-4; // π − PI_HI
    let q = (x * std::f32::consts::FRAC_1_PI + 0.5f32.copysign(x)) as i32;
    let qf = q as f32;
    let r = (x - qf * PI_HI) - qf * PI_LO;
    let r2 = r * r;
    let p = r
        * (1.0
            + r2 * (-0.166_666_48
                + r2 * (8.332_9e-3 + r2 * (-1.980_089_7e-4 + r2 * 2.590_488_5e-6))));
    f32::from_bits(p.to_bits() ^ ((q as u32) << 31))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::{self, Scene};

    fn test_map(w: usize, h: usize) -> LensingMap {
        let scene = Scene {
            px_aspect: 1.0,
            ..Scene::default()
        };
        map::build(&scene, w, h)
    }

    /// The module-level formula, summed in f64 with exact phases.
    fn reference_frame(m: &LensingMap, t: f64) -> Vec<f64> {
        (0..m.w * m.h)
            .map(|i| {
                let mut acc = m.base[i] as f64;
                for j in m.offsets[i] as usize..m.offsets[i + 1] as usize {
                    let tx = tex(m.hits.phi0[j] as f64, m.hits.omega[j] as f64, t);
                    acc += m.hits.weight[j] as f64 * tx;
                }
                acc
            })
            .collect()
    }

    fn max_rel_err(got: &[f32], want: &[f64]) -> f64 {
        got.iter()
            .zip(want)
            .map(|(&g, &w)| (g as f64 - w).abs() / w.abs().max(1e-3))
            .fold(0.0, f64::max)
    }

    #[test]
    fn fast_sin_accuracy() {
        let mut worst = 0.0f64;
        for i in -1_200_000..=1_200_000 {
            let x = i as f32 * 5e-4; // |x| ≤ 600, beyond the kernel's range
            worst = worst.max((fast_sin(x) as f64 - (x as f64).sin()).abs());
        }
        assert!(worst < 3e-7, "fast_sin max abs error {worst:e}");
        assert_eq!(fast_sin(0.0), 0.0);
    }

    #[test]
    fn animator_matches_reference_for_days() {
        let m = test_map(64, 48);
        let mut anim = Animator::new(&m);
        let mut out = vec![0.0f32; m.w * m.h];
        // start-up, mid-epoch, 1 h, 1 day, 1 week of wall time at speed 1
        for t in [0.0, 1234.5, 28_800.0, 691_200.0, 4_838_400.0] {
            anim.evaluate(&m, t, &mut out);
            let err = max_rel_err(&out, &reference_frame(&m, t));
            assert!(err < 1e-4, "t = {t}: max rel error {err:e}");
        }
    }

    #[test]
    fn frame_steps_resolve_after_a_week() {
        // The f32-time kernel this replaced could no longer tell a 0.4-unit
        // frame step apart after ~6 days (ulp(t) = 0.5).
        let m = test_map(32, 24);
        let mut anim = Animator::new(&m);
        let (mut a, mut b) = (vec![0.0f32; 32 * 24], vec![0.0f32; 32 * 24]);
        let t = 7.0 * 86_400.0 * 8.0;
        anim.evaluate(&m, t, &mut a);
        anim.evaluate(&m, t + 0.4, &mut b);
        assert_ne!(a, b);
        let err = max_rel_err(&b, &reference_frame(&m, t + 0.4));
        assert!(err < 1e-4, "max rel error {err:e}");
    }

    #[test]
    fn animator_is_history_independent() {
        let m = test_map(40, 30);
        let mut fresh = vec![0.0f32; 40 * 30];
        let mut warm = vec![0.0f32; 40 * 30];
        let t = 10_000.3;
        Animator::new(&m).evaluate(&m, t, &mut fresh);
        let mut anim = Animator::new(&m);
        for t_prev in [0.0, 5e5, -3.0, 9_999.0] {
            anim.evaluate(&m, t_prev, &mut warm);
        }
        anim.evaluate(&m, t, &mut warm);
        assert_eq!(fresh, warm, "a frame must be a pure function of t");
    }

    #[test]
    fn downsampled_map_equals_box_filtered_frame() {
        let (w, h, ss) = (40usize, 30usize, 3usize);
        let full = test_map(w * ss, h * ss);
        let folded = full.clone().downsample(ss);
        let t = 28_800.0;
        let mut hi = vec![0.0f32; w * h * ss * ss];
        let mut lo = vec![0.0f32; w * h];
        Animator::new(&full).evaluate(&full, t, &mut hi);
        Animator::new(&folded).evaluate(&folded, t, &mut lo);
        for y in 0..h {
            for x in 0..w {
                let mut acc = 0.0f64;
                for sy in 0..ss {
                    for sx in 0..ss {
                        acc += hi[(y * ss + sy) * w * ss + x * ss + sx] as f64;
                    }
                }
                let want = acc / (ss * ss) as f64;
                let got = lo[y * w + x] as f64;
                assert!(
                    (got - want).abs() <= 1e-5 * want.max(1e-3),
                    "({x},{y}): {got} vs {want}"
                );
            }
        }
    }

    #[test]
    fn omega_isco() {
        assert!((omega(6.0) - 6.0f64.powf(-1.5)).abs() < 1e-15);
        assert!((omega(6.0) - 0.06804138174397717).abs() < 1e-12);
    }

    #[test]
    fn tex_positive_and_centered() {
        let mut min = f64::MAX;
        let mut sum = 0.0f64;
        let n = 100_000;
        for i in 0..n {
            let x = i as f64;
            let v = tex(x * 0.01, omega(8.0 + (x * 0.37) % 36.0), x * 0.7);
            min = min.min(v);
            sum += v;
        }
        assert!(min > 0.0, "tex must stay positive, min = {min}");
        assert!(
            (sum / n as f64 - 1.0).abs() < 0.02,
            "tex should average ≈ 1"
        );
    }

    #[test]
    fn generations_partition_time() {
        for i in -2000..2000 {
            let t = i as f64 * 1.37;
            let [young, old] = generations(t);
            assert!((young.weight + old.weight - 1.0).abs() < 1e-12, "t = {t}");
            assert!((0.0..LIFETIME / 2.0).contains(&young.age), "t = {t}");
            assert!((old.age - young.age - LIFETIME / 2.0).abs() < 1e-9);
            assert_ne!(young.phase, old.phase, "each generation is a new pattern");
        }
    }

    #[test]
    fn tex_is_continuous_across_generation_changes() {
        // Births and deaths happen at weight 0: no visible jump.
        let eps = 1e-6;
        for g in -3..6 {
            let t = g as f64 * LIFETIME / 2.0;
            for (phi0, om) in [(0.3, omega(8.0)), (-2.0, -omega(20.0)), (1.1, omega(40.0))] {
                let jump = (tex(phi0, om, t + eps) - tex(phi0, om, t - eps)).abs();
                assert!(jump < 1e-5, "jump {jump} at t = {t}");
            }
        }
    }

    /// Mean horizontal-neighbour correlation of the texture modulation
    /// I/I_static − 1 over the disk cells of a default ASCII frame.
    fn neighbour_correlation(m: &LensingMap, t: f64) -> f64 {
        let stat: Vec<f64> = (0..m.w * m.h)
            .map(|i| {
                (m.offsets[i]..m.offsets[i + 1])
                    .map(|j| m.hits.weight[j as usize] as f64)
                    .sum()
            })
            .collect();
        let mut out = vec![0.0f32; m.w * m.h];
        Animator::new(m).evaluate(m, t, &mut out);
        let md = |i: usize| (stat[i] > 1e-3).then(|| out[i] as f64 / stat[i] - 1.0);
        let (mut s, mut s2, mut n, mut c, mut nc) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for y in 0..m.h {
            for x in 0..m.w {
                let Some(v) = md(y * m.w + x) else { continue };
                (s, s2, n) = (s + v, s2 + v * v, n + 1.0);
                if let Some(u) = (x + 1 < m.w).then(|| md(y * m.w + x + 1)).flatten() {
                    (c, nc) = (c + v * u, nc + 1.0);
                }
            }
        }
        let var = s2 / n - (s / n).powi(2);
        (c / nc - (s / n).powi(2)) / var
    }

    #[test]
    fn texture_stays_coherent_for_hours() {
        // Unbounded shear winding used to decorrelate neighbouring cells
        // (correlation 0.24 after an hour); finite lifetimes cap it.
        let scene = Scene {
            stars: false,
            ..Scene::default()
        };
        let m = map::build(&scene, 120 * 3, 40 * 3).downsample(3);
        for t in [10.0, 300.0, 28_800.0, 28_800.0 + 700.0, 691_200.0] {
            let c = neighbour_correlation(&m, t);
            assert!(c > 0.7, "t = {t}: neighbour correlation {c:.2}");
        }
    }

    #[test]
    fn profile_support_and_shape() {
        let (r_in, r_out) = (8.0, 44.0);
        assert_eq!(profile(r_in, r_in, r_out), 0.0);
        assert_eq!(profile(r_out, r_in, r_out), 0.0);
        assert_eq!(profile(2.0, r_in, r_out), 0.0);
        assert_eq!(profile(100.0, r_in, r_out), 0.0);
        // positive inside, peak value ≈ 1, single interior maximum
        let vals: Vec<f64> = (1..400)
            .map(|i| profile(r_in + (r_out - r_in) * i as f64 / 400.0, r_in, r_out))
            .collect();
        assert!(vals.iter().all(|&v| v > 0.0));
        let peak = vals.iter().cloned().fold(0.0, f64::max);
        assert!((peak - 1.0).abs() < 0.15, "peak = {peak}");
        let imax = vals
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let sign_changes = vals
            .windows(2)
            .map(|w| (w[1] - w[0]).signum())
            .collect::<Vec<_>>()
            .windows(2)
            .filter(|s| s[0] > 0.0 && s[1] < 0.0)
            .count();
        assert_eq!(sign_changes, 1, "profile must be unimodal (peak at {imax})");
    }
}
