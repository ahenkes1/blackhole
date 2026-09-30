//! Accretion-disk physics and the animated emission texture.
//!
//! The disk is a thin annulus r ∈ [r_in, r_out] of matter on circular
//! Keplerian orbits, Ω(r) = r^(−3/2) (M = 1). Emission is optically thin;
//! the *static* geometry factors are baked per-hit by [`crate::map`], so the
//! per-frame work here is only:
//!
//! ```text
//!     I(cell) = base + Σ_hits weight · tex(φ₀ − ω·t, t)
//! ```
//!
//! `tex` is the brightness of the co-rotating turbulent pattern at disk
//! azimuth ψ (already advected: ψ = φ₀ − ω t with ω = ±Ω(r_hit) baked into
//! the hit). Differential rotation shears the pattern into spirals, and the
//! slow drift phases make it flicker:
//!
//! ```text
//!     tex(ψ, t) = 1 + Σ_k a_k · sin(m_k·ψ + c_k·t)
//! ```
//!
//! with Σ a_k < 1 so tex > 0 everywhere. [`Animator`] evaluates exactly this
//! sum, restructured for speed and for precision over days of animation.

use crate::map::LensingMap;
use std::f64::consts::TAU;

/// Keplerian angular velocity of a circular orbit at radius r (M = 1).
pub fn omega(r: f64) -> f64 {
    r.powf(-1.5)
}

/// Azimuthal harmonics of the turbulence texture: (m, amplitude, drift rad/s).
pub const HARMONICS: [(f32, f32, f32); 2] = [(3.0, 0.55, 0.11), (7.0, 0.30, -0.07)];

/// Turbulent brightness at advected disk azimuth `psi`, animation time `t`.
/// Strictly positive; ≈ 1 on average. Reference definition: [`Animator`]
/// computes the same function without its f32 precision loss at large `t`.
pub fn tex(psi: f32, t: f32) -> f32 {
    let mut v = 1.0;
    for (m, a, c) in HARMONICS {
        v += a * (m * psi + c * t).sin();
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

/// Spacing of the time origins the [`Animator`] re-anchors to (sim units;
/// 32 s of wall time at default speed). Bounds the kernel's local time
/// |τ| ≤ EPOCH_STEP/2, hence its sine arguments to |m·ψ| ≲ 90.
pub const EPOCH_STEP: f64 = 256.0;

/// Cells per kernel block: the block's per-hit values stay in L1 between
/// the texture pass and the per-cell sums.
const BLOCK: usize = 256;

/// Per-frame evaluator of the animated disk through a [`LensingMap`].
///
/// Precision: the pattern phase m·(φ₀ − ω·t) + c·t grows without bound, and
/// an f32 `t` stops resolving the per-frame step (0.4 sim units at defaults)
/// after a few days. So every hit's phase is re-anchored, in f64, at an epoch
/// T — the multiple of [`EPOCH_STEP`] nearest to t:
///
/// ```text
///     ψ₀ = wrap(φ₀ − ω·T),   drift_k = (c_k·t) mod 2π
/// ```
///
/// and the f32 kernel only sees the small local time τ = t − T. T depends on
/// t alone, so a frame is a pure function of t, whatever was evaluated before.
///
/// Speed: libm `sinf` is an opaque call whose branches mispredict once
/// differential rotation has scrambled the phases (after minutes of runtime,
/// frames got ~2.5× slower than at start-up). The kernel instead streams the
/// structure-of-arrays hits through a branch-free polynomial sine
/// (`fast_sin`), which vectorizes, then sums each cell's hits.
#[derive(Debug, Clone)]
pub struct Animator {
    epoch: f64,
    /// Per-hit pattern azimuth at `epoch`, wrapped to [−π, π].
    psi0: Vec<f32>,
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
            epoch: f64::NAN,
            psi0: vec![0.0; map.hits.len()],
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
            self.psi0.len(),
            map.hits.len(),
            "Animator used with a different map"
        );
        let epoch = (t / EPOCH_STEP).round() * EPOCH_STEP;
        if epoch != self.epoch {
            self.rebase(map, epoch);
        }
        let tau = (t - epoch) as f32;
        let drift = HARMONICS.map(|(_, _, c)| (c as f64 * t).rem_euclid(TAU) as f32);
        let hits = &map.hits;

        for (block, out) in out.chunks_mut(BLOCK).enumerate() {
            let c0 = block * BLOCK;
            let (h0, h1) = (
                map.offsets[c0] as usize,
                map.offsets[c0 + out.len()] as usize,
            );
            let vals = &mut self.vals[..h1 - h0];
            let (w, psi0, om) = (
                &hits.weight[h0..h1],
                &self.psi0[h0..h1],
                &hits.omega[h0..h1],
            );
            for (((v, &w), &p), &o) in vals.iter_mut().zip(w).zip(psi0).zip(om) {
                let psi = p - o * tau;
                let mut tx = 1.0;
                for (k, &(m, a, _)) in HARMONICS.iter().enumerate() {
                    tx += a * fast_sin(m * psi + drift[k]);
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

    fn rebase(&mut self, map: &LensingMap, epoch: f64) {
        for ((p, &phi0), &om) in self
            .psi0
            .iter_mut()
            .zip(&map.hits.phi0)
            .zip(&map.hits.omega)
        {
            let x = phi0 as f64 - om as f64 * epoch;
            *p = (x - (x / TAU).round() * TAU) as f32;
        }
        self.epoch = epoch;
    }
}

/// sin(x) for the texture kernel, branch-free so the per-hit loop vectorizes.
/// Reduces x = q·π + r with |r| ≤ π/2 (π split Cody–Waite style so r stays
/// exact for the kernel's |q| ≲ 30), then an odd degree-9 near-minimax
/// polynomial (fitted on [−π/2, π/2]); sin x = (−1)^q · sin r. Absolute
/// error ≲ 2e-7 for |x| ≲ 1e3.
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
                    let psi = m.hits.phi0[j] as f64 - m.hits.omega[j] as f64 * t;
                    let mut tx = 1.0;
                    for (mk, a, c) in HARMONICS {
                        tx += a as f64 * (mk as f64 * psi + c as f64 * t).sin();
                    }
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
        for i in -200_000..=200_000 {
            let x = i as f32 * 5e-4; // |x| ≤ 100, beyond the kernel's range
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
        let mut min = f32::MAX;
        let mut sum = 0.0f64;
        let n = 10_000;
        for i in 0..n {
            let v = tex(i as f32 * 0.01, i as f32 * 0.003);
            min = min.min(v);
            sum += v as f64;
        }
        assert!(min > 0.0, "tex must stay positive, min = {min}");
        assert!(
            (sum / n as f64 - 1.0).abs() < 0.05,
            "tex should average ≈ 1"
        );
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
