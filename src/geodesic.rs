//! Null-geodesic integration in Schwarzschild spacetime, G = c = M = 1
//! (horizon r = 2, photon sphere r = 3, ISCO r = 6).
//!
//! Photon orbits are planar. In the orbital plane, u(φ) = 1/r(φ) obeys the
//! Binet equation
//!
//! ```text
//!     d²u/dφ² = 3u² − u
//! ```
//!
//! which we integrate with RK4 and an adaptive φ-step. A ray leaving a static
//! camera at radius `r_cam` at angle `delta` from the *inward radial*
//! direction has conserved impact parameter
//!
//! ```text
//!     b = r_cam · sin(delta) / sqrt(1 − 2/r_cam)
//! ```
//!
//! and the null condition gives, exactly, at every point:
//!
//! ```text
//!     (du/dφ)² = 1/b² − u²(1 − 2u)                                   (∗)
//! ```
//!
//! Initial conditions: u(0) = 1/r_cam, du/dφ(0) = +sqrt of (∗) (inward,
//! φ increases along the ray). Step control: dφ = min(DPHI_MAX,
//! DR_TARGET·u²/|u'|) so that |Δr| ≲ DR_TARGET per step and turning points
//! (u' = 0) stay resolved. Store every accepted step in the polyline.
//!
//! Termination:
//!  - capture:  r ≤ R_CAPTURE (2.05)
//!  - escape:   r ≥ r_escape  → record asymptotic in-plane direction angle
//!  - too many windings: φ ≥ PHI_MAX (6π) → `Fate::MaxTurns` (visually
//!    equivalent to capture; only near-critical rays reach it)
//!
//! In-plane Cartesian convention used throughout the crate: position of a
//! polyline point is P(φ) = r·(ê₁ cos φ + ê₂ sin φ) where ê₁ points from the
//! black hole to the camera and ê₂ is fixed per pixel by `map`. The direction
//! angle of the photon's velocity v = (dr/dφ) ê_r + r ê_φ in that Cartesian
//! frame is θ_v = φ + atan2(r, dr/dφ), which is constant for a straight line
//! — so the total bending is |θ_v(end) − θ_v(start)| (≈ 4/b for large b).

pub const R_CAPTURE: f64 = 2.05;
pub const PHI_MAX: f64 = 6.0 * std::f64::consts::PI;
pub const DPHI_MAX: f64 = 2.0e-2;
pub const DR_TARGET: f64 = 0.35;
/// Critical impact parameter 3·√3: smaller b is captured, larger escapes.
pub const B_CRIT: f64 = 5.196152422706632;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Fate {
    Captured,
    /// Ray reached `r_escape`. `dir_angle` is the in-plane Cartesian polar
    /// angle of its final velocity direction: θ_v = φ_end + atan2(r, dr/dφ).
    Escaped {
        dir_angle: f64,
    },
    MaxTurns,
}

/// One integrated null geodesic, as a polyline (φ strictly increasing).
#[derive(Debug, Clone)]
pub struct Ray {
    pub delta: f64,
    pub b: f64,
    pub phi: Vec<f32>,
    pub r: Vec<f32>,
    pub fate: Fate,
}

impl Ray {
    /// Interpolated radius and exact dr/dφ (sign from the polyline, magnitude
    /// from the null condition (∗)) at orbital angle `phi`, or `None` if the
    /// ray terminated before reaching it.
    pub fn r_at_phi(&self, phi: f64) -> Option<(f64, f64)> {
        if self.phi.len() < 2 || phi < 0.0 || phi > *self.phi.last().unwrap() as f64 {
            return None;
        }
        let i = match self.phi.binary_search_by(|p| (*p as f64).total_cmp(&phi)) {
            Ok(i) => i.min(self.phi.len() - 2),
            Err(i) => i.saturating_sub(1).min(self.phi.len() - 2),
        };
        Some(self.interp(i, phi))
    }

    /// Like [`Ray::r_at_phi`], but reuses `cursor` (a segment index carried by
    /// the caller across increasing queries) instead of binary searching:
    /// O(advance) per call, O(total) for a monotone sweep of φ. The cursor is
    /// clamped and walked in both directions, so it never returns wrong data
    /// for non-monotone use; `phi` outside `[0, phi_end]` returns `None` and
    /// leaves `cursor` untouched.
    pub fn r_at_phi_with_hint(&self, phi: f64, cursor: &mut usize) -> Option<(f64, f64)> {
        let n = self.phi.len();
        if n < 2 || phi < 0.0 || phi > *self.phi.last().unwrap() as f64 {
            return None;
        }
        let max_seg = n - 2;
        let mut i = (*cursor).min(max_seg);
        while i < max_seg && (self.phi[i + 1] as f64) < phi {
            i += 1;
        }
        while i > 0 && (self.phi[i] as f64) > phi {
            i -= 1;
        }
        *cursor = i;
        Some(self.interp(i, phi))
    }

    fn interp(&self, i: usize, phi: f64) -> (f64, f64) {
        let (p0, p1) = (self.phi[i] as f64, self.phi[i + 1] as f64);
        let (r0, r1) = (self.r[i] as f64, self.r[i + 1] as f64);
        let t = if p1 > p0 { (phi - p0) / (p1 - p0) } else { 0.0 };
        let r = r0 + t * (r1 - r0);
        let u = 1.0 / r;
        let du = (1.0 / (self.b * self.b) - u * u * (1.0 - 2.0 * u))
            .max(0.0)
            .sqrt();
        let dr_dphi = (du / (u * u)).copysign(r1 - r0);
        (r, dr_dphi)
    }

    /// Largest orbital angle reached by the polyline.
    pub fn phi_end(&self) -> f64 {
        self.phi.last().copied().unwrap_or(0.0) as f64
    }
}

/// Integrate one backward ray from a static camera at `r_cam`, launched at
/// angle `delta` (radians) from the inward radial direction.
/// `delta < 1e-6` degenerates to a radial ray: returned immediately as
/// `Captured` with a single-point polyline.
pub fn integrate_ray(r_cam: f64, delta: f64, r_escape: f64) -> Ray {
    let b = r_cam * delta.sin() / (1.0 - 2.0 / r_cam).sqrt();
    if delta < 1e-6 {
        return Ray {
            delta,
            b,
            phi: vec![0.0],
            r: vec![r_cam as f32],
            fate: Fate::Captured,
        };
    }
    let mut u = 1.0 / r_cam;
    let mut w = (1.0 / (b * b) - u * u * (1.0 - 2.0 * u)).max(0.0).sqrt();
    let mut phi = 0.0_f64;
    let mut phis = vec![0.0_f32];
    let mut rs = vec![r_cam as f32];
    let u_escape = 1.0 / r_escape;
    let u_capture = 1.0 / R_CAPTURE;
    // Hard backstop far above any physical trajectory length (a full fan ray
    // is a few thousand steps); keeps pathological inputs from spinning.
    let max_steps = 3_000_000;
    let mut fate = Fate::MaxTurns;
    for _ in 0..max_steps {
        let dphi = DPHI_MAX.min(DR_TARGET * u * u / (w.abs() + 1e-9));
        let (k1u, k1w) = (w, 3.0 * u * u - u);
        let (u2, w2) = (u + 0.5 * dphi * k1u, w + 0.5 * dphi * k1w);
        let (k2u, k2w) = (w2, 3.0 * u2 * u2 - u2);
        let (u3, w3) = (u + 0.5 * dphi * k2u, w + 0.5 * dphi * k2w);
        let (k3u, k3w) = (w3, 3.0 * u3 * u3 - u3);
        let (u4, w4) = (u + dphi * k3u, w + dphi * k3w);
        let (k4u, k4w) = (w4, 3.0 * u4 * u4 - u4);
        u += dphi / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u);
        w += dphi / 6.0 * (k1w + 2.0 * k2w + 2.0 * k3w + k4w);
        phi += dphi;
        if !(u.is_finite() && w.is_finite()) {
            // Numerical blow-up: stop before poisoning the polyline with
            // NaN/inf nodes (which downstream comparisons could not see).
            fate = Fate::MaxTurns;
            break;
        }
        if u <= u_escape {
            let ue = u.max(1e-12);
            let r = 1.0 / ue;
            phis.push(phi as f32);
            rs.push(r as f32);
            let dr_dphi = -w / (ue * ue);
            fate = Fate::Escaped {
                dir_angle: phi + r.atan2(dr_dphi),
            };
            break;
        }
        let r = 1.0 / u;
        phis.push(phi as f32);
        rs.push(r as f32);
        if u >= u_capture {
            fate = Fate::Captured;
            break;
        }
        if phi >= PHI_MAX {
            fate = Fate::MaxTurns;
            break;
        }
    }
    Ray {
        delta,
        b,
        phi: phis,
        r: rs,
        fate,
    }
}

/// The 1D fan: rays for view angles delta_i = delta_max · i/(n−1), i = 0..n.
#[derive(Debug)]
pub struct Fan {
    pub rays: Vec<Ray>,
    pub delta_max: f64,
    pub r_cam: f64,
}

impl Fan {
    /// Integrate the whole fan across worker threads. Rays are dealt out
    /// round-robin (thread t takes rays t, t + n_threads, …): the expensive
    /// near-critical rays that wind around the photon sphere sit in one
    /// narrow δ-band, which a contiguous split would hand to a single thread.
    /// Each ray is an independent f64 integration, so the result is
    /// bit-identical to a serial build regardless of thread count.
    pub fn build(r_cam: f64, delta_max: f64, n: usize, r_escape: f64) -> Fan {
        let n = n.max(2);
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(n);
        let mut slots: Vec<Option<Ray>> = vec![None; n];
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..n_threads)
                .map(|t| {
                    s.spawn(move || {
                        (t..n)
                            .step_by(n_threads)
                            .map(|i| {
                                let delta = delta_max * i as f64 / (n - 1) as f64;
                                (i, integrate_ray(r_cam, delta, r_escape))
                            })
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
            for hd in handles {
                for (i, ray) in hd.join().expect("fan build thread panicked") {
                    slots[i] = Some(ray);
                }
            }
        });
        Fan {
            rays: slots
                .into_iter()
                .map(|r| r.expect("every ray is integrated"))
                .collect(),
            delta_max,
            r_cam,
        }
    }

    /// Nearest fan ray for view angle `delta` (clamped).
    pub fn ray_for(&self, delta: f64) -> &Ray {
        let x = (delta / self.delta_max * (self.rays.len() - 1) as f64).round();
        &self.rays[(x.max(0.0) as usize).min(self.rays.len() - 1)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn delta_of_b(b: f64, r_cam: f64) -> f64 {
        (b * (1.0 - 2.0 / r_cam).sqrt() / r_cam).asin()
    }

    fn is_captured(fate: Fate) -> bool {
        matches!(fate, Fate::Captured | Fate::MaxTurns)
    }

    #[test]
    fn critical_impact_parameter() {
        // Bisect the capture/escape threshold in b; must match 3√3 to <0.5 %.
        let (r_cam, r_escape) = (1000.0, 1500.0);
        let mut lo = 4.0; // captured
        let mut hi = 7.0; // escapes
        assert!(is_captured(
            integrate_ray(r_cam, delta_of_b(lo, r_cam), r_escape).fate
        ));
        assert!(matches!(
            integrate_ray(r_cam, delta_of_b(hi, r_cam), r_escape).fate,
            Fate::Escaped { .. }
        ));
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            let ray = integrate_ray(r_cam, delta_of_b(mid, r_cam), r_escape);
            if is_captured(ray.fate) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let b_crit = 0.5 * (lo + hi);
        assert!(
            (b_crit - B_CRIT).abs() / B_CRIT < 5e-3,
            "b_crit = {b_crit}, expected {B_CRIT}"
        );
    }

    #[test]
    fn weak_field_deflection() {
        // Far-field ray, b = 50: bending ≈ 4/b within 10 %.
        let (r_cam, b) = (10_000.0, 50.0);
        let ray = integrate_ray(r_cam, delta_of_b(b, r_cam), r_cam);
        let Fate::Escaped { dir_angle } = ray.fate else {
            panic!("b = 50 must escape, got {:?}", ray.fate)
        };
        let (r0, dr0) = ray.r_at_phi(0.0).unwrap();
        let theta_v0 = 0.0 + r0.atan2(dr0);
        let bending = (dir_angle - theta_v0).abs();
        let expected = 4.0 / b;
        assert!(
            (bending - expected).abs() / expected < 0.10,
            "bending = {bending}, expected ≈ {expected}"
        );
    }

    #[test]
    fn polyline_is_monotone_and_bounded() {
        let ray = integrate_ray(60.0, 0.3, 150.0);
        assert!(ray.phi.len() > 10);
        assert_eq!(ray.phi.len(), ray.r.len());
        assert!((ray.phi[0], ray.r[0] as f64) == (0.0, 60.0));
        for w in ray.phi.windows(2) {
            assert!(w[1] > w[0], "phi must be strictly increasing");
        }
        for &r in &ray.r {
            assert!(r as f64 >= R_CAPTURE * 0.99 && r as f64 <= 150.0 * 1.05);
        }
    }

    #[test]
    fn r_at_phi_matches_nodes() {
        let ray = integrate_ray(60.0, 0.25, 150.0);
        let mid = ray.phi.len() / 2;
        let (r, _) = ray.r_at_phi(ray.phi[mid] as f64).unwrap();
        assert!((r - ray.r[mid] as f64).abs() < 1e-3);
        assert!(ray.r_at_phi(ray.phi_end() + 1.0).is_none());
        assert!(ray.r_at_phi(-0.1).is_none());
    }

    #[test]
    fn radial_ray_degenerates() {
        let ray = integrate_ray(60.0, 0.0, 150.0);
        assert_eq!(ray.fate, Fate::Captured);
        assert_eq!(ray.phi.len(), 1);
        assert!(ray.r_at_phi(0.5).is_none());
    }

    #[test]
    fn r_at_phi_with_hint_matches_binary_search() {
        let ray = integrate_ray(60.0, 0.3, 150.0);
        let mut cursor = 0usize;
        let steps = 512;
        for i in 1..steps {
            let phi = ray.phi_end() * i as f64 / steps as f64;
            let (r_bin, dr_bin) = ray.r_at_phi(phi).unwrap();
            let prev = cursor;
            let (r_hint, dr_hint) = ray.r_at_phi_with_hint(phi, &mut cursor).unwrap();
            assert!(
                (r_bin - r_hint).abs() < 1e-9 && (dr_bin - dr_hint).abs() < 1e-6,
                "mismatch at phi={phi}: ({r_bin},{dr_bin}) vs ({r_hint},{dr_hint})"
            );
            assert!(cursor >= prev, "cursor must advance for increasing phi");
        }
        // Exact node lookups reproduce the stored radii, even with a stale
        // (overshooting) cursor left from the sweep above.
        for (k, &p) in ray.phi.iter().enumerate().step_by(37) {
            let (r, _) = ray.r_at_phi_with_hint(p as f64, &mut cursor).unwrap();
            assert!((r - ray.r[k] as f64).abs() < 1e-4, "node {k}");
        }
        assert!(ray.r_at_phi_with_hint(-0.1, &mut cursor).is_none());
        assert!(
            ray.r_at_phi_with_hint(ray.phi_end() + 1.0, &mut cursor)
                .is_none()
        );
    }

    #[test]
    fn fan_build_is_deterministic() {
        let a = Fan::build(60.0, 0.35, 97, 150.0);
        let b = Fan::build(60.0, 0.35, 97, 150.0);
        assert_eq!(a.rays.len(), 97);
        for (i, ray) in a.rays.iter().enumerate() {
            let delta = 0.35 * i as f64 / 96.0;
            assert_eq!(ray.delta, delta, "ray {i} out of order");
        }
        for i in [0, 48, 96] {
            assert_eq!(a.rays[i].b, b.rays[i].b, "b differs at ray {i}");
            assert_eq!(a.rays[i].phi, b.rays[i].phi, "polyline differs at ray {i}");
            assert_eq!(a.rays[i].r, b.rays[i].r, "polyline differs at ray {i}");
            assert_eq!(a.rays[i].fate, b.rays[i].fate, "fate differs at ray {i}");
        }
        assert_eq!(a.rays[0].b, 0.0); // i = 0 is the radial ray
    }
}
