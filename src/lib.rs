//! Realistic (general-relativistic) black-hole accretion-disk animation for
//! monochrome terminals, after Florentin Jaffredo's "Black hole rendering
//! with SageMath" notebook, but cheap enough to run as a screensaver.
//!
//! Pipeline:
//!  1. [`geodesic`] — integrate a 1D fan of null geodesics (spherical symmetry
//!     means every pixel reuses a fan ray rotated to its position angle).
//!  2. [`map`] — bake, once per terminal size, a per-cell "lensing map":
//!     every disk crossing of every pixel's ray with its static weight
//!     (emission profile × optical-thickness × relativistic beaming
//!     g^beaming), with the anti-aliasing supersampling folded in.
//!  3. [`disk`] — per frame, evaluate the animated emission texture
//!     (Keplerian differential rotation) through the baked map.
//!  4. [`render`] — quantize intensities to an ASCII ramp or Bayer-dithered
//!     braille glyph per cell, and encode full or changed-cells-only frames.

pub mod disk;
pub mod geodesic;
pub mod map;
pub mod render;
