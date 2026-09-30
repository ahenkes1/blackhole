//! End-to-end: bake a map, evaluate a frame, render it — the image must
//! contain background, midtones and the bright ring, and nothing may panic
//! at degenerate sizes.

use blackhole::{disk, map, render};

fn frame(cols: usize, rows: usize, style: render::Style, t: f64) -> String {
    let (w, h, aspect) = style.sample_dims(cols, rows);
    let scene = map::Scene {
        px_aspect: aspect,
        ..map::Scene::default()
    };
    let m = map::build(&scene, w, h);
    let mut raw = vec![0.0f32; w * h];
    disk::Animator::new(&m).evaluate(&m, t, &mut raw);
    let q = render::Quantizer::new(0.9);
    let mut cells = vec![0u8; cols * rows];
    match style {
        render::Style::Ascii => q.ascii_cells(&raw, &mut cells),
        render::Style::Braille => q.braille_cells(&raw, w, h, &mut cells),
    }
    let mut f = Vec::new();
    render::encode_full(style, &cells, cols, b"\r\n", &mut f);
    String::from_utf8(f).unwrap()
}

#[test]
fn ascii_frame_shows_a_black_hole() {
    let f = frame(120, 40, render::Style::Ascii, 0.0);
    assert!(f.contains(' '), "needs dark background");
    assert!(
        f.contains('#') || f.contains('%') || f.contains('@'),
        "needs a bright ring"
    );
    let distinct = {
        let mut cs: Vec<char> = f.chars().filter(|c| *c != '\r' && *c != '\n').collect();
        cs.sort_unstable();
        cs.dedup();
        cs.len()
    };
    assert!(
        distinct >= 4,
        "needs midtones, got {distinct} distinct chars"
    );
}

#[test]
fn animation_actually_animates() {
    let a = frame(80, 24, render::Style::Ascii, 0.0);
    let b = frame(80, 24, render::Style::Ascii, 30.0);
    assert_ne!(a, b, "frames at different times must differ");
}

#[test]
fn degenerate_sizes_do_not_panic() {
    for (c, r) in [(1, 1), (2, 1), (3, 2), (500, 150)] {
        let _ = frame(c, r, render::Style::Ascii, 1.0);
    }
    let _ = frame(2, 1, render::Style::Braille, 1.0);
    let _ = frame(40, 12, render::Style::Braille, 1.0);
}
