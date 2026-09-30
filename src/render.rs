//! Monochrome rendering: tone mapping plus two output styles.
//!
//! - ASCII: one character per cell from the density ramp [`RAMP`].
//! - Braille: each terminal cell shows a 2×4 dot grid (U+2800 block); the
//!   sample grid is therefore (2·cols)×(4·rows) and grayscale is faked with
//!   4×4 Bayer ordered dithering.
//!
//! A frame is first quantized to one glyph code per terminal cell (the ramp
//! byte, or the braille dot mask) by a [`Quantizer`], then encoded either as
//! a full repaint (rows joined with a separator, no trailing newline) or as
//! a diff against the previous frame: cursor-addressed runs of just the
//! changed cells. No ANSI colors — pure monochrome text.

use std::io::Write as _;

/// Density ramp, dark → bright.
pub const RAMP: &[u8] = b" .:-=+*#%@";

/// 4×4 Bayer matrix, values 0..16 (index [y % 4][x % 4]).
pub const BAYER4: [[u8; 4]; 4] = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]];

/// `(BAYER4 + 0.5) / 16` as f32 — the dot-on thresholds on tone-mapped
/// values (all multiples of 1/32, exactly representable).
pub const BAYER4_THRESH: [[f32; 4]; 4] = [
    [0.03125, 0.53125, 0.15625, 0.65625],
    [0.78125, 0.28125, 0.90625, 0.40625],
    [0.21875, 0.71875, 0.09375, 0.59375],
    [0.96875, 0.46875, 0.84375, 0.34375],
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Style {
    Ascii,
    Braille,
}

impl Style {
    /// Sample-grid dimensions for a terminal of `cols × rows` cells,
    /// and the sample-cell aspect (height/width in pixel-width units):
    /// ASCII 1 sample/cell (aspect 2.0), braille 2×4 samples/cell — braille
    /// dots are approximately square (aspect 1.0).
    pub fn sample_dims(self, cols: usize, rows: usize) -> (usize, usize, f64) {
        match self {
            Style::Ascii => (cols, rows, 2.0),
            Style::Braille => (cols * 2, rows * 4, 1.0),
        }
    }

    /// UTF-8 length of one glyph.
    pub fn glyph_len(self) -> usize {
        match self {
            Style::Ascii => 1,
            Style::Braille => 3,
        }
    }
}

/// Filmic-ish exposure tone map: raw intensity ≥ 0 → [0, 1).
/// The clamp keeps the result strictly below 1.0 where f32 rounding of
/// 1 − exp(−x) would otherwise saturate for large x.
pub fn tone_map(i: f32, exposure: f32) -> f32 {
    (1.0 - (-exposure * i.max(0.0)).exp()).min(1.0 - f32::EPSILON)
}

/// Map v ∈ [0, 1] to a ramp character (monotone in v).
pub fn ramp_char(v: f32) -> u8 {
    let n = RAMP.len();
    let idx = (v.clamp(0.0, 1.0) * (n as f32 - 1.0)).round() as usize;
    RAMP[idx.min(n - 1)]
}

/// Unicode braille char for a 2×4 dot bitmask. Bit layout (col, row):
/// (0,0)=bit0 (0,1)=bit1 (0,2)=bit2 (1,0)=bit3 (1,1)=bit4 (1,2)=bit5
/// (0,3)=bit6 (1,3)=bit7 — i.e. U+2800 + bits.
pub fn braille_cell(bits: u8) -> char {
    char::from_u32(0x2800 + bits as u32).unwrap()
}

/// Tone map and glyph quantization folded into thresholds on *linear*
/// intensity. [`tone_map`] is strictly increasing, so comparing its output
/// against a level is the same as comparing its input against the level's
/// preimage −ln(1 − level)/exposure: no `exp` per sample.
///
/// Equivalent to `ramp_char(tone_map(x))` for ASCII and to
/// `tone_map(x) > BAYER4_THRESH[y%4][x%4]` per braille dot, except possibly
/// for inputs within f32 rounding of a threshold.
#[derive(Debug, Clone)]
pub struct Quantizer {
    /// `ramp[j]`: least intensity drawn as `RAMP[j + 1]`. [`ramp_char`]
    /// rounds tone·(n−1), so level j + 1 starts at tone (j + ½)/(n−1).
    ramp: [f32; RAMP.len() - 1],
    /// Per Bayer position: a dot is on iff intensity > threshold.
    bayer: [[f32; 4]; 4],
}

impl Quantizer {
    pub fn new(exposure: f32) -> Quantizer {
        let e = exposure as f64;
        let preimage = |level: f64| (-(1.0 - level).ln() / e) as f32;
        let n1 = (RAMP.len() - 1) as f64;
        Quantizer {
            ramp: std::array::from_fn(|j| preimage((j as f64 + 0.5) / n1)),
            bayer: BAYER4_THRESH.map(|row| row.map(|t| preimage(t as f64))),
        }
    }

    /// One [`RAMP`] byte per sample of `v` (the ASCII grid is the cell grid).
    pub fn ascii_cells(&self, v: &[f32], cells: &mut [u8]) {
        assert_eq!(v.len(), cells.len());
        for (c, &x) in cells.iter_mut().zip(v) {
            let level: usize = self.ramp.iter().map(|&t| (x >= t) as usize).sum();
            *c = RAMP[level];
        }
    }

    /// Bayer-dithered braille dot masks (see [`braille_cell`]) from the
    /// `w_sub × h_sub` sample grid (w_sub = 2·cols, h_sub = 4·rows): dot
    /// (x, y) is on iff `v > threshold[y % 4][x % 4]`.
    pub fn braille_cells(&self, v: &[f32], w_sub: usize, h_sub: usize, cells: &mut [u8]) {
        assert_eq!(v.len(), w_sub * h_sub);
        assert!(w_sub.is_multiple_of(2) && h_sub.is_multiple_of(4));
        let cols = w_sub / 2;
        assert_eq!(cells.len(), cols * (h_sub / 4));
        const DOTS: [(usize, usize); 8] = [
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
            (0, 3),
            (1, 3),
        ];
        for (cy, row) in cells.chunks_exact_mut(cols).enumerate() {
            // A cell row spans exactly the 4 Bayer rows (y % 4 = dy).
            let band = &v[cy * 4 * w_sub..(cy + 1) * 4 * w_sub];
            for (cx, c) in row.iter_mut().enumerate() {
                let mut bits = 0u8;
                for (bit, &(dx, dy)) in DOTS.iter().enumerate() {
                    let x = cx * 2 + dx;
                    if band[dy * w_sub + x] > self.bayer[dy][x & 3] {
                        bits |= 1 << bit;
                    }
                }
                *c = bits;
            }
        }
    }
}

/// Append the UTF-8 glyph for cell code `code` (a ramp byte for ASCII, a dot
/// mask for braille: U+2800 + mask = E2, A0 | mask>>6, 80 | mask&3F).
#[inline]
fn push_glyph(style: Style, code: u8, dst: &mut Vec<u8>) {
    match style {
        Style::Ascii => dst.push(code),
        Style::Braille => dst.extend_from_slice(&[0xE2, 0xA0 | (code >> 6), 0x80 | (code & 0x3F)]),
    }
}

/// Full frame: rows of `cols` glyphs joined by `sep` ("\r\n" in raw mode),
/// no trailing separator. Appends to `dst`.
pub fn encode_full(style: Style, cells: &[u8], cols: usize, sep: &[u8], dst: &mut Vec<u8>) {
    for (y, row) in cells.chunks_exact(cols).enumerate() {
        if y > 0 {
            dst.extend_from_slice(sep);
        }
        for &c in row {
            push_glyph(style, c, dst);
        }
    }
}

/// Typical cost of a cursor-position escape `ESC[row;colH` in bytes. Gaps of
/// unchanged cells cheaper than this to re-send are bridged into one run.
const CUP_COST: usize = 8;

/// Diff frame: for each run of cells where `cells` differs from `prev`, a
/// cursor-position escape (1-based) followed by the run's glyphs. Unchanged
/// gaps shorter than a cursor escape are re-sent to merge runs. Appends
/// nothing if the frames are identical. Painting the result over a screen
/// showing `prev` leaves it showing `cells`.
pub fn encode_diff(style: Style, cells: &[u8], prev: &[u8], cols: usize, dst: &mut Vec<u8>) {
    assert_eq!(cells.len(), prev.len());
    let max_gap = CUP_COST / style.glyph_len();
    for (y, (cur, old)) in cells
        .chunks_exact(cols)
        .zip(prev.chunks_exact(cols))
        .enumerate()
    {
        let mut x = 0;
        while x < cols {
            if cur[x] == old[x] {
                x += 1;
                continue;
            }
            let start = x;
            let mut end = x + 1; // one past the run's last changed cell
            let mut i = end;
            while i < cols && i - end <= max_gap {
                if cur[i] != old[i] {
                    end = i + 1;
                }
                i += 1;
            }
            let _ = write!(dst, "\x1b[{};{}H", y + 1, start + 1);
            for &c in &cur[start..end] {
                push_glyph(style, c, dst);
            }
            x = end;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tone_map_monotone_bounded() {
        let mut prev = -1.0f32;
        for i in 0..100 {
            let v = tone_map(i as f32 * 0.3, 0.8);
            assert!((0.0..1.0).contains(&v));
            assert!(v >= prev);
            prev = v;
        }
        assert_eq!(tone_map(0.0, 0.8), 0.0);
    }

    #[test]
    fn ramp_monotone_covers_ends() {
        assert_eq!(ramp_char(0.0), b' ');
        assert_eq!(ramp_char(1.0), b'@');
        let mut prev_idx = 0usize;
        for i in 0..=100 {
            let c = ramp_char(i as f32 / 100.0);
            let idx = RAMP.iter().position(|&r| r == c).unwrap();
            assert!(idx >= prev_idx);
            prev_idx = idx;
        }
    }

    #[test]
    fn braille_bit_packing() {
        assert_eq!(braille_cell(0x00), '\u{2800}');
        assert_eq!(braille_cell(0xFF), '\u{28FF}');
        assert_eq!(braille_cell(0x01), '\u{2801}'); // top-left dot only
        for bits in 0..=255u8 {
            let mut v = Vec::new();
            push_glyph(Style::Braille, bits, &mut v);
            assert_eq!(
                v,
                braille_cell(bits).to_string().as_bytes(),
                "mask {bits:#x}"
            );
        }
    }

    /// Inputs this close (relative) to a threshold may round either way.
    fn near_any(x: f32, ts: impl IntoIterator<Item = f32>) -> bool {
        ts.into_iter().any(|t| (x - t).abs() <= 1e-5 * t)
    }

    #[test]
    fn quantizer_matches_tone_map_reference() {
        for exposure in [0.3f32, 0.6, 0.9, 2.5] {
            let q = Quantizer::new(exposure);
            let xs: Vec<f32> = (0..200_000)
                .map(|i| i as f32 * 1e-4)
                .chain([-1.0, 1e9])
                .collect();
            let mut cells = vec![0u8; xs.len()];
            q.ascii_cells(&xs, &mut cells);
            for (&x, &c) in xs.iter().zip(&cells) {
                if !near_any(x, q.ramp) {
                    assert_eq!(
                        c,
                        ramp_char(tone_map(x, exposure)),
                        "x = {x}, exposure {exposure}"
                    );
                }
            }
            let tones: Vec<f32> = xs.iter().map(|&x| tone_map(x, exposure)).collect();
            for (y, row) in BAYER4_THRESH.iter().enumerate() {
                for (x, &t) in row.iter().enumerate() {
                    let lin = q.bayer[y][x];
                    for (&v, &tone) in xs.iter().zip(&tones) {
                        if !near_any(v, [lin]) {
                            assert_eq!(v > lin, tone > t, "v = {v} at ({x},{y})");
                        }
                    }
                }
            }
        }
    }

    fn ascii(v: &[f32], w: usize) -> String {
        let q = Quantizer::new(0.6);
        let mut cells = vec![0u8; v.len()];
        q.ascii_cells(v, &mut cells);
        let mut out = Vec::new();
        encode_full(Style::Ascii, &cells, w, b"\r\n", &mut out);
        String::from_utf8(out).unwrap()
    }

    fn braille(v: &[f32], ws: usize, hs: usize) -> String {
        let q = Quantizer::new(0.6);
        let mut cells = vec![0u8; ws * hs / 8];
        q.braille_cells(v, ws, hs, &mut cells);
        let mut out = Vec::new();
        encode_full(Style::Braille, &cells, ws / 2, b"\r\n", &mut out);
        String::from_utf8(out).unwrap()
    }

    #[test]
    fn ascii_frame_dims() {
        let (w, h) = (7, 3);
        let f = ascii(&vec![0.0; w * h], w);
        let rows: Vec<&str> = f.split("\r\n").collect();
        assert_eq!(rows.len(), h);
        assert!(rows.iter().all(|r| r.len() == w));
        assert!(f.chars().all(|c| c == ' ' || c == '\r' || c == '\n'));
    }

    #[test]
    fn ascii_frame_bright_pixel() {
        let (w, h) = (3, 2);
        let mut v = vec![0.0f32; w * h];
        v[w + 2] = 1e6; // row 1, col 2
        let f = ascii(&v, w);
        let rows: Vec<&str> = f.split("\r\n").collect();
        assert_eq!(rows[1].as_bytes()[2], b'@');
        assert_eq!(rows[0], "   ");
    }

    #[test]
    fn braille_frame_dims_and_extremes() {
        let (cols, rows) = (5, 2);
        let (ws, hs) = (cols * 2, rows * 4);
        let dark = braille(&vec![0.0; ws * hs], ws, hs);
        let lines: Vec<&str> = dark.split("\r\n").collect();
        assert_eq!(lines.len(), rows);
        assert!(lines.iter().all(|l| l.chars().count() == cols));
        assert!(
            dark.chars()
                .all(|c| c == '\u{2800}' || c == '\r' || c == '\n')
        );
        let bright = braille(&vec![1e6; ws * hs], ws, hs);
        assert!(
            bright
                .split("\r\n")
                .all(|l| l.chars().all(|c| c == '\u{28FF}'))
        );
    }

    #[test]
    fn bayer_thresholds_match_matrix() {
        for (y, row) in BAYER4_THRESH.iter().enumerate() {
            for (x, &t) in row.iter().enumerate() {
                assert_eq!(t, (BAYER4[y][x] as f32 + 0.5) / 16.0);
            }
        }
    }

    /// Minimal terminal: applies `ESC[r;cH` and glyphs (1 cell each) to a
    /// grid of chars; returns the screen.
    fn paint(screen: &mut [Vec<char>], bytes: &[u8]) {
        let s = std::str::from_utf8(bytes).unwrap();
        let (mut r, mut c) = (0usize, 0usize);
        let mut it = s.chars().peekable();
        while let Some(ch) = it.next() {
            match ch {
                '\x1b' => {
                    assert_eq!(it.next(), Some('['));
                    let seq: String = it.by_ref().take_while(|&ch| ch != 'H').collect();
                    let (rr, cc) = seq.split_once(';').unwrap();
                    (r, c) = (
                        rr.parse::<usize>().unwrap() - 1,
                        cc.parse::<usize>().unwrap() - 1,
                    );
                }
                '\r' => c = 0,
                '\n' => r += 1,
                _ => {
                    screen[r][c] = ch;
                    c += 1;
                }
            }
        }
    }

    #[test]
    fn diff_repaints_exactly_the_changes() {
        let (cols, rows) = (37usize, 5usize);
        let mut rng = 0x1234_5678u64;
        let mut next = move || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 33) as u32
        };
        for style in [Style::Ascii, Style::Braille] {
            let code = |n: u32| match style {
                Style::Ascii => RAMP[n as usize % RAMP.len()],
                Style::Braille => n as u8,
            };
            let mut prev: Vec<u8> = (0..cols * rows).map(|_| code(next())).collect();
            let mut screen = vec![vec![' '; cols]; rows];
            let mut full = Vec::new();
            encode_full(style, &prev, cols, b"\r\n", &mut full);
            paint(&mut screen, &full);
            for density in [0u32, 1, 5, 30, 100] {
                let cur: Vec<u8> = prev
                    .iter()
                    .map(|&c| {
                        if next() % 100 < density {
                            code(next())
                        } else {
                            c
                        }
                    })
                    .collect();
                let mut diff = Vec::new();
                encode_diff(style, &cur, &prev, cols, &mut diff);
                if cur == prev {
                    assert!(diff.is_empty(), "identical frames must encode to nothing");
                }
                paint(&mut screen, &diff);
                let mut want = vec![vec![' '; cols]; rows];
                let mut f = Vec::new();
                encode_full(style, &cur, cols, b"\r\n", &mut f);
                paint(&mut want, &f);
                assert_eq!(screen, want, "{style:?}, density {density}%");
                if density <= 5 {
                    assert!(diff.len() < f.len() / 2, "sparse diff should be small");
                }
                prev = cur;
            }
        }
    }

    #[test]
    fn diff_bridges_short_gaps() {
        let prev = b"aaaaaaaaaaaaaaaaaaaa".to_vec();
        let mut cur = prev.clone();
        cur[2] = b'b';
        cur[4] = b'b'; // gap of 1 → same run
        cur[18] = b'b'; // gap of 13 > CUP_COST → new run
        let mut out = Vec::new();
        encode_diff(Style::Ascii, &cur, &prev, 20, &mut out);
        assert_eq!(out, b"\x1b[1;3Hbab\x1b[1;19Hb");
    }
}
