//! Terminal frontend: CLI, raw-mode lifecycle, frame loop.
//!
//! Zero dependencies: raw mode via `stty -F /dev/tty` (settings saved with
//! `stty -g` and restored on exit/panic), screen control via ANSI escapes
//! (alternate screen 1049h/l, cursor hide/show and positioning), input via a
//! thread doing blocking 1-byte reads from /dev/tty into an mpsc channel —
//! ANY byte (including Ctrl-C = 0x03 in raw mode) exits cleanly. Terminal
//! size from the TIOCGWINSZ ioctl on /dev/tty, re-polled ~4×/s; on change the
//! lensing map is rebuilt (debounced: skip rebuild until size is stable for
//! two consecutive polls).
//!
//! Frame loop: fixed-period pacing (deadline += period; sleep the remainder),
//! t_sim = elapsed_wall_seconds · speed · TIME_SCALE (f64). Each frame:
//! Animator::evaluate → Quantizer (one glyph per cell) → encode only the
//! cells that changed since the previous frame (full repaint at start, after
//! a resize, and every FULL_REPAINT_SECS) → a single write(2), wrapped in
//! synchronized-output markers so supporting terminals never show a
//! half-drawn frame. At 20 fps only ~1–2 % of cells change per frame, so the
//! terminal parses and redraws roughly a tenth of a full repaint.
//!
//! `--bench` touches no terminal state: it builds the map for the given (or
//! current) size and times it, then renders 200 consecutive frames at the
//! target fps starting an hour into the animation — steady state: by then
//! differential rotation has sheared the texture, and frame-to-frame changes
//! are what the live loop sees — and prints map-build ms, mean frame ms,
//! estimated CPU share, and output bytes/s.

use std::io::Write as _;
use std::os::fd::AsFd as _;
use std::process::Command;
use std::time::{Duration, Instant};

use blackhole::disk;
use blackhole::map::{self, Scene};
use blackhole::render::{self, Style};

/// Sim-time seconds per wall second (before --speed): inner-edge orbital
/// period ≈ 2π/Ω(8) ≈ 142 time units → ≈ 18 s of wall time by default.
const TIME_SCALE: f64 = 8.0;

/// Frame pacing bound: beyond this the fixed-period loop is pure busy-spin.
const MAX_FPS: u32 = 240;
/// r_escape = 2·r_out drives fan integration cost; keep startup bounded.
const MAX_ROUT: f64 = 500.0;

/// Unconditional full-repaint period: repairs anything else that drew on
/// the terminal since the last one.
const FULL_REPAINT_SECS: u64 = 10;
/// Terminal-size polls per second (one ioctl each).
const RESIZE_POLLS_PER_SEC: u32 = 4;
/// `--bench` measures frames starting this far into the animation (wall
/// seconds at the given --speed).
const BENCH_START_SECS: f64 = 3600.0;
/// DEC private mode 2026 "synchronized output": supporting terminals present
/// the enclosed update atomically; others ignore the unknown mode.
const SYNC_BEGIN: &[u8] = b"\x1b[?2026h";
const SYNC_END: &[u8] = b"\x1b[?2026l";

const HELP: &str = "\
blackhole — Schwarzschild accretion-disk screensaver for monochrome terminals

USAGE: blackhole [OPTIONS]           (any key exits)

  --style ascii|braille   renderer (default ascii)
  --fps N                 target frames per second, 1..=240 (default 20)
  --inclination DEG       viewing inclination, 90 = edge-on (default 81)
  --rin R --rout R        disk annulus in units of M (default 8, 44; rout <= 500)
  --stars on|off          lensed background star field (default on)
  --rotation cw|ccw       disk rotation sense (default ccw)
  --beaming N             Doppler beaming exponent g^N (default 3, bolometric 4)
  --zoom F                magnification (default 1.0)
  --speed F               animation speed multiplier (default 1.0)
  --exposure F            tone-map exposure (default 0.6)
  --aspect F              terminal cell height/width (default 2.0)
  --seed N                star-field seed
  --bench                 print precompute/frame timings and exit
  --help                  this text
";

#[derive(Debug, Clone)]
struct Args {
    style: Style,
    fps: u32,
    zoom: f64,
    speed: f64,
    exposure: f32,
    aspect: f64,
    bench: bool,
    scene: Scene,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            style: Style::Ascii,
            fps: 20,
            zoom: 1.0,
            speed: 1.0,
            exposure: 0.6,
            aspect: 2.0,
            bench: false,
            scene: Scene::default(),
        }
    }
}

/// Parse std::env::args(); on --help print HELP and exit(0); on bad input
/// print a one-line error + HELP to stderr and exit(2).
/// --zoom sets scene.fov = Scene::default().fov / zoom; --aspect sets
/// scene.px_aspect for ASCII (braille keeps aspect/2 per dot);
/// --rotation cw → scene.rot = -1.0.
fn parse_args() -> Args {
    match try_parse_args() {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("error: {msg}");
            eprint!("{HELP}");
            std::process::exit(2);
        }
    }
}

fn next_value<T: std::str::FromStr>(
    it: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<T, String> {
    let v = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
    v.parse()
        .map_err(|_| format!("invalid value '{v}' for {flag}"))
}

/// Numeric CLI values: `f{32,64}::from_str` happily accepts "NaN"/"inf",
/// which would silently turn the screen black or white instead of erroring.
trait FiniteNum: std::str::FromStr + Copy {
    fn is_fin(self) -> bool;
}
impl FiniteNum for f64 {
    fn is_fin(self) -> bool {
        self.is_finite()
    }
}
impl FiniteNum for f32 {
    fn is_fin(self) -> bool {
        self.is_finite()
    }
}

fn next_finite<T: FiniteNum>(
    it: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<T, String> {
    let raw = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
    let v: T = raw
        .parse()
        .map_err(|_| format!("invalid value '{raw}' for {flag}"))?;
    if v.is_fin() {
        Ok(v)
    } else {
        Err(format!("invalid value '{raw}' for {flag}: must be finite"))
    }
}

fn try_parse_args() -> Result<Args, String> {
    try_parse_from(std::env::args().skip(1))
}

fn try_parse_from(mut it: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut args = Args::default();
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--help" | "-h" => {
                print!("{HELP}");
                std::process::exit(0);
            }
            "--bench" => args.bench = true,
            "--style" => {
                let v: String = next_value(&mut it, "--style")?;
                args.style = match v.as_str() {
                    "ascii" => Style::Ascii,
                    "braille" => Style::Braille,
                    _ => return Err(format!("--style takes ascii|braille, got '{v}'")),
                };
            }
            "--fps" => {
                let v: u32 = next_value(&mut it, "--fps")?;
                if !(1..=MAX_FPS).contains(&v) {
                    return Err(format!("--fps must be in 1..={MAX_FPS}"));
                }
                args.fps = v;
            }
            "--inclination" => {
                let v: f64 = next_finite(&mut it, "--inclination")?;
                args.scene.incl_deg = v.clamp(0.0, 89.9);
            }
            "--rin" => args.scene.r_in = next_finite(&mut it, "--rin")?,
            "--rout" => args.scene.r_out = next_finite(&mut it, "--rout")?,
            "--stars" => {
                let v: String = next_value(&mut it, "--stars")?;
                args.scene.stars = match v.as_str() {
                    "on" => true,
                    "off" => false,
                    _ => return Err(format!("--stars takes on|off, got '{v}'")),
                };
            }
            "--rotation" => {
                let v: String = next_value(&mut it, "--rotation")?;
                args.scene.rot = match v.as_str() {
                    "cw" => -1.0,
                    "ccw" => 1.0,
                    _ => return Err(format!("--rotation takes cw|ccw, got '{v}'")),
                };
            }
            "--beaming" => args.scene.beaming = next_finite(&mut it, "--beaming")?,
            "--zoom" => {
                let v: f64 = next_finite(&mut it, "--zoom")?;
                if v <= 0.0 {
                    return Err("--zoom must be > 0".into());
                }
                args.zoom = v;
                args.scene.fov = Scene::default().fov / v;
            }
            "--speed" => args.speed = next_finite(&mut it, "--speed")?,
            "--exposure" => {
                let v: f32 = next_finite(&mut it, "--exposure")?;
                if v <= 0.0 {
                    return Err("--exposure must be > 0".into());
                }
                args.exposure = v;
            }
            "--aspect" => {
                let v: f64 = next_finite(&mut it, "--aspect")?;
                if v <= 0.0 {
                    return Err("--aspect must be > 0".into());
                }
                args.aspect = v;
            }
            "--seed" => args.scene.seed = next_value(&mut it, "--seed")?,
            _ => return Err(format!("unknown option '{flag}'")),
        }
    }
    if !(args.scene.r_in >= 6.1 && args.scene.r_in < args.scene.r_out) {
        return Err(format!(
            "disk radii must satisfy 6.1 <= rin < rout <= {MAX_ROUT}"
        ));
    }
    if args.scene.r_out > MAX_ROUT {
        return Err(format!("--rout must be <= {MAX_ROUT}"));
    }
    Ok(args)
}

/// Sample-cell aspect after the --aspect override (see module docs):
/// one ASCII cell is `aspect` high per unit width; a braille dot is half a
/// cell wide and a quarter high → aspect/2.
fn px_aspect_for(style: Style, aspect: f64) -> f64 {
    match style {
        Style::Ascii => aspect,
        Style::Braille => aspect / 2.0,
    }
}

/// RAII guard: enters raw mode + alternate screen on `new`, restores
/// everything on Drop. Also installs a panic hook that restores first so a
/// panic never leaves the terminal unusable. `stty -g` snapshot is taken
/// before changing anything. The `restored` flag makes hook + Drop a single
/// restore (a panic runs the hook, then unwinds into `Drop`).
struct TerminalGuard {
    saved_stty: String,
    restored: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

fn restore_terminal(saved_stty: &str, restored: &std::sync::atomic::AtomicBool) {
    if restored.swap(true, std::sync::atomic::Ordering::SeqCst) {
        return;
    }
    let mut out = std::io::stdout().lock();
    let _ = out.write_all(b"\x1b[?1049l\x1b[?25h");
    let _ = out.flush();
    let _ = Command::new("stty")
        .args(["-F", "/dev/tty", saved_stty])
        .status();
}

impl TerminalGuard {
    fn new() -> std::io::Result<TerminalGuard> {
        let err = |m: &str| std::io::Error::other(m);
        let saved = Command::new("stty")
            .args(["-g", "-F", "/dev/tty"])
            .output()?;
        if !saved.status.success() {
            return Err(err("stty -g failed (no controlling tty?)"));
        }
        let saved_stty = String::from_utf8_lossy(&saved.stdout).trim().to_string();
        if saved_stty.is_empty() {
            return Err(err("empty stty -g snapshot"));
        }
        let status = Command::new("stty")
            .args(["-F", "/dev/tty", "raw", "-echo"])
            .status()?;
        if !status.success() {
            return Err(err("stty raw failed"));
        }
        {
            let mut out = std::io::stdout().lock();
            let _ = out.write_all(b"\x1b[?1049h\x1b[?25l\x1b[2J");
            let _ = out.flush();
        }
        let prev_hook = std::panic::take_hook();
        let saved_for_hook = saved_stty.clone();
        let restored = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let restored_for_hook = restored.clone();
        std::panic::set_hook(Box::new(move |info| {
            restore_terminal(&saved_for_hook, &restored_for_hook);
            prev_hook(info);
        }));
        Ok(TerminalGuard {
            saved_stty,
            restored,
        })
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        restore_terminal(&self.saved_stty, &self.restored);
    }
}

/// (cols, rows) of the controlling terminal; falls back to $COLUMNS/$LINES,
/// then (80, 24).
fn term_size() -> (usize, usize) {
    if let Some(size) = tty_winsize() {
        return size;
    }
    let env_dim = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok());
    if let (Some(cols), Some(rows)) = (env_dim("COLUMNS"), env_dim("LINES")) {
        if cols > 0 && rows > 0 {
            return (cols, rows);
        }
    }
    (80, 24)
}

/// (cols, rows) via the TIOCGWINSZ ioctl on /dev/tty: one syscall, where
/// spawning `stty size` cost ~0.5 ms of fork+exec per poll. std already
/// links libc, so declaring `ioctl` adds no dependency.
#[cfg(any(
    target_os = "linux",
    target_os = "android",
    target_vendor = "apple",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "openbsd",
    target_os = "dragonfly"
))]
fn tty_winsize() -> Option<(usize, usize)> {
    use std::ffi::{c_int, c_ulong};
    use std::os::fd::AsRawFd as _;

    /// `struct winsize` from <sys/ioctl.h>.
    #[repr(C)]
    struct Winsize {
        row: u16,
        col: u16,
        xpixel: u16,
        ypixel: u16,
    }
    extern "C" {
        fn ioctl(fd: c_int, request: c_ulong, ...) -> c_int;
    }
    #[cfg(any(target_os = "linux", target_os = "android"))]
    const TIOCGWINSZ: c_ulong = 0x5413;
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    const TIOCGWINSZ: c_ulong = 0x4008_7468;

    let tty = std::fs::File::open("/dev/tty").ok()?;
    let mut ws = Winsize {
        row: 0,
        col: 0,
        xpixel: 0,
        ypixel: 0,
    };
    // SAFETY: `tty` keeps the fd open for the duration of the call, and
    // TIOCGWINSZ writes exactly one `struct winsize` — four u16, the layout
    // of the repr(C) `Winsize` — through the pointer, which is valid and
    // exclusively borrowed.
    let rc = unsafe { ioctl(tty.as_raw_fd(), TIOCGWINSZ, &mut ws as *mut Winsize) };
    (rc == 0 && ws.row > 0 && ws.col > 0).then_some((ws.col as usize, ws.row as usize))
}

#[cfg(not(any(
    target_os = "linux",
    target_os = "android",
    target_vendor = "apple",
    target_os = "freebsd",
    target_os = "netbsd",
    target_os = "openbsd",
    target_os = "dragonfly"
)))]
fn tty_winsize() -> Option<(usize, usize)> {
    None
}

/// Spawn the blocking /dev/tty reader thread; returns a receiver that yields
/// once per byte read (payload irrelevant — any byte means "exit"). Note:
/// after `rx` is dropped the thread parks forever in `read` — fine for this
/// binary (the process exits right after the frame loop), not a reusable
/// component: `Reader::drop` cannot cancel a blocking read without a second
/// tty handle to shut it down.
fn spawn_input_thread() -> std::sync::mpsc::Receiver<u8> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        use std::io::Read as _;
        let Ok(mut tty) = std::fs::File::open("/dev/tty") else {
            return;
        };
        let mut buf = [0u8; 1];
        loop {
            match tty.read(&mut buf) {
                Ok(n) if n > 0 => {
                    if tx.send(buf[0]).is_err() {
                        return;
                    }
                }
                _ => return,
            }
        }
    });
    rx
}

fn run_bench(args: &Args) {
    let (cols, rows) = term_size();
    let t0 = Instant::now();
    let mut st = FrameState::build(args, cols, rows);
    let build_ms = t0.elapsed().as_secs_f64() * 1e3;

    let dt = args.speed * TIME_SCALE / args.fps as f64;
    let t_start = BENCH_START_SECS * args.speed * TIME_SCALE;
    st.render(t_start);
    let full_len = st.update(true).len();
    let n = 200;
    let mut diff_bytes = 0;
    let t1 = Instant::now();
    for i in 1..=n {
        st.render(t_start + i as f64 * dt);
        diff_bytes += st.update(false).len();
    }
    let frame_ms = t1.elapsed().as_secs_f64() * 1e3 / n as f64;
    let diff_mean = diff_bytes as f64 / n as f64;
    // Per repaint period: one full frame, diffs for the rest.
    let period_frames = (args.fps as u64 * FULL_REPAINT_SECS) as f64;
    let bytes_per_s =
        ((period_frames - 1.0) * diff_mean + full_len as f64) / FULL_REPAINT_SECS as f64;
    println!(
        "size {cols}x{rows} ({}x{} samples, ss {}, {} hits)",
        st.w * st.ss,
        st.h * st.ss,
        st.ss,
        st.map.hits.len(),
    );
    println!("map build: {build_ms:.1} ms");
    println!(
        "frame:     {frame_ms:.3} ms  → {:.2}% of one core at {} fps",
        frame_ms * args.fps as f64 / 10.0,
        args.fps
    );
    println!(
        "output:    {diff_mean:.0} B/frame of changes + {full_len} B full repaint every \
         {FULL_REPAINT_SECS} s → {:.1} KiB/s",
        bytes_per_s / 1024.0
    );
}

struct FrameState {
    style: Style,
    cols: usize,
    rows: usize,
    /// Sample grid: one sample per ASCII cell, 2×4 per braille cell.
    w: usize,
    h: usize,
    /// Supersampling factor folded into the map (anti-aliases the photon
    /// ring, which is thinner than a cell): 3 for ASCII, 2 on huge
    /// terminals; 1 for braille (already 2×4 samples per cell).
    ss: usize,
    map: map::LensingMap,
    anim: disk::Animator,
    quant: render::Quantizer,
    raw: Vec<f32>,
    /// Glyph code per terminal cell of the last rendered frame.
    cells: Vec<u8>,
    /// Glyph codes the terminal currently shows (valid once `painted`).
    shown: Vec<u8>,
    painted: bool,
    /// Reusable terminal-update buffer (filled by `update`).
    out: Vec<u8>,
}

impl FrameState {
    fn build(args: &Args, cols: usize, rows: usize) -> FrameState {
        let (w, h, _) = args.style.sample_dims(cols, rows);
        let ss = match args.style {
            // An ss×ss subcell of an aspect-2 cell keeps aspect 2, so the
            // scene's px_aspect is independent of ss. On huge terminals the
            // cells are small enough that 2×2 anti-aliasing suffices.
            Style::Ascii => {
                if cols * rows > 30_000 {
                    2
                } else {
                    3
                }
            }
            Style::Braille => 1,
        };
        let mut scene = args.scene.clone();
        scene.px_aspect = px_aspect_for(args.style, args.aspect);
        let map = map::build(&scene, w * ss, h * ss).downsample(ss);
        FrameState {
            style: args.style,
            cols,
            rows,
            w,
            h,
            ss,
            anim: disk::Animator::new(&map),
            map,
            quant: render::Quantizer::new(args.exposure),
            raw: vec![0.0; w * h],
            cells: vec![0; cols * rows],
            shown: vec![0; cols * rows],
            painted: false,
            out: Vec::new(),
        }
    }

    /// Evaluate and quantize the frame at sim time `t_sim` into `cells`.
    fn render(&mut self, t_sim: f64) {
        self.anim.evaluate(&self.map, t_sim, &mut self.raw);
        match self.style {
            Style::Ascii => self.quant.ascii_cells(&self.raw, &mut self.cells),
            Style::Braille => self
                .quant
                .braille_cells(&self.raw, self.w, self.h, &mut self.cells),
        }
    }

    /// Terminal update that brings the screen to the last rendered frame: a
    /// full repaint if `full` or nothing was painted yet, otherwise just the
    /// changed cells — empty if none changed.
    fn update(&mut self, full: bool) -> &[u8] {
        self.out.clear();
        self.out.extend_from_slice(SYNC_BEGIN);
        let body = self.out.len();
        if full || !self.painted {
            self.out.extend_from_slice(b"\x1b[H");
            render::encode_full(self.style, &self.cells, self.cols, b"\r\n", &mut self.out);
        } else {
            render::encode_diff(
                self.style,
                &self.cells,
                &self.shown,
                self.cols,
                &mut self.out,
            );
        }
        if self.out.len() == body {
            self.out.clear();
        } else {
            self.out.extend_from_slice(SYNC_END);
        }
        self.shown.copy_from_slice(&self.cells);
        self.painted = true;
        &self.out
    }
}

fn main() {
    let args = parse_args();
    if args.bench {
        run_bench(&args);
        return;
    }

    let guard = match TerminalGuard::new() {
        Ok(g) => g,
        Err(_) => {
            // No controlling tty (piped/redirected): emit one frame and exit
            // instead of hanging in a screensaver loop.
            let (cols, rows) = term_size();
            let mut st = FrameState::build(&args, cols, rows);
            st.render(0.0);
            let mut out = Vec::new();
            render::encode_full(args.style, &st.cells, cols, b"\n", &mut out);
            out.push(b'\n');
            let _ = std::io::stdout().write_all(&out);
            return;
        }
    };

    // One write(2) per frame: a dup of stdout as a plain File bypasses
    // stdout's LineWriter, which split each frame at its last newline into
    // three writes that the terminal could draw between.
    let mut term = std::fs::File::from(
        std::io::stdout()
            .as_fd()
            .try_clone_to_owned()
            .expect("dup stdout"),
    );
    let input = spawn_input_thread();
    let period = Duration::from_secs_f64(1.0 / args.fps as f64);
    let full_every = args.fps as u64 * FULL_REPAINT_SECS;
    let poll_every = (args.fps / RESIZE_POLLS_PER_SEC).max(1) as u64;
    let start = Instant::now();
    let (cols, rows) = term_size();
    let mut st = FrameState::build(&args, cols, rows);
    let mut pending_resize: Option<(usize, usize)> = None;
    let mut deadline = Instant::now();
    let mut frame_no: u64 = 0;

    loop {
        if input.try_recv().is_ok() {
            break;
        }
        let t_sim = start.elapsed().as_secs_f64() * args.speed * TIME_SCALE;
        st.render(t_sim);
        let update = st.update(frame_no.is_multiple_of(full_every));
        if !update.is_empty() {
            let _ = term.write_all(update);
        }

        frame_no += 1;
        if frame_no.is_multiple_of(poll_every) {
            // Rebuild only once the new size has been stable for two polls.
            let size = term_size();
            if size != (st.cols, st.rows) {
                if pending_resize == Some(size) {
                    st = FrameState::build(&args, size.0, size.1);
                    pending_resize = None;
                    let _ = term.write_all(b"\x1b[2J");
                } else {
                    pending_resize = Some(size);
                }
            } else {
                pending_resize = None;
            }
        }

        deadline += period;
        let now = Instant::now();
        if deadline > now {
            std::thread::sleep(deadline - now);
        } else {
            deadline = now; // fell behind: don't accumulate frame debt
        }
    }

    drop(guard);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(argv: &[&str]) -> Result<Args, String> {
        try_parse_from(argv.iter().map(|s| (*s).to_string()))
    }

    #[test]
    fn rejects_non_finite_numbers() {
        for (flag, bad) in [
            ("--zoom", "NaN"),
            ("--zoom", "inf"),
            ("--zoom", "-inf"),
            ("--exposure", "NaN"),
            ("--speed", "inf"),
            ("--aspect", "NaN"),
            ("--beaming", "-inf"),
            ("--inclination", "NaN"),
            ("--rin", "inf"),
            ("--rout", "NaN"),
        ] {
            assert!(
                parse(&[flag, bad]).is_err(),
                "{flag} {bad} must be rejected"
            );
        }
    }

    #[test]
    fn fps_bounds() {
        assert!(parse(&["--fps", "0"]).is_err());
        assert!(parse(&["--fps", "241"]).is_err());
        assert!(parse(&["--fps", "240"]).is_ok());
        assert!(parse(&["--fps", "1e39"]).is_err()); // overflows u32
    }

    #[test]
    fn disk_radius_bounds() {
        assert!(parse(&["--rin", "6", "--rout", "44"]).is_err()); // below ISCO-ish
        assert!(parse(&["--rin", "44", "--rout", "8"]).is_err()); // inverted
        assert!(parse(&["--rin", "8", "--rout", "501"]).is_err()); // past cap
        assert!(parse(&["--rin", "6.1", "--rout", "500"]).is_ok());
    }

    #[test]
    fn flags_take_effect() {
        let a = parse(&[
            "--zoom",
            "2",
            "--rotation",
            "cw",
            "--style",
            "braille",
            "--stars",
            "off",
            "--seed",
            "7",
            "--speed",
            "-1.5",
        ])
        .unwrap();
        assert!((a.scene.fov - Scene::default().fov / 2.0).abs() < 1e-12);
        assert_eq!(a.scene.rot, -1.0);
        assert_eq!(a.style, Style::Braille);
        assert!(!a.scene.stars);
        assert_eq!(a.scene.seed, 7);
        assert_eq!(a.speed, -1.5); // reverse animation is legal
    }

    #[test]
    fn unknown_flag_and_bad_values() {
        assert!(parse(&["--nope"]).is_err());
        assert!(parse(&["--fps"]).is_err()); // missing value
        assert!(parse(&["--style", "x11"]).is_err());
        assert!(parse(&["--stars", "maybe"]).is_err());
        assert!(parse(&[]).is_ok());
    }

    #[test]
    fn px_aspect_matches_style() {
        assert_eq!(px_aspect_for(Style::Ascii, 2.0), 2.0);
        assert_eq!(px_aspect_for(Style::Braille, 2.0), 1.0);
        assert_eq!(px_aspect_for(Style::Braille, 1.6), 0.8);
    }
}
