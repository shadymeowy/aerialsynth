//! `terrain` — procedural aerial-odometry dataset toolchain.
//!
//! `run`, `tiles`, `view` and `config` read one scenario YAML (`-c` / `--config`; `terrain
//! config` prints a template); `info` takes a tile store or sequence file.

mod commands;
mod preview;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "terrain", version, about = "A procedural planet for aerial vision: terrain tiles, flights and camera datasets", after_help = "Start: `terrain config > my.yaml`, edit, `terrain run -c my.yaml`, `terrain view -c my.yaml`.")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Args, Clone, Debug)]
pub struct Common {
    /// Scenario YAML (every section optional; a camera needs its path and intrinsics).
    #[arg(long, short)]
    pub config: Option<PathBuf>,
    /// Override world.seed.
    #[arg(long)]
    pub seed: Option<u64>,
    /// Number of worker threads (default: all cores).
    #[arg(long, short = 'j')]
    pub threads: Option<usize>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Make a dataset: trajectory → tiles → render → events (or --step …).
    Run(commands::RunArgs),
    /// Plan and generate the flight's tiles (or a region's), list them, or preview them as PNGs.
    Tiles(commands::TilesArgs),
    /// Open the world: a map of the tile store and the camera through the dataset renderer, flown live.
    View(commands::ViewArgs),
    /// Summarize a tile store or a sequence file.
    Info(commands::InfoArgs),
    /// Print a scenario template (--all: every setting; -c: a scenario with its defaults filled in).
    Config(commands::ConfigArgs),
}

fn main() -> anyhow::Result<()> {
    // end quietly when the reader of the output goes away (`terrain config --all | head`)
    #[cfg(unix)]
    // SAFETY: restoring the default disposition of SIGPIPE before any thread starts.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Run(a) => commands::run(a),
        Cmd::Tiles(a) => commands::tiles(a),
        Cmd::View(a) => commands::view(a),
        Cmd::Info(a) => commands::info(a),
        Cmd::Config(a) => commands::config(a),
    }
}
