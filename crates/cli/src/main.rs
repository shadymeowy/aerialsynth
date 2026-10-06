//! `terrain` — procedural aerial-odometry dataset toolchain.
//!
//! Every subcommand reads one scenario YAML (`--config`); see `terrain config` for all options.

mod commands;
mod preview;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "terrain", version, about = "Procedural XYZ terrain tiles + onboard camera renderer")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Args, Clone, Debug)]
pub struct Common {
    /// Scenario YAML (all sections optional).
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
    /// Print the full default scenario YAML (or the merged one with --config).
    Config(commands::ConfigArgs),
    /// Synthesize a flight record (spline path + ODE disturbances) → trajectory.file.
    Traj(commands::TrajArgs),
    /// List the XYZ tiles needed to render every camera along the trajectory.
    Plan(commands::PlanArgs),
    /// Generate tiles into the HDF5 tile store (from the plan, a tile list or a bbox).
    Gen(commands::GenArgs),
    /// Render the sequence file: body poses, IMU, every camera's frame modalities.
    Render(commands::RenderArgs),
    /// Simulate the cameras with an `events` modality (ESIM-style) into the sequence file.
    Events(commands::RenderArgs),
    /// traj (if missing) → plan → gen → render (→ events if enabled).
    Run(commands::RunArgs),
    /// Summarize a tile store or a rendered sequence file.
    Info(commands::InfoArgs),
    /// Generate tiles straight into a PNG mosaic (no HDF5), for inspecting the generator.
    Preview(preview::Args),
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Config(a) => commands::config(a),
        Cmd::Traj(a) => commands::traj(a),
        Cmd::Plan(a) => commands::plan(a),
        Cmd::Gen(a) => commands::gen(a),
        Cmd::Render(a) => commands::render(a),
        Cmd::Events(a) => commands::events(a),
        Cmd::Run(a) => commands::run(a),
        Cmd::Info(a) => commands::info(a),
        Cmd::Preview(a) => preview::run(a),
    }
}
