//! `terrain` — procedural aerial-odometry dataset toolchain.

mod preview;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "terrain", version, about = "Procedural XYZ terrain tiles + onboard camera renderer")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Generate tiles directly into a PNG mosaic (no HDF5), for inspecting the generator.
    Preview(preview::Args),
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Preview(a) => preview::run(a),
    }
}
