#[cfg(feature = "camera")]
mod camera;
mod ekf;
mod model;
mod pipeline;
mod ws;

use anyhow::Result;
use clap::Parser;

#[derive(Parser)]
#[command(name = "vimu", about = "Vision-based proprioception — inference server")]
struct Cli {
    /// Path to ONNX model file
    #[arg(short, long)]
    model: String,

    /// Path to model metadata JSON (default: model path with .json extension)
    #[arg(long)]
    meta: Option<String>,

    /// Camera device index
    #[arg(short, long, default_value = "0")]
    camera: i32,

    /// WebSocket server port
    #[arg(short, long, default_value = "9001")]
    port: u16,

    /// Target capture FPS (caps loop rate)
    #[arg(long, default_value = "60")]
    fps: u32,

    /// EKF process noise (higher = more responsive to fast motion)
    #[arg(long, default_value = "10.0")]
    process_noise: f64,

    /// EKF measurement noise (lower = trust model predictions more)
    #[arg(long, default_value = "0.01")]
    measurement_noise: f64,

    /// Show OpenCV preview window
    #[arg(long)]
    display: bool,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let cli = Cli::parse();

    let meta_path = cli.meta.unwrap_or_else(|| cli.model.replace(".onnx", ".json"));

    let config = pipeline::Config {
        model_path: cli.model,
        meta_path,
        camera_id: cli.camera,
        ws_port: cli.port,
        target_fps: cli.fps,
        process_noise: cli.process_noise,
        measurement_noise: cli.measurement_noise,
        display: cli.display,
    };

    #[cfg(feature = "camera")]
    {
        pipeline::run(config)?;
    }

    #[cfg(not(feature = "camera"))]
    {
        let _ = config;
        anyhow::bail!(
            "Built without the 'camera' feature. \
             Rebuild with `cargo build --features camera` to enable the inference pipeline."
        );
    }

    #[allow(unreachable_code)]
    Ok(())
}
