use anyhow::Result;
use serde::Serialize;
use std::time::Instant;

use crate::camera::Camera;
use crate::ekf::PoseEkf;
use crate::model::Model;
use crate::ws::WsServer;

pub struct Config {
    pub model_path: String,
    pub meta_path: String,
    pub camera_id: i32,
    pub ws_port: u16,
    pub target_fps: u32,
    pub process_noise: f64,
    pub measurement_noise: f64,
    pub display: bool,
}

/// WebSocket message payload — one per frame.
#[derive(Serialize)]
struct StateMessage {
    /// Seconds since inference started
    timestamp: f64,
    /// Measured FPS
    fps: f64,
    /// Per-dimension state estimates
    dims: Vec<DimState>,
}

#[derive(Serialize)]
struct DimState {
    name: String,
    /// Raw model output (unfiltered)
    raw: f64,
    /// EKF-filtered position
    position: f64,
    /// EKF-estimated velocity
    velocity: f64,
    /// EKF-estimated acceleration
    acceleration: f64,
}

pub fn run(config: Config) -> Result<()> {
    // Load model
    let model = Model::load(&config.model_path, &config.meta_path)?;
    let dim_names = model.meta.outputs.clone();
    let num_dims = model.meta.output_dim;

    // Open camera
    let mut camera = Camera::open(config.camera_id, config.target_fps)?;

    // Initialize EKF
    let mut ekf = PoseEkf::new(
        num_dims,
        1.0 / config.target_fps as f64,
        config.process_noise,
        config.measurement_noise,
    );

    // Start WebSocket server
    let rt = tokio::runtime::Runtime::new()?;
    let ws = WsServer::start(config.ws_port, &rt)?;

    // Main loop
    let frame_duration = std::time::Duration::from_secs_f64(1.0 / config.target_fps as f64);
    let start = Instant::now();
    let mut frame_count: u64 = 0;
    let mut fps = 0.0;
    let mut fps_timer = Instant::now();

    log::info!(
        "Inference running | ws://0.0.0.0:{} | {} dims | press 'q' to stop",
        config.ws_port,
        num_dims,
    );

    loop {
        let loop_start = Instant::now();

        // 1. Capture
        let (rgb_bytes, display_frame) = camera.grab()?;

        // 2. Infer
        let raw = model.predict(&rgb_bytes)?;

        // 3. EKF
        let timestamp = start.elapsed().as_secs_f64();
        let measurement: Vec<f64> = raw.iter().map(|&x| x as f64).collect();
        let state = ekf.update(&measurement, timestamp);

        frame_count += 1;

        // FPS (update every 0.5s)
        if fps_timer.elapsed().as_secs_f64() >= 0.5 {
            fps = frame_count as f64 / start.elapsed().as_secs_f64();
            fps_timer = Instant::now();
        }

        // 4. Build message
        let msg = StateMessage {
            timestamp,
            fps,
            dims: dim_names
                .iter()
                .enumerate()
                .map(|(i, name)| DimState {
                    name: name.clone(),
                    raw: raw[i] as f64,
                    position: state.positions[i],
                    velocity: state.velocities[i],
                    acceleration: state.accelerations[i],
                })
                .collect(),
        };

        // 5. Broadcast
        if let Ok(json) = serde_json::to_string(&msg) {
            ws.broadcast(&json);
        }

        // 6. Optional display
        if config.display {
            if show_preview(display_frame, &dim_names, &state, fps)? {
                break; // 'q' pressed
            }
        }

        // Console status (sparse)
        if frame_count % 120 == 0 {
            log::info!(
                "{fps:.0} FPS | {:.2}s | {} clients",
                timestamp,
                ws.client_count(),
            );
        }

        // Rate limit
        let elapsed = loop_start.elapsed();
        if elapsed < frame_duration {
            std::thread::sleep(frame_duration - elapsed);
        }
    }

    log::info!("Stopped after {} frames", frame_count);
    Ok(())
}

fn show_preview(
    frame: &opencv::core::Mat,
    names: &[String],
    state: &crate::ekf::EkfState,
    fps: f64,
) -> Result<bool> {
    let mut display = frame.clone();
    let font = opencv::imgproc::FONT_HERSHEY_SIMPLEX;
    let green = opencv::core::Scalar::new(0.0, 255.0, 0.0, 0.0);

    opencv::imgproc::put_text(
        &mut display,
        &format!("{fps:.0} FPS"),
        opencv::core::Point::new(10, 20),
        font, 0.5, green, 1, opencv::imgproc::LINE_8, false,
    )?;

    for (i, name) in names.iter().enumerate().take(10) {
        let text = format!(
            "{}: {:+.2} v={:+.2} a={:+.2}",
            name, state.positions[i], state.velocities[i], state.accelerations[i],
        );
        opencv::imgproc::put_text(
            &mut display,
            &text,
            opencv::core::Point::new(10, 40 + i as i32 * 18),
            font, 0.38, green, 1, opencv::imgproc::LINE_8, false,
        )?;
    }

    opencv::highgui::imshow("VIMU", &display)?;
    Ok(opencv::highgui::wait_key(1)? == 'q' as i32)
}
