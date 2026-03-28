use anyhow::{Context, Result};
use opencv::{
    core::{Mat, Size, CV_8UC3},
    imgproc,
    prelude::*,
    videoio,
};

use crate::model::INPUT_SIZE;

pub struct Camera {
    capture: videoio::VideoCapture,
    frame: Mat,
    resized: Mat,
    rgb: Mat,
}

impl Camera {
    pub fn open(device_id: i32, target_fps: u32) -> Result<Self> {
        let mut capture = videoio::VideoCapture::new(device_id, videoio::CAP_V4L2)
            .or_else(|_| videoio::VideoCapture::new(device_id, videoio::CAP_ANY))
            .context("Failed to open camera")?;

        if !capture.is_opened()? {
            anyhow::bail!("Camera {} not available", device_id);
        }

        // Optimize for latency
        capture.set(videoio::CAP_PROP_FRAME_WIDTH, 640.0)?;
        capture.set(videoio::CAP_PROP_FRAME_HEIGHT, 480.0)?;
        capture.set(videoio::CAP_PROP_FPS, target_fps as f64)?;
        capture.set(videoio::CAP_PROP_BUFFERSIZE, 1.0)?;

        // Prefer MJPEG for lower latency
        let _ = capture.set(
            videoio::CAP_PROP_FOURCC,
            videoio::VideoWriter::fourcc('M', 'J', 'P', 'G')? as f64,
        );

        let w = capture.get(videoio::CAP_PROP_FRAME_WIDTH)?;
        let h = capture.get(videoio::CAP_PROP_FRAME_HEIGHT)?;
        let fps = capture.get(videoio::CAP_PROP_FPS)?;
        log::info!("Camera: {w}x{h} @ {fps:.0}fps");

        Ok(Self {
            capture,
            frame: Mat::default(),
            resized: Mat::default(),
            rgb: Mat::default(),
        })
    }

    /// Grab a frame, return preprocessed 224×224 RGB bytes + reference to raw frame.
    pub fn grab(&mut self) -> Result<(Vec<u8>, &Mat)> {
        self.capture.read(&mut self.frame).context("Camera read")?;
        if self.frame.empty() {
            anyhow::bail!("Empty frame");
        }

        imgproc::resize(
            &self.frame,
            &mut self.resized,
            Size::new(INPUT_SIZE as i32, INPUT_SIZE as i32),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        imgproc::cvt_color(&self.resized, &mut self.rgb, imgproc::COLOR_BGR2RGB, 0)?;

        let bytes = self.rgb.data_bytes()?.to_vec();
        Ok((bytes, &self.frame))
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        let _ = self.capture.release();
    }
}
