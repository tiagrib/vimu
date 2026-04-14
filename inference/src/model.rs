use anyhow::{Context, Result};
use ort::{session::Session, value::Value};
use serde::Deserialize;

pub const INPUT_SIZE: usize = 224;
pub const SEG_INPUT_SIZE: usize = 640;

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Deserialize, Clone)]
pub struct ModelMeta {
    pub num_joints: usize,
    pub output_dim: usize,
    pub outputs: Vec<String>,
}

pub struct Model {
    session: Session,
    pub meta: ModelMeta,
}

impl Model {
    /// Load ONNX model with CUDA execution provider + metadata.
    pub fn load(model_path: &str, meta_path: &str) -> Result<Self> {
        let meta_str = std::fs::read_to_string(meta_path)
            .with_context(|| format!("Failed to read metadata: {meta_path}"))?;
        let meta: ModelMeta = serde_json::from_str(&meta_str)?;

        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_execution_providers([
                ort::execution_providers::CUDAExecutionProvider::default().build(),
            ])
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("Failed to load ONNX model: {model_path}"))?;

        log::info!(
            "Model loaded: {} joints, {} outputs, input {}×{}",
            meta.num_joints,
            meta.output_dim,
            INPUT_SIZE,
            INPUT_SIZE,
        );

        Ok(Self { session, meta })
    }

    /// Run inference on a 224×224 RGB byte buffer (HWC layout, u8).
    /// Returns raw model output as Vec<f32>.
    pub fn predict(&mut self, rgb_224: &[u8]) -> Result<Vec<f32>> {
        debug_assert_eq!(rgb_224.len(), INPUT_SIZE * INPUT_SIZE * 3);

        // HWC u8 → CHW f32, ImageNet normalization
        let mut input = vec![0.0f32; 3 * INPUT_SIZE * INPUT_SIZE];
        for c in 0..3 {
            for y in 0..INPUT_SIZE {
                for x in 0..INPUT_SIZE {
                    let src = y * INPUT_SIZE * 3 + x * 3 + c;
                    let dst = c * INPUT_SIZE * INPUT_SIZE + y * INPUT_SIZE + x;
                    let px = rgb_224[src] as f32 / 255.0;
                    input[dst] = (px - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
                }
            }
        }

        let tensor = Value::from_array(
            ndarray::Array4::from_shape_vec((1, 3, INPUT_SIZE, INPUT_SIZE), input)?,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;

        let outputs = self.session.run(ort::inputs![tensor]).map_err(|e| anyhow::anyhow!("{e}"))?;
        let out = outputs[0].try_extract_tensor::<f32>().map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(out.1.to_vec())
    }
}

/// YOLO11n-seg segmentor for real-time robot masking.
pub struct SegmentorModel {
    session: Session,
}

impl SegmentorModel {
    /// Load segmentor ONNX model with CUDA execution provider.
    pub fn load(model_path: &str) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .with_execution_providers([
                ort::execution_providers::CUDAExecutionProvider::default().build(),
            ])
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("{e}"))
            .with_context(|| format!("Failed to load segmentor: {model_path}"))?;

        log::info!("Segmentor loaded: {model_path}");
        Ok(Self { session })
    }

    /// Run segmentation on a 640×480 RGB byte buffer (HWC layout, u8).
    /// Returns a binary mask (Vec<u8>, same H×W, 0 or 255).
    pub fn segment(&mut self, rgb_hwc: &[u8], width: usize, height: usize) -> Result<Vec<u8>> {
        // Resize to SEG_INPUT_SIZE × SEG_INPUT_SIZE and normalize to [0, 1]
        let mut input = vec![0.0f32; 3 * SEG_INPUT_SIZE * SEG_INPUT_SIZE];
        for c in 0..3 {
            for y in 0..SEG_INPUT_SIZE {
                for x in 0..SEG_INPUT_SIZE {
                    let src_x = x * width / SEG_INPUT_SIZE;
                    let src_y = y * height / SEG_INPUT_SIZE;
                    let src_idx = src_y * width * 3 + src_x * 3 + c;
                    let dst_idx = c * SEG_INPUT_SIZE * SEG_INPUT_SIZE + y * SEG_INPUT_SIZE + x;
                    input[dst_idx] = rgb_hwc.get(src_idx).copied().unwrap_or(0) as f32 / 255.0;
                }
            }
        }

        let tensor = Value::from_array(
            ndarray::Array4::from_shape_vec(
                (1, 3, SEG_INPUT_SIZE, SEG_INPUT_SIZE),
                input,
            )?,
        )
        .map_err(|e| anyhow::anyhow!("{e}"))?;

        let outputs = self.session.run(ort::inputs![tensor]).map_err(|e| anyhow::anyhow!("{e}"))?;

        // YOLO seg output: the last output contains per-pixel mask logits.
        let num_outputs = outputs.len();
        let mask_output = &outputs[num_outputs - 1];
        let mask_data = mask_output.try_extract_tensor::<f32>().map_err(|e| anyhow::anyhow!("{e}"))?;
        let mask_flat: Vec<f32> = mask_data.1.to_vec();

        // Resize mask back to original dimensions and threshold
        let mut result = vec![0u8; width * height];
        let mask_h = SEG_INPUT_SIZE;
        let mask_w = SEG_INPUT_SIZE;

        for y in 0..height {
            for x in 0..width {
                let mx = x * mask_w / width;
                let my = y * mask_h / height;
                let idx = my * mask_w + mx;
                if idx < mask_flat.len() && mask_flat[idx] > 0.5 {
                    result[y * width + x] = 255;
                }
            }
        }

        Ok(result)
    }

    /// Apply a binary mask to an RGB buffer in-place (zero out background).
    pub fn apply_mask(rgb: &mut [u8], mask: &[u8], width: usize, height: usize) {
        for y in 0..height {
            for x in 0..width {
                let px = y * width + x;
                if mask[px] == 0 {
                    let i = px * 3;
                    rgb[i] = 0;
                    rgb[i + 1] = 0;
                    rgb[i + 2] = 0;
                }
            }
        }
    }
}
