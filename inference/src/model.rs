use anyhow::{Context, Result};
use ort::{session::Session, value::Value};
use serde::Deserialize;

pub const INPUT_SIZE: usize = 224;

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
