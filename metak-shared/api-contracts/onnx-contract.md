# ONNX Model Contract (Training → Inference)

## Model File: `vimu.onnx`

### Input
- **Name**: `input`
- **Shape**: `[batch, 3, 224, 224]` (dynamic batch)
- **Type**: float32
- **Preprocessing**: `(pixel_u8 / 255.0 - mean) / std`
  - mean: `[0.485, 0.456, 0.406]`
  - std: `[0.229, 0.224, 0.225]`

### Output
- **Name**: `output`
- **Shape**: `[batch, num_joints + 2]`
- **Type**: float32
- **Semantics**: First `num_joints` values are joint angles (radians). Last 2 are `base_roll`, `base_pitch` (radians).

## Metadata File: `vimu.json`

```json
{
  "num_joints": 6,
  "output_dim": 8,
  "input_size": [224, 224],
  "outputs": ["joint_1", "joint_2", ..., "base_roll", "base_pitch"],
  "imagenet_mean": [0.485, 0.456, 0.406],
  "imagenet_std": [0.229, 0.224, 0.225]
}
```
