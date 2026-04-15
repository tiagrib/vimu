# VIMU v2 -- Vision-Based Proprioception

Estimates joint angles, base orientation, velocity, and acceleration
of a hobby servo robot from an external camera. Broadcasts state over
WebSocket for consumption by a separate behavior/motion system.

v2 introduces a segmentation-first pipeline: segment the robot, then
estimate pose from the masked image. This decouples the model from the
background and camera position.

## Documentation

- **[Training Guide](training/GUIDE.md)** -- step-by-step instructions (start here)
- **[Architecture](../docs/vimu/architecture.md)** -- full design rationale, phase details, research references

## Architecture

```
Phase 1-2: Segmentation (SAM2 annotation + YOLO training)
    Video clips  -->  SAM2 (annotate)  -->  masks  -->  YOLO11n-seg training

Phase 3: Pose Data Collection (nuttymoves controller + live segmentation)
    Controller sweeps random poses, webcam captures, YOLO strips background live

Phase 4: Training
    masked frames + labels.csv --> DINOv2-small + LoRA --> checkpoints/best.pt

Phase 5: Export
    best.pt --> vimu_pose.onnx    |    vimu_seg.pt --> vimu_seg.onnx

Phase 6: Inference (Rust)
    Camera --> YOLO seg --> Mask --> DINOv2 --> EKF --> WebSocket (:9001)
    ~11ms per frame --> ~90 FPS
```

## Quick Start

```bash
cd vimu/training/
cp .env.sample .env   # configure VIDEO_DIR, OUTPUT_DIR, etc.

# Phase 1: Annotate segmentation data
python annotate_seg.py --annotate-only
python annotate_seg.py --process-only
python annotate_seg.py --status

# Phase 2: Train segmentor
python train_segmentor.py --data seg_data/ --output vimu_seg.pt

# Phase 3: Collect pose data (with controller running)
python collect_pose.py sweep --calibration calibration.toml --seg-model vimu_seg.pt --camera 0

# Phase 4: Train pose model
python train.py --data ./pose_data --epochs 100

# Phase 5: Export
python export_onnx.py --checkpoint checkpoints/best.pt --output vimu_pose.onnx
python export_seg.py --model vimu_seg.pt --output vimu_seg.onnx

# Phase 6: Inference
cd ../inference && cargo run --release --features camera -- \
    --model ../training/vimu_pose.onnx \
    --seg-model ../training/vimu_seg.onnx \
    --camera 0 --port 9001 --display
```

## WebSocket Message Format

```json
{
  "timestamp": 1.234,
  "fps": 92.3,
  "dims": [
    {"name": "joint_1", "raw": 0.523, "position": 0.518, "velocity": 0.032, "acceleration": -0.104}
  ]
}
```

## EKF Tuning

| Scenario | process_noise | measurement_noise |
|----------|--------------|-------------------|
| Slow poses (accuracy) | 5.0 | 0.02 |
| Normal motion | 10.0 | 0.01 |
| Hopping (responsiveness) | 50.0 | 0.005 |

## Project Structure

```
vimu/
├── training/
│   ├── annotate_seg.py             # Phase 1: SAM2 mask annotation (interactive)
│   ├── train_segmentor.py          # Phase 2: YOLO11n-seg training
│   ├── collect_pose.py             # Phase 3: pose collection with live segmentation
│   ├── model.py                    # DINOv2-small + LoRA + regression head
│   ├── dataset.py                  # Data loader
│   ├── train.py                    # Phase 4: training loop
│   ├── export_onnx.py              # Phase 5: pose model ONNX export
│   ├── export_seg.py               # Phase 5: segmentor ONNX export
│   ├── .env.sample                 # Configuration template
│   └── GUIDE.md                    # Detailed training guide
└── inference/
    ├── Cargo.toml
    └── src/
        ├── main.rs                 # CLI (--model + --seg-model)
        ├── model.rs                # Pose model + SegmentorModel (ONNX)
        ├── ekf.rs                  # Kalman filter
        ├── camera.rs               # OpenCV capture
        ├── ws.rs                   # WebSocket broadcast
        ├── display.rs              # Preview overlay
        └── pipeline.rs             # Main loop
```
