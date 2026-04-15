# VIMU v2 — Vision-Based Proprioception

Estimates joint angles, base orientation, velocity, and acceleration
of a hobby servo robot from an external camera. Broadcasts state over
WebSocket for consumption by a separate behavior/motion system.

v2 introduces a segmentation-first pipeline: segment the robot, then
estimate pose from the masked image. This decouples the model from the
background and camera position.

## Architecture

```
Phase 1-2: Segmentation (SAM2 annotation + YOLO training)
    Video clips  →  SAM2-tiny (one click)  →  masks  →  YOLO11n-seg training

Phase 3: Pose Data Collection (nuttymoves controller + live segmentation)
┌──────────┐     ws      ┌──────────────┐   serial   ┌─────────┐
│  Python  │────────────→│  adelino-    │───────────→│ Arduino │
│ collect  │  JSON cmds  │  standalone  │  bin proto  │ servos  │
│ _pose    │←────────────│  (Rust)      │←───────────│         │
└────┬─────┘             └──────────────┘             └─────────┘
     │ + YOLO seg live     calibration.toml
     │                                                ┌──────────┐
     └──── pose_data/masked/*.jpg + labels.csv ───────│  Webcam  │
                                                      └──────────┘

Phase 4: Training (Python)
    masked frames + labels.csv → DINOv2-small + LoRA → checkpoints/best.pt

Phase 5: Export
    best.pt → merge LoRA → vimu_pose.onnx
    vimu_seg.pt → vimu_seg.onnx

Phase 6: Inference (Rust)
┌─────────────────────────────────────────────────────────┐
│  vimu binary                                            │
│                                                         │
│  Camera → YOLO seg → Mask → DINOv2 → EKF → WebSocket   │
│  ~2ms     ~3ms       <1ms   ~4ms    <0.1ms  broadcast   │
│                                                         │
│  Total: ~11ms per frame → ~90 FPS                       │
└─────────────────────┬───────────────────────────────────┘
                      │ ws://localhost:9001
                      ▼
              [ Your behavior system ]
```

## Setup & Workflow

### Before you start

Complete the Adelino controller setup first (see `projects/adelino/guides/02-adelino-control.md`):
1. Wire hardware and flash firmware
2. Calibrate servos -- this produces `calibration.toml`
3. Verify the controller runs: `adelino-standalone run --port COM3 --calibration calibration.toml`

The calibration TOML is used throughout the VIMU pipeline to know how many joints the robot has and their safe angular ranges.

### Prerequisites

```bash
# Python (all phases)
# Check https://pytorch.org/get-started/locally/ for your specific PyTorch install command based on your CUDA version

# Windows + CUDA 13:
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Linux + CUDA 13:
pip3 install torch torchvision

# Then other dependencies:
pip install opencv-python pandas numpy onnx onnxruntime-gpu
pip install websocket-client   # for collect_pose.py
pip install transformers peft  # DINOv2 + LoRA
pip install ultralytics        # YOLO11n-seg
pip install git+https://github.com/facebookresearch/sam2.git  # Phase 1 only

# Rust (inference)
# - Rust toolchain: https://rustup.rs
# - OpenCV dev: sudo apt install libopencv-dev
# - CUDA toolkit for GPU inference
# - ONNX Runtime is fetched automatically by the ort crate
```

### Phase 1: Collect Segmentation Data

Film the robot from various angles and environments. No servo control needed --
the robot just sits there in static poses. Use your phone or a handheld camera.

1. Place the robot on a table, record a 30-60s video orbiting around it
2. Move to a different room or lighting, record another orbit
3. Put the robot in 3-5 different poses (manually bend joints), orbit each
4. Save videos to a folder (e.g. `vimu/training/videos/`)

Then annotate with SAM2 (one click per clip):

```bash
cd vimu/training/

python annotate_seg.py --video-dir ./videos/ --output seg_data/
```

This extracts frames and shows the first frame of each video. Click once on the
robot, press Enter, and SAM2 propagates the mask through the entire clip.

### Phase 2: Train Segmentor

```bash
python train_segmentor.py --data seg_data/ --output vimu_seg.pt --epochs 50
```

Target: >95% mIoU on held-out frames.

### Phase 3: Collect Pose Data

Start the nuttymoves controller, then collect from multiple camera angles:

```bash
# Terminal 1: start controller
cd projects/adelino
cargo run --release -p adelino-standalone -- run --port COM3 --calibration calibration.toml

# Terminal 2: collect pose data
cd vimu/training/

# First angle
python collect_pose.py sweep \
    --calibration ../../projects/adelino/target/release/calibration.toml \
    --seg-model vimu_seg.pt \
    --camera 0 --num-poses 500

# Move tripod, then append from a new angle
python collect_pose.py sweep \
    --calibration ../../projects/adelino/target/release/calibration.toml \
    --seg-model vimu_seg.pt \
    --camera 0 --num-poses 500 --append
```

### Phase 4: Train Pose Model

```bash
python train.py --data ./pose_data --epochs 100

# Target: joint MAE under 3° (0.05 rad)
```

The number of joints is auto-detected from `labels.csv`.

### Phase 5: Export to ONNX

```bash
python export_onnx.py --checkpoint checkpoints/best.pt --output vimu_pose.onnx
python export_seg.py --model vimu_seg.pt --output vimu_seg.onnx
```

### Phase 6: Run Inference

```bash
cd inference/
cargo build --release

./target/release/vimu \
    --model ../training/vimu_pose.onnx \
    --seg-model ../training/vimu_seg.onnx \
    --camera 0 --port 9001 --display
```

## WebSocket Message Format

Same as v1 — no changes for downstream consumers:

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
├── arduino/                        # Legacy standalone firmware (not needed with nuttymoves)
├── training/
│   ├── annotate_seg.py             # Phase 1: SAM2 mask annotation
│   ├── train_segmentor.py          # Phase 2: YOLO11n-seg training
│   ├── collect_pose.py             # Phase 3: pose collection with live segmentation
│   ├── model.py                    # DINOv2-small + LoRA + regression head
│   ├── dataset.py                  # Data loader (reads from pose_data/masked/)
│   ├── train.py                    # Phase 4: training loop
│   ├── export_onnx.py              # Phase 5: pose model ONNX export
│   └── export_seg.py               # Phase 5: segmentor ONNX export
└── inference/
    ├── Cargo.toml
    └── src/
        ├── main.rs                 # CLI (--model + --seg-model)
        ├── model.rs                # Pose model + SegmentorModel (ONNX)
        ├── ekf.rs                  # Kalman filter (unchanged)
        ├── camera.rs               # OpenCV capture (unchanged)
        ├── ws.rs                   # WebSocket broadcast (unchanged)
        ├── display.rs              # Preview overlay
        └── pipeline.rs             # Main loop (capture → segment → mask → predict → EKF → broadcast)
```

See `metak-shared/VIMU_v2_Architecture.md` for the full architectural design document.
