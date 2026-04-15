# VIMU v2 Training Guide

Step-by-step instructions to go from a bare robot + camera to working ONNX models for real-time inference.

For the full architectural rationale, see [docs/vimu/architecture.md](../../docs/vimu/architecture.md).

## What You Need

### Hardware
- **Hobby servo robot** with up to 6 joints (any configuration)
- **Arduino** with the nuttymoves firmware flashed and calibrated (see [Adelino Control Guide](../../projects/adelino/guides/02-adelino-control.md))
- **USB webcam** (or laptop camera) with a clear view of the robot
- **Phone or handheld camera** for filming segmentation videos (Phase 1)
- **Tripod** for pose data collection (Phase 3)

### Software

```bash
# PyTorch (check https://pytorch.org/get-started/locally/ for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Dependencies
pip install opencv-python pandas numpy onnx onnxruntime-gpu
pip install websocket-client        # for collect_pose.py
pip install transformers peft       # DINOv2 + LoRA
pip install ultralytics             # YOLO11n-seg
pip install git+https://github.com/facebookresearch/sam2.git  # Phase 1 only

# Rust toolchain for inference: https://rustup.rs
```

## Configuration

Copy the env sample and edit to match your setup:

```bash
cd vimu/training/
cp .env.sample .env
```

Edit `.env` to set your video directory and output paths. All commands below assume `.env` is configured, so you can omit `--video-dir` and `--output` flags.

## Phase 1: Collect Segmentation Data

Film the robot from various angles and environments. No servo control needed -- the robot just sits there in static poses. Use your phone or a handheld camera at 30fps.

Tips for good segmentation data:
- Orbit around the robot at different heights (table level, above, below)
- Film in multiple rooms/lighting conditions
- Put the robot in 3-5 different manual poses and orbit each
- 30-60 seconds per video clip, 10-20 clips total

Save videos to the directory configured in `.env` (default: `./adelino_v1/`).

### Annotate

```bash
python annotate_seg.py --annotate-only
```

For each video, the first frame appears. Place points on the robot:
- **Left-click** = foreground (green dot) -- place 5-10 on distinct robot parts
- **Right-click** = background (red dot) -- place on objects SAM2 might confuse for the robot
- **Ctrl+Z** = undo last point
- **Enter** = accept
- **q** = skip video

Annotations are saved to `annotations.json` per video. Re-running loads saved points for editing.

### Process

```bash
# Run with large model (default, best quality)
python annotate_seg.py --process-only

# Or compare with a faster model
python annotate_seg.py --process-only --model tiny
```

Available models: `tiny` (fastest), `small`, `base_plus`, `large` (best, default).

### Check status

```bash
python annotate_seg.py --status
```

Shows which videos are annotated, which models have been run, and frame counts.

### Review and iterate

- Bad video? Delete its folder in `seg_data/<video_name>/` and re-run
- Bad masks from one model? Delete `seg_data/<video_name>/masks/<model>/` and re-process
- Want to refine points? Run `--annotate-only` again -- saved points are pre-loaded

### Output

```
seg_data/
    <video_name>/
        annotations.json            # click points (shared across models)
        frames/*.jpg                # extracted video frames
        masks/
            large/*.png             # masks from SAM2-large
            tiny/*.png              # masks from SAM2-tiny
```

## Phase 2: Train Segmentor

```bash
python train_segmentor.py --data seg_data/ --output vimu_seg.pt --epochs 50
```

Target: >95% mIoU on held-out frames.

## Phase 3: Collect Pose Data

Start the controller, then collect from multiple camera angles:

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

Repeat from 3-5 different camera positions. The segmentor strips backgrounds live, so the pose model trains on clean masked images.

## Phase 4: Train Pose Model

```bash
python train.py --data ./pose_data --epochs 100
```

Target: joint MAE under 3 degrees (0.05 rad). The number of joints is auto-detected from `labels.csv`.

| Symptom | Fix |
|---------|-----|
| MAE stuck above 15 degrees | Collect more poses (3000+) |
| MAE stuck above 10 degrees | Check segmentor quality, re-collect if masks are noisy |
| MAE oscillating | Lower learning rate (`--lr 0.0003`) |
| GPU out of memory | Lower batch size (`--batch-size 16`) |

## Phase 5: Export to ONNX

```bash
python export_onnx.py --checkpoint checkpoints/best.pt --output vimu_pose.onnx
python export_seg.py --model vimu_seg.pt --output vimu_seg.onnx
```

## Phase 6: Run Inference

```bash
cd inference/
cargo build --release --features camera

./target/release/vimu \
    --model ../training/vimu_pose.onnx \
    --seg-model ../training/vimu_seg.onnx \
    --camera 0 --port 9001 --display
```

Connect a WebSocket client to `ws://localhost:9001` to receive the state stream.

## Quick Reference

```bash
cd vimu/training/

# Phase 1: Annotate + process segmentation data
python annotate_seg.py --annotate-only
python annotate_seg.py --process-only
python annotate_seg.py --status

# Phase 2: Train segmentor (~10 min)
python train_segmentor.py --data seg_data/ --output vimu_seg.pt

# Phase 3: Collect pose data (~5 min per angle, repeat 3-5 times)
python collect_pose.py sweep --calibration calibration.toml --seg-model vimu_seg.pt --camera 0 --num-poses 500

# Phase 4: Train pose model (~20 min with GPU)
python train.py --data ./pose_data --epochs 100

# Phase 5: Export (~1 min)
python export_onnx.py --checkpoint checkpoints/best.pt --output vimu_pose.onnx
python export_seg.py --model vimu_seg.pt --output vimu_seg.onnx

# Phase 6: Run inference
cd ../inference && cargo run --release --features camera -- --model ../training/vimu_pose.onnx --seg-model ../training/vimu_seg.onnx --camera 0 --port 9001 --display
```
