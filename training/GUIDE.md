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
python annotate_seg.py --process-only
```

This runs SAM2-large by default, which gives the best mask quality. If your GPU can handle it (needs ~3-4 GB VRAM), stick with `large` -- there's no reason to use a smaller model unless you're constrained on VRAM or processing time. Smaller models are available (`--model tiny`, `small`, `base_plus`) but will produce lower quality masks that propagate into every downstream phase.

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

### Verify: Compare masks

Before moving on, visually verify the segmentation quality across all videos:

```bash
python compare_masks.py
```

This generates a grid image per model in `seg_data/comparison/`. Each video's masks are stacked into a single frame with a yellow-to-blue gradient (first frame = yellow, last = blue), so you can see at a glance:
- **Solid, consistent shape** = good tracking throughout the video
- **Scattered or fading color** = mask drifted or was lost
- **Yellow only (no blue)** = tracking failed early

Open the grid image and check every video. If any look bad:
1. Delete that video's output folder: `rm -rf seg_data/<video_name>/`
2. Re-annotate with better point placement (`--annotate-only`)
3. Re-process (`--process-only`)

If you tested multiple models, compare their grid images side by side:
```bash
python annotate_seg.py --process-only --model tiny
python compare_masks.py --models large tiny
```

**This is the quality gate for Phase 1.** Only proceed to Phase 2 once all videos show clean, consistent masks in the grid.

## Model Organization

All trained models are stored under a shared `models/` directory (at the repo root by default, or override with `MODELS_DIR` in `.env`). Models are organized by family and variant:

```
models/
  segmentation/                     # family
    large_600frames/                # variant (you choose the name)
      vimu_seg.pt                   # YOLO checkpoint
      vimu_seg.onnx                 # exported ONNX (Phase 5)
    diverse_lighting/               # another variant
      vimu_seg.pt
  pose/                             # future family
    dinov2_v1/
      best.pt
      vimu_pose.onnx
```

List existing variants: `python train_segmentor.py --list`

## Phase 2: Train Segmentor

```bash
python train_segmentor.py --variant large_600frames
```

You choose the variant name -- it should describe what's distinctive about this training run (e.g. `large_600frames`, `diverse_lighting`, `neg_examples_v2`).

The script auto-detects the best available SAM2 masks (prefers `large` > `base_plus` > `small` > `tiny`). Before training, it runs a pre-flight check that verifies every annotated video has matching frames and masks:

```
Pre-flight check (model: large)
Video                      Frames     Masks  Status
-----------------------------------------------------------------
IMG_7975                       152       152  OK
IMG_7976                        98        98  OK
IMG_7977                       120         0  MISSING MASKS
-----------------------------------------------------------------
Total                          370       250  2 videos OK
```

If any video has missing or mismatched data, the script aborts with instructions on how to fix it.

To use a specific SAM2 model's masks: `python train_segmentor.py --variant my_variant --model tiny`

### Reading the training output

YOLO prints several metrics per epoch. Here's what they mean:

**Losses (lower is better):**

| Loss | What it measures |
|------|-----------------|
| `box_loss` | How well the predicted bounding box matches the ground truth (CIoU loss) |
| `seg_loss` | How well the predicted mask matches the ground truth mask (binary cross-entropy) |
| `cls_loss` | Classification confidence -- since we only have one class ("robot"), this drops fast |
| `dfl_loss` | Distribution focal loss -- refines bounding box edge precision |

`sem_loss` is semantic segmentation loss (unused in instance segmentation mode, always 0).

**Validation metrics (higher is better):**

| Metric | What it measures | Target |
|--------|-----------------|--------|
| `Box(P)` | Bounding box precision (% of predictions that are correct) | >0.95 |
| `R` | Recall (% of ground truth objects found) | 1.0 |
| `mAP50` | Mean Average Precision at 50% IoU overlap | >0.99 |
| `mAP50-95` | mAP averaged across IoU thresholds 50%-95% (stricter) | >0.90 |
| `Mask(P)` | Same as Box(P) but for the segmentation mask | >0.95 |
| `Mask mAP50` | Mask quality at 50% IoU | >0.99 |
| `Mask mAP50-95` | Mask quality across IoU thresholds (the key metric) | >0.85 |

**What good training looks like:**
- `seg_loss` and `box_loss` should decrease steadily
- `Mask mAP50` should reach 0.99+ (SAM2 masks are high quality, so YOLO learns quickly)
- `Mask mAP50-95` above 0.85 means the masks are tight, not just roughly correct
- If `R` stays at 1.0, the model never misses the robot -- that's what matters most

**Your results look excellent** -- `mAP50: 0.995` and `Mask mAP50-95: 0.836` at epoch 24 is already very good for this task.

### Verify: Live segmentor test

Before moving to pose data collection, test the segmentor live on camera:

```bash
python test_segmentor.py --variant large_600frames
```

This opens a camera window with a green overlay on the detected robot and an FPS counter. Walk around with the camera, change lighting, move to different rooms -- the segmentor should reliably detect and mask the robot in all conditions it will encounter during Phase 3 and inference.

**What to check:**
- Robot is detected at all angles you plan to collect pose data from
- Mask is tight around the robot, not bleeding onto background objects
- No false positives (random objects detected as robot)
- Detection stays stable when you move the camera quickly

**Controls:** `q` = quit, `r` = start/stop recording.

**If you see false positives or bleeding:** remove the robot from view, press `r` to start recording, move the camera around the problem areas, press `r` to stop. The video is saved directly to your video directory (from `.env`) as a negative example. Then re-run Phase 1 annotation on the new video (skip it with `n` since there's no robot to annotate -- this produces frames with no labels, teaching YOLO "nothing here"), re-process, and retrain.

## Phase 3: Collect Pose Data

Start the controller, then collect from multiple camera angles. Use the segmentor `--variant` you validated in Phase 2:

```bash
# Terminal 1: start controller
cd projects/adelino
cargo run --release -p adelino-standalone -- run --port COM3 --calibration calibration.toml

# Terminal 2: collect pose data
cd vimu/training/

# First angle
python collect_pose.py sweep \ 
    --variant sparse_large_yolo26
    --calibration ../../projects/adelino/target/release/calibration.toml --camera 1 --resolution 1920x1080 \ 
    --num-poses 500 --settle 2.0 --max-delta 0.03

# Move tripod, then append from a new angle
python collect_pose.py sweep \ 
    --variant sparse_large_yolo26
    --calibration ../../projects/adelino/target/release/calibration.toml --camera 1 --resolution 1920x1080 \ 
    --num-poses 500 --settle 2.0 --max-delta 0.03 --append
```

Repeat from 3-5 different camera positions. The segmentor strips backgrounds live, so the pose model trains on clean masked images.

You can also pass a direct checkpoint with `--seg-model /path/to/vimu_seg.pt` if you want to test one outside the standard `models/` layout.

`collect_pose.py` also saves the unsegmented camera frames to `<POSE_DATA_DIR>/<variant>/raw/` alongside the masked images. These are not used for pose training but are invaluable for the segmentor refinement workflow below if you spot detection errors during or after collection.

### Segmentor Refinement (optional)

If the segmentor misses the robot or includes wrong objects in some frames, you can refine it with just those frames without redoing any video annotation:

1. Copy problem frames from `raw/` to a new folder, e.g. `refinement_v1/`
2. Annotate each image individually with positive/negative points (same UI as Phase 1):
   ```bash
   python annotate_seg.py --images-dir ./refinement_v1/ --annotate-only
   python annotate_seg.py --images-dir ./refinement_v1/ --process-only
   ```
   Output goes to `seg_data/refinement_v1/frames/` + `masks/<model>/`, compatible with the training script.
3. Finetune from your existing variant so the model preserves what it already learned:
   ```bash
   python train_segmentor.py --variant sparse_large_v2 --from-variant sparse_large_v1
   ```
   The training picks up both the original video-derived data and the new `refinement_v1/` images automatically.
4. Verify with `test_segmentor.py --variant sparse_large_v2` and iterate if needed.

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
python train_segmentor.py --variant my_variant
python test_segmentor.py --variant my_variant
python train_segmentor.py --list                    # see all variants

# Phase 3: Collect pose data (~5 min per angle, repeat 3-5 times)
python collect_pose.py sweep --variant my_variant --calibration calibration.toml --num-poses 500

# Phase 4: Train pose model (~20 min with GPU)
python train.py --data ./pose_data --epochs 100

# Phase 5: Export (~1 min)
python export_onnx.py --checkpoint checkpoints/best.pt --output vimu_pose.onnx
python export_seg.py --variant my_variant

# Phase 6: Run inference
cd ../inference && cargo run --release --features camera -- --model ../training/vimu_pose.onnx --seg-model ../training/vimu_seg.onnx --camera 0 --port 9001 --display
```
