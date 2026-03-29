# VIMU Training Guide

Step-by-step instructions to go from a bare robot + camera to a working `vimu.onnx` model.

## What You Need

### Hardware
- **Hobby servo robot** with up to 6 joints (any configuration)
- **Arduino** (Uno, Mega, Nano — anything with enough PWM pins)
- **USB webcam** (or laptop camera) with a clear view of the robot
- **USB cable** for Arduino serial connection
- **Stable surface** to mount the camera (tripod recommended)

### Software (already set up if you followed the main README)
- Python 3.11 with the `vimu` virtual environment
- Arduino IDE (for uploading the sketch)

## Phase 1: Arduino Setup

### 1.1 Configure the sketch

Open `arduino/data_collection_controller/data_collection_controller.ino` and edit the configuration section at the top:

```cpp
#define NUM_SERVOS    6    // ← number of joints on YOUR robot
const uint8_t SERVO_PINS[NUM_SERVOS] = {3, 5, 6, 9, 10, 11};  // ← YOUR pin assignments

#define SERVO_MIN_US  500   // ← minimum pulse width for your servos
#define SERVO_MAX_US  2500  // ← maximum pulse width for your servos
```

**How to find your values:**
- `NUM_SERVOS`: Count the servo motors on your robot.
- `SERVO_PINS`: Which Arduino digital pins are wired to each servo's signal wire.
- `SERVO_MIN_US` / `SERVO_MAX_US`: Check your servo datasheet. Most hobby servos use 500–2500μs. Some use 1000–2000μs. Using wrong values can damage servos or cause jitter.

### 1.2 Upload to Arduino

1. Open the `.ino` file in Arduino IDE.
2. Select your board and port.
3. Upload.
4. Note the serial port name (e.g., `COM3` on Windows, `/dev/ttyUSB0` on Linux).

### 1.3 Verify connection

After upload, the built-in LED should blink 3 times. The sketch is now listening for binary commands at 500000 baud.

## Phase 2: Collect Training Data

### 2.1 Configure joint ranges

Open `training/collect.py` and edit `DEFAULT_JOINT_RANGES` to match your robot's mechanical limits:

```python
DEFAULT_JOINT_RANGES = [
    (-1.2, 1.2),   # Joint 1 — (min_radians, max_radians)
    (-0.8, 1.2),   # Joint 2
    (-1.0, 1.0),   # Joint 3
    (-1.2, 1.2),   # Joint 4
    (-1.5, 1.5),   # Joint 5
    (-0.8, 0.8),   # Joint 6
]
```

**Important:** These ranges must stay within what your servos can physically reach. Going beyond mechanical limits will stall servos and produce bad training data. Start conservative and widen later if needed.

### 2.2 Set up the camera

Position the camera so the **entire robot is visible in frame** at all possible poses. Tips:
- Use a fixed mount — the camera must not move between collection and inference.
- Good, even lighting (avoid harsh shadows that change with arm position).
- Plain background if possible (less visual noise for the model).
- The robot should fill roughly 30–60% of the frame.

### 2.3 Activate the Python environment

```bash
pyenv-venv activate vimu
cd training/
```

### 2.4 Run automated sweep collection

```bash
python collect.py sweep \
    --serial COM3 \
    --camera 0 \
    --num-joints 6 \
    --num-poses 1500 \
    --settle 0.6 \
    --output-dir ./data
```

**Parameters:**
| Parameter | What it does | Recommended value |
|-----------|-------------|-------------------|
| `--serial` | Arduino serial port | `COM3` (Windows) or `/dev/ttyUSB0` (Linux) |
| `--camera` | Camera device index | `0` (default webcam), `1`, `2`... |
| `--num-joints` | Must match `NUM_SERVOS` in the Arduino sketch | Your joint count |
| `--num-poses` | Total training frames to collect | 1500–3000 for good results |
| `--settle` | Seconds to wait after each servo command | 0.4–0.8 (heavier robots need more) |
| `--output-dir` | Where to save frames + labels | `./data` |

**What happens:**
1. The script connects to the Arduino and opens the camera.
2. For each pose, it sends random joint angles (within your ranges) to the servos.
3. Waits for settle time (servos stop vibrating).
4. Captures a frame and saves it as a JPEG.
5. Writes the commanded angles to `labels.csv`.
6. Shows a preview window — press `q` to stop early.

**Expected time:** ~1500 poses × 0.6s settle ≈ **15 minutes**.

**Output:**
```
data/
├── frames/
│   ├── 000000.jpg
│   ├── 000001.jpg
│   └── ... (1500 files)
└── labels.csv
```

### 2.5 (Optional) Add tilted base samples

If you want the model to also estimate base orientation (e.g., when the robot is on an uneven surface):

```bash
python collect.py tilted \
    --camera 0 \
    --num-joints 6 \
    --output-dir ./data
```

This is interactive: physically tilt the robot, press SPACE to capture, and enter the angles manually. You need a protractor or IMU to know the actual tilt. Skip this if you only need joint angles.

### 2.6 Verify your data

Check that `data/labels.csv` has the right number of rows and that images look correct:

```bash
python -c "
import pandas as pd
df = pd.read_csv('./data/labels.csv')
print(f'Frames: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(df.describe())
"
```

You should see `frame`, `joint_1` through `joint_N`, `base_roll`, `base_pitch` columns. Joint values should be within your configured ranges.

## Phase 3: Train the Model

### 3.1 Start training

```bash
python train.py \
    --data-dir ./data \
    --num-joints 6 \
    --epochs 100 \
    --batch-size 32
```

**Parameters:**
| Parameter | What it does | Default |
|-----------|-------------|---------|
| `--data-dir` | Path to your collected data | (required) |
| `--num-joints` | Must match collection | 6 |
| `--epochs` | Training iterations over full dataset | 100 |
| `--batch-size` | Images per gradient step | 32 (lower if you run out of GPU memory) |
| `--lr` | Learning rate | 0.001 |
| `--val-split` | Fraction held out for validation | 0.15 |
| `--output-dir` | Where to save checkpoints | `./checkpoints` |

### 3.2 What to expect

The training loop prints per-epoch stats:

```
Epoch   1/100 (12.3s) | Loss 0.2841 | Joint MAE 0.4123 (23.6°) | Base MAE 0.0012 | LR 1.00e-03
Epoch   2/100 (11.8s) | Loss 0.1923 | Joint MAE 0.3012 (17.3°) | ...
...
Epoch  50/100 (11.5s) | Loss 0.0089 | Joint MAE 0.0654 (3.7°)  | ...
Epoch 100/100 (11.4s) | Loss 0.0045 | Joint MAE 0.0412 (2.4°)  | ...
```

**Target: Joint MAE under 5° (0.087 rad).** If you reach this, the model is good. Under 3° is excellent.

**Typical timelines:**
- With GPU (CUDA): ~12s/epoch → **~20 min** for 100 epochs
- CPU only: ~60s/epoch → **~1.5 hours** for 100 epochs

### 3.3 If training isn't converging

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| MAE stuck above 15° | Too few training samples | Collect more poses (3000+) |
| MAE stuck above 10° | Camera moved during collection | Recollect with fixed camera |
| MAE oscillating wildly | Learning rate too high | Add `--lr 0.0003` |
| Loss is NaN | Bad data (corrupted images) | Check `data/frames/` for broken JPEGs |
| GPU out of memory | Batch size too large | Try `--batch-size 16` or `--batch-size 8` |

### 3.4 Output

Training saves to `checkpoints/`:
```
checkpoints/
├── best.pt    ← best validation MAE (this is what you export)
└── log.csv    ← per-epoch metrics for plotting
```

## Phase 4a: Export to ONNX

### 4.1 Export

```bash
python export_onnx.py \
    --checkpoint ./checkpoints/best.pt \
    --output ./vimu.onnx
```

### 4.2 What it produces

```
vimu.onnx   ← the model (inference engine loads this)
vimu.json   ← metadata (dimension names, normalization constants)
```

### 4.3 Verify the export

The script automatically:
1. Validates the ONNX file structure.
2. Runs a test inference with random input.
3. Checks output shape matches expectations.

If it prints "ONNX validated" — you're good.

## Phase 4b: Run Inference

```bash
cd ../inference
cargo run --release --features camera -- \
    --model ../training/vimu.onnx \
    --camera 0 \
    --port 9001 \
    --display
```

You should see a preview window with joint angle overlays and FPS counter. Connect a WebSocket client to `ws://localhost:9001` to receive the state stream.

## Quick Reference

```
# Full pipeline from scratch:
pyenv-venv activate vimu
cd training/

# 1. Collect (~15 min)
python collect.py sweep --serial COM3 --camera 0 --num-joints 6 --num-poses 1500 --output-dir ./data

# 2. Train (~20 min with GPU)
python train.py --data-dir ./data --num-joints 6 --epochs 100

# 3. Export (<1 min)
python export_onnx.py --checkpoint ./checkpoints/best.pt --output ./vimu.onnx

# 4. Run inference
cd ../inference
cargo run --release --features camera -- --model ../training/vimu.onnx --camera 0 --port 9001 --display
```
