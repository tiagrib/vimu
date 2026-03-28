# VIMU — Vision-Based Proprioception

Estimates joint angles, base orientation, velocity, and acceleration
of a hobby servo robot from an external camera. Broadcasts state over
WebSocket for consumption by a separate behavior/motion system.

## Architecture

```
Phase 1-2: Data Collection (Python + Arduino)
┌──────────┐    serial     ┌─────────┐     camera    ┌──────────┐
│  Python  │──────────────→│ Arduino │               │  Webcam  │
│ collect  │  binary proto │ servos  │               │          │
│          │←──────────────│         │               │          │
└────┬─────┘               └─────────┘               └────┬─────┘
     │                                                     │
     └──────────────── labels.csv + frames/ ───────────────┘

Phase 3: Training (Python)
    frames/ + labels.csv → ResNet-18 → checkpoints/best.pt → vimu.onnx

Phase 4: Inference (Rust)
┌─────────────────────────────────────────────────────┐
│  vimu binary                                        │
│                                                     │
│  Camera ──→ ONNX Runtime (GPU) ──→ EKF ──→ WebSocket│
│  ~2ms         ~3-5ms              <0.1ms    broadcast│
│                                                     │
│  Total: ~7ms per frame → 100+ FPS                   │
└─────────────────────┬───────────────────────────────┘
                      │ ws://localhost:9001
                      ▼
              [ Your behavior system ]
```

## WebSocket Message Format

Each frame produces a JSON message:

```json
{
  "timestamp": 1.234,
  "fps": 92.3,
  "dims": [
    {
      "name": "joint_1",
      "raw": 0.523,
      "position": 0.518,
      "velocity": 0.032,
      "acceleration": -0.104
    },
    {
      "name": "joint_2",
      "raw": -0.291,
      "position": -0.287,
      "velocity": -0.015,
      "acceleration": 0.042
    },
    ...
    {
      "name": "base_roll",
      "raw": 0.012,
      "position": 0.010,
      "velocity": 0.001,
      "acceleration": -0.003
    },
    {
      "name": "base_pitch",
      "raw": -0.034,
      "position": -0.031,
      "velocity": -0.008,
      "acceleration": 0.005
    }
  ]
}
```

- `raw`: direct model output (noisy)
- `position`: EKF-filtered value (smooth)
- `velocity`: first derivative estimate (rad/s)
- `acceleration`: second derivative estimate (rad/s²)
- All angles in radians

## Setup & Workflow

### Prerequisites

```bash
# Python (training)
pip install torch torchvision opencv-python pandas numpy onnx onnxruntime-gpu

# Rust (inference)
# - Rust toolchain: https://rustup.rs
# - OpenCV dev: sudo apt install libopencv-dev
# - CUDA toolkit for GPU inference
# - ONNX Runtime is fetched automatically by the ort crate
```

### Phase 1: Arduino Setup

1. Open `arduino/data_collection_controller/data_collection_controller.ino`
2. Set `NUM_SERVOS`, `SERVO_PINS[]`, `SERVO_MIN_US`, `SERVO_MAX_US`
3. Upload to your Arduino
4. Note the serial port (e.g., `/dev/ttyUSB0`)

### Phase 2: Collect Training Data

```bash
cd training/

# Automated sweep: robot cycles through random poses
python collect.py sweep \
    --serial /dev/ttyUSB0 \
    --camera 0 \
    --num-joints 6 \
    --num-poses 1500 \
    --settle 0.6 \
    --output-dir ./data

# Optional: add tilted base samples for orientation training
python collect.py tilted \
    --camera 0 \
    --num-joints 6 \
    --output-dir ./data
```

Customize `DEFAULT_JOINT_RANGES` in `collect.py` to match your
robot's mechanical limits per joint.

### Phase 3: Train

```bash
python train.py \
    --data-dir ./data \
    --num-joints 6 \
    --epochs 100 \
    --batch-size 32

# Target: joint MAE under 5° (0.087 rad)
```

### Phase 4a: Export to ONNX

```bash
python export_onnx.py \
    --checkpoint ./checkpoints/best.pt \
    --output ./vimu.onnx

# Produces: vimu.onnx + vimu.json (metadata)
```

### Phase 4b: Build & Run Inference

```bash
cd inference/
cargo build --release

./target/release/vimu \
    --model ../training/vimu.onnx \
    --camera 0 \
    --port 9001 \
    --fps 60 \
    --display   # optional preview window
```

### Connect Your Client

```javascript
const ws = new WebSocket("ws://localhost:9001");
ws.onmessage = (event) => {
    const state = JSON.parse(event.data);
    // state.dims[0].position  → joint_1 angle (filtered)
    // state.dims[0].velocity  → joint_1 angular velocity
    // state.dims[6].position  → base_roll
    // etc.
};
```

```python
import asyncio, websockets, json

async def listen():
    async with websockets.connect("ws://localhost:9001") as ws:
        async for msg in ws:
            state = json.loads(msg)
            joint_angles = [d["position"] for d in state["dims"][:6]]
            base_roll = state["dims"][6]["position"]
            # feed to your behavior system
```

## EKF Tuning

| Scenario | process_noise | measurement_noise |
|----------|--------------|-------------------|
| Slow poses (accuracy) | 5.0 | 0.02 |
| Normal motion | 10.0 | 0.01 |
| Hopping (responsiveness) | 50.0 | 0.005 |

Higher process noise → faster response to sudden changes, noisier derivatives.
Lower measurement noise → trusts model more, but won't correct model errors.

## Project Structure

```
vimu/
├── arduino/
│   └── data_collection_controller/
│       └── data_collection_controller.ino
├── training/
│   ├── collect.py          # Phase 2: data collection
│   ├── model.py            # ResNet-18 + regression head
│   ├── dataset.py          # Data loader
│   ├── train.py            # Phase 3: training loop
│   └── export_onnx.py      # Phase 4a: ONNX export
└── inference/
    ├── Cargo.toml
    └── src/
        ├── main.rs          # CLI
        ├── model.rs         # ONNX GPU inference
        ├── ekf.rs           # Kalman filter
        ├── camera.rs        # OpenCV capture
        ├── ws.rs            # WebSocket broadcast
        └── pipeline.rs      # Main loop
```
