# System Architecture

## Overview

VIMU is a four-phase pipeline: data collection → training → export → real-time inference.

```
Phase 1-2: Data Collection
┌──────────┐    serial     ┌─────────┐     camera    ┌──────────┐
│  Python  │──────────────→│ Arduino │               │  Webcam  │
│ collect  │  binary proto │ servos  │               │          │
│          │←──────────────│         │               │          │
└────┬─────┘               └─────────┘               └────┬─────┘
     │                                                     │
     └──────────── labels.csv + frames/ ───────────────────┘

Phase 3: Training
    frames/ + labels.csv → ResNet-18 → checkpoints/best.pt → vimu.onnx + vimu.json

Phase 4: Inference
┌─────────────────────────────────────────────────────┐
│  vimu binary (Rust)                                 │
│  Camera → ONNX Runtime (GPU) → EKF → WebSocket     │
│  ~2ms      ~3-5ms              <0.1ms   broadcast   │
│  Total: ~7ms per frame → 100+ FPS                   │
└─────────────────────┬───────────────────────────────┘
                      │ ws://localhost:9001
                      ▼
              [ Client behavior system ]
```

## Service Map

| Folder | Role | Language | Inputs | Outputs |
|--------|------|----------|--------|---------|
| `arduino/` | Servo controller for data collection | C++ (Arduino) | Serial commands (binary protocol) | Servo positions, status ACKs |
| `training/` | Data collection, model training, ONNX export | Python 3.11 | Camera frames + serial link to Arduino | `vimu.onnx` + `vimu.json` |
| `inference/` | Real-time vision inference + state broadcast | Rust | Camera frames + ONNX model | WebSocket JSON messages |

## Data Flow Boundaries

1. **Arduino ↔ Training**: Binary serial protocol (500k baud). Commands: SET_POSITIONS, QUERY_STATUS, DETACH_ALL. See `api-contracts/serial-protocol.md`.
2. **Training → Inference**: ONNX model file (`vimu.onnx`) + metadata JSON (`vimu.json`). See `api-contracts/onnx-contract.md`.
3. **Inference → Clients**: WebSocket JSON messages on `ws://0.0.0.0:9001`. See `api-contracts/websocket-message.md`.

## ADRs

### ADR-001: ResNet-18 as backbone
- **Decision**: Use ResNet-18 pretrained on ImageNet with frozen early layers.
- **Rationale**: Good accuracy/speed tradeoff for 224×224 input. Fine-tuning only blocks 6-7 reduces overfitting on small datasets.

### ADR-002: EKF for state estimation
- **Decision**: Per-dimension constant-acceleration Extended Kalman Filter.
- **Rationale**: Provides smooth position + velocity + acceleration estimates from noisy model output. Tunable process/measurement noise for different motion profiles.

### ADR-003: Rust for inference
- **Decision**: Rust with ONNX Runtime for the real-time inference loop.
- **Rationale**: Predictable latency, zero-cost abstractions, native GPU support via ort crate. Target < 10ms total pipeline latency.
