# VIMU — Vision-Based Proprioception

## Goal

Estimate joint angles, base orientation, velocity, and acceleration of a hobby servo robot from an external camera feed. Broadcast the full state over WebSocket at 100+ FPS for consumption by a separate behavior/motion system.

## Phases

1. **Arduino Controller** — Binary-protocol servo controller for automated data collection poses.
2. **Data Collection** (Python) — Sweep through random servo poses, capture camera frames, build `labels.csv`.
3. **Training** (Python) — Train a ResNet-18 regression model on `(frame → joint angles + base orientation)`, export to ONNX.
4. **Inference** (Rust) — Load ONNX model, capture camera frames, run inference with GPU, filter through an Extended Kalman Filter, broadcast state over WebSocket.

## Key Constraints

- Target latency: < 10 ms per frame (camera → WebSocket broadcast).
- Model input: 224×224 RGB, ImageNet-normalized.
- Output dimensions: `num_joints + 2` (last two are `base_roll`, `base_pitch`).
- All angles in radians.
- EKF provides smooth `position`, `velocity`, `acceleration` per dimension.

## Tech Stack

| Component | Language | Key Dependencies |
|-----------|----------|-----------------|
| Arduino controller | C++ (Arduino) | Servo.h |
| Data collection | Python 3.11 | OpenCV, PySerial, NumPy |
| Training pipeline | Python 3.11 | PyTorch, torchvision, ONNX |
| Inference engine | Rust | ort (ONNX Runtime), OpenCV, tokio, tungstenite, nalgebra |
