# inference Agent Guide

Repo-specific agent instructions for `inference`.
Read the root `AGENTS.md` first for global rules, project structure, and coding standards.

## Repo Overview

Rust real-time inference engine. Captures camera frames, runs ONNX model via GPU, filters through an EKF, and broadcasts state over WebSocket. Modules:

- `main.rs` — CLI entry point (clap).
- `model.rs` — ONNX Runtime wrapper with GPU execution.
- `camera.rs` — OpenCV camera capture and preprocessing.
- `ekf.rs` — Per-dimension constant-acceleration Kalman filter.
- `ws.rs` — Tokio-based WebSocket broadcast server.
- `pipeline.rs` — Main inference loop tying everything together.

## Agent Rules

1. Follow all rules in the root `AGENTS.md`.
2. **Do not modify `metak-shared/`.** Propose changes via the orchestrator for user review.
3. Read your assignments from `metak-orchestrator/TASKS.md` and update `metak-orchestrator/STATUS.md` when done or blocked.
4. The ONNX input contract is defined in `metak-shared/api-contracts/onnx-contract.md`.
5. The WebSocket output contract is defined in `metak-shared/api-contracts/websocket-message.md`.

## Coding Standards

- Follow the coding standards defined in `metak-shared/coding-standards.md`.
- Rust 2021 edition, `cargo fmt` and `cargo clippy` must pass.
- Tests use `#[cfg(test)]` modules and `cargo test`.

## Custom Instructions

Read and follow `CUSTOM.md` in this directory for repo-specific custom instructions.
