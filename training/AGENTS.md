# training Agent Guide

Repo-specific agent instructions for `training`.
Read the root `AGENTS.md` first for global rules, project structure, and coding standards.

## Repo Overview

Python pipeline for segmentation annotation, data collection, model training, and ONNX export. Contains:

- `annotate_seg.py` — SAM2-based interactive mask annotation (Phase 1).
- `train_segmentor.py` — YOLO11n-seg training from SAM2 masks (Phase 2).
- `collect_pose.py` — Pose data collection with live segmentation via WebSocket controller (Phase 3).
- `model.py` — DINOv2-small + LoRA + regression head for joint angle estimation.
- `dataset.py` — PyTorch Dataset with augmentation and masked base orientation labels.
- `train.py` — Training loop with cosine LR schedule, masked losses, checkpointing (Phase 4).
- `export_onnx.py` — Export pose model to ONNX + metadata JSON (Phase 5).
- `export_seg.py` — Export segmentor to ONNX (Phase 5).

## Agent Rules

1. Follow all rules in the root `AGENTS.md`.
2. **Do not modify `metak-shared/`.** Propose changes via the orchestrator for user review.
3. Read your assignments from `metak-orchestrator/TASKS.md` and update `metak-orchestrator/STATUS.md` when done or blocked.
4. The ONNX output contract is defined in `metak-shared/api-contracts/onnx-contract.md`.
5. The serial protocol for `collect.py` is defined in `metak-shared/api-contracts/serial-protocol.md`.
6. Use `pyenv-venv` for virtual environments per the root `CUSTOM.md`.

## Coding Standards

- Follow the coding standards defined in `metak-shared/coding-standards.md`.
- Python 3.11, type hints encouraged.
- Tests use `pytest`.
- Dependencies in `requirements.txt`.

## Custom Instructions

Read and follow `CUSTOM.md` in this directory for repo-specific custom instructions.
