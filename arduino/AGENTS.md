# arduino Agent Guide

Repo-specific agent instructions for `arduino`.
Read the root `AGENTS.md` first for global rules, project structure, and coding standards.

## Repo Overview

Arduino servo controller for automated training data collection. Implements a binary serial protocol (500k baud) to receive servo position commands from the Python data collection script and report current positions.

Key file: `data_collection_controller/data_collection_controller.ino`

## Agent Rules

1. Follow all rules in the root `AGENTS.md`.
2. **Do not modify `metak-shared/`.** Propose changes via the orchestrator for user review.
3. Read your assignments from `metak-orchestrator/TASKS.md` and update `metak-orchestrator/STATUS.md` when done or blocked.
4. The serial protocol contract is defined in `metak-shared/api-contracts/serial-protocol.md`.
5. This is an Arduino sketch — use Arduino IDE conventions (`.ino` files, `setup()`/`loop()` entry points).

## Coding Standards

- Follow the coding standards defined in `metak-shared/coding-standards.md`.
- Use Arduino-style C++ (camelCase functions, UPPER_CASE constants).
- Keep the sketch self-contained — no external library dependencies beyond `Servo.h`.

## Custom Instructions

Read and follow `CUSTOM.md` in this directory for repo-specific custom instructions.
