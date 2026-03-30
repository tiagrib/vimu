# VIMU — Project Custom Instructions

For python use pyenv-venv-win to manage virtual environments. Always create a new virtual environment for each project and activate it before installing dependencies. This helps to avoid conflicts between different projects and ensures that each project has its own isolated environment.

The command to list available python versions in pyenv-venv is:
```pyenv-venv list python
```

The command to list existing virtual environments in pyenv-venv is:
```pyenv-venv list envs
```

The command to install a new python version in pyenv is:
```pyenv install <version>
```

The command to create a new virtual environment in pyenv-venv is:
```pyenv-venv install <env-name> <python-version>
```

The command to activate a virtual environment in pyenv-venv is:
```pyenv-venv activate <env-name>
```

## Architecture

VIMU is a standalone vision-based proprioception library. It must NOT depend on nuttymoves or adelino.

Three components:
- `training/` — Python: data collection, model training, ONNX export
- `inference/` — Rust: real-time ONNX inference + EKF + WebSocket broadcast
- `arduino/` — Arduino firmware for standalone data collection (legacy, not needed when using nuttymoves controller)

## Integration Points

- `training/collect.py` has an abstract `RobotController` interface with two backends:
  - `WebSocketController` — talks to any WebSocket controller (e.g., nuttymoves' adelino-standalone)
  - `SerialController` — direct serial to VIMU's own Arduino firmware
- `inference/` broadcasts state via WebSocket. Message format is in `../metak-shared/api-contracts/vimu-websocket.md`.

## Testing

- Python: `cd training && pytest`
- Rust: `cd inference && cargo test`
