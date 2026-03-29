# Task Board

## Current Sprint

### T1: [inference] Install Rust and verify compilation
**Epic**: E1.1 | **Repo**: `inference/` | **Status**: In Progress
**Acceptance Criteria**:
- Rust stable toolchain installed
- `cargo check` passes in `inference/`
- `cargo clippy` passes with no errors
- Fix any compilation issues found

### T2: [training] Set up Python environment and verify imports
**Epic**: E1.2 | **Repo**: `training/` | **Status**: In Progress
**Acceptance Criteria**:
- `requirements.txt` created with all dependencies (torch, torchvision, opencv-python, pandas, numpy, onnx, onnxruntime, pyserial)
- Python venv created via `pyenv-venv install vimu 3.11.9`
- All five scripts (`model.py`, `dataset.py`, `train.py`, `collect.py`, `export_onnx.py`) import without error
- `pytest` runnable

### T3: [arduino] Verify sketch and add protocol tests
**Epic**: E1.3, E2.1 | **Repo**: `arduino/` | **Status**: In Progress
**Acceptance Criteria**:
- Sketch compiles with `arduino-cli compile` or review for correctness
- If arduino-cli not available: manual code review for syntax/logic issues
- Create a Python-side protocol test that validates checksum computation and frame encoding/decoding

### T4: [training] Add unit tests
**Epic**: E2.2 | **Repo**: `training/` | **Status**: Pending
**Depends on**: T2
**Acceptance Criteria**:
- `test_model.py`: Model instantiation, forward pass shape, output range
- `test_dataset.py`: Dataset loading with mock data, augmentation, label masking
- `test_export.py`: ONNX export from a fresh model, metadata JSON validation
- All tests pass with `pytest`

### T5: [inference] Add unit tests
**Epic**: E2.3 | **Repo**: `inference/` | **Status**: Pending
**Depends on**: T1
**Acceptance Criteria**:
- `ekf.rs` tests: initialization, predict/update cycle, convergence to known value
- `ws.rs` tests: message serialization matches WebSocket contract
- `pipeline.rs` tests: config construction, dimension naming
- All tests pass with `cargo test`

### T6: [integration] Cross-repo contract verification
**Epic**: E4 | **Status**: Pending
**Depends on**: T2, T4, T5
**Acceptance Criteria**:
- Training export metadata JSON schema matches what inference model.rs parses
- WebSocket StateMessage struct serializes to match the contract
