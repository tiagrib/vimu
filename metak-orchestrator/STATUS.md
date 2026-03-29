# Execution Status

## T3 -- Arduino Protocol Review & Tests (COMPLETE)

**Agent:** arduino worker | **Date:** 2026-03-28

### Code Review Findings

1. **Checksum: CONSISTENT.** Both Arduino and Python compute checksum as XOR of cmd/status, length, and all payload bytes.
2. **State machine: CORRECT.** Handles zero-length payloads, oversized payloads, and bad checksums properly. Minor note: no timeout recovery for partial frames (low risk at 500kbaud).
3. **Servo timing: FUNCTIONAL.** Bit-banged PWM with worst case 15ms blocking per 20ms cycle. Acceptable for data collection.

### Tests: 33 passed
- `arduino/test_protocol.py` — checksum round-trips, all 3 commands, edge cases, byte-level verification.

---

## T2 -- Training Environment & Imports (COMPLETE)

**Agent:** training worker | **Date:** 2026-03-28

- Python 3.11.9 venv (`vimu`) created and active
- Torch 2.11.0, torchvision 0.26.0, onnx 1.21.0, onnxruntime 1.24.4 installed
- Added `onnxscript` (required by torch 2.11 ONNX export)
- All 5 scripts import without error
- `requirements.txt` created

---

## T4 -- Training Unit Tests (COMPLETE)

**Agent:** training worker | **Date:** 2026-03-28

### Tests: 19 passed
- `tests/test_model.py` (7) — instantiation, forward pass shapes, frozen/unfrozen params
- `tests/test_dataset.py` (7) — loading, shapes, base_mask correctness, transforms
- `tests/test_export.py` (4) — ONNX creation, validity, metadata JSON, output shape

---

## T1 -- Inference Compilation (COMPLETE)

**Agent:** inference worker | **Date:** 2026-03-28

### Fixes Applied
1. `ort` version: `"2"` → `"2.0.0-rc.12"` (no stable v2 yet)
2. `ndarray`: `"0.16"` → `"0.17"` (ort RC dependency)
3. `opencv` made optional behind `camera` feature flag (needs LLVM/clang on Windows)
4. Fixed ort v2 RC API: `Session::builder()` error handling, `inputs!` macro, `try_extract_tensor()` tuple return, `predict(&mut self)`
5. Fixed clippy lints: `needless_range_loop`, unused import

### Known Blocker
- `opencv` crate requires LLVM/clang dev libraries. Full pipeline needs `cargo build --features camera` with OpenCV installed.

---

## T5 -- Inference Unit Tests (COMPLETE)

**Agent:** inference worker | **Date:** 2026-03-28

### Tests: 7 passed
- `ekf::tests` (5) — initialization, single update, convergence, velocity detection, variable dt
- `pipeline::tests` (2) — DimState and StateMessage JSON serialization match WebSocket contract

---

## T6 -- Integration Contract Verification (COMPLETE)

**Date:** 2026-03-28

- Training export metadata JSON fields (`num_joints`, `output_dim`, `outputs`, `imagenet_mean`, `imagenet_std`) match inference `ModelMeta` struct
- WebSocket `StateMessage` JSON structure matches `api-contracts/websocket-message.md` (verified via `test_state_message_serialization`)
- ONNX input/output names: training exports `image`/`state`, inference loads via positional indexing — compatible
