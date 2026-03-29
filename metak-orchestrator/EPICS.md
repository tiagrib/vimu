# VIMU Epics

## E1: Project Foundation ✓
**Status**: Complete
**Goal**: Ensure all code compiles/runs, environments are set up, dependencies resolved.

- ✅ E1.1: Rust toolchain installed, inference compiles (without camera feature)
- ✅ E1.2: Python venv set up, requirements.txt created, all scripts import cleanly
- ✅ E1.3: Arduino sketch reviewed for correctness (no arduino-cli available)

## E2: Test Coverage ✓
**Status**: Complete
**Goal**: Add unit tests to all three components.

- ✅ E2.1: Arduino — 33 protocol tests (Python)
- ✅ E2.2: Training — 19 tests (model, dataset, ONNX export)
- ✅ E2.3: Inference — 7 tests (EKF math, JSON serialization)

## E3: Documentation ✓
**Status**: Complete
**Goal**: Fill out all metak-shared docs and orchestrator tracking.

- ✅ E3.1: overview.md
- ✅ E3.2: architecture.md
- ✅ E3.3: glossary.md
- ✅ E3.4: API contracts (serial, ONNX, WebSocket)

## E4: Integration Verification ✓
**Status**: Complete
**Goal**: End-to-end verification that training output matches inference input contract.

- ✅ E4.1: ONNX export metadata matches inference ModelMeta struct
- ✅ E4.2: WebSocket StateMessage JSON matches contract (verified by Rust test)

## Known Blockers

- **OpenCV on Windows**: The `opencv` Rust crate requires LLVM/clang dev libraries. Install LLVM and set `LIBCLANG_PATH` to build with `--features camera`.
