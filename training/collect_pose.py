"""
VIMU v2 Phase 3: Pose Data Collection with Live Segmentation

Commands the robot through random poses via WebSocket (through nuttymoves'
adelino-standalone controller), runs the trained YOLO segmentor live to
strip the background, and saves masked frames + joint angle labels.

Requires:
    - adelino-standalone running (WebSocket server on port 8765)
    - A calibration TOML file (from adelino-standalone calibrate)
    - A trained YOLO segmentor (vimu_seg.pt from Phase 2)

Usage:
    # First camera angle (configure defaults in .env)
    python collect_pose.py sweep \
        --variant sparse_large_v1 \
        --calibration /path/to/calibration.toml \
        --num-poses 500

    # Move tripod to a different angle, then append
    python collect_pose.py sweep \
        --variant sparse_large_v1 \
        --calibration /path/to/calibration.toml \
        --num-poses 500 --append

    # Tilted base collection (manual)
    python collect_pose.py tilted \
        --variant sparse_large_v1 \
        --calibration /path/to/calibration.toml

Output:
    pose_data/
        masked/*.jpg    # segmented robot images (black background)
        labels.csv      # frame, joint_1, ..., joint_N, base_roll, base_pitch
"""

import argparse
import csv
import json
import os
import time
from pathlib import Path

os.environ["OPENCV_LOG_LEVEL"] = "FATAL"

import cv2  # noqa: E402
import numpy as np  # noqa: E402

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from model_paths import get_model_path, list_variants, get_models_dir  # noqa: E402


def load_dotenv():
    """Load .env file from the script's directory if it exists."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def resolve_seg_model(args) -> Path:
    """Resolve segmentor model path from --variant or --seg-model."""
    if getattr(args, "seg_model", None):
        p = Path(args.seg_model)
        if not p.exists():
            print(f"ERROR: Segmentor file not found: {p}")
            raise SystemExit(1)
        return p

    variant = getattr(args, "variant", None)
    models_dir = getattr(args, "models_dir", None)

    if variant:
        p = get_model_path("segmentation", variant, "vimu_seg.pt", models_dir)
        if not p.exists():
            print(f"ERROR: No vimu_seg.pt found for variant '{variant}'")
            print(f"  Expected: {p}")
            variants = list_variants("segmentation", models_dir)
            if variants:
                print(f"  Available variants: {', '.join(variants)}")
            raise SystemExit(1)
        return p

    variants = list_variants("segmentation", models_dir)
    models_dir_path = get_models_dir(models_dir)
    if variants:
        print(f"Available segmentation variants ({models_dir_path.resolve()}):")
        for v in variants:
            print(f"  --variant {v}")
    else:
        print(f"No segmentation variants found in {models_dir_path.resolve()}")
    print("\nSpecify --variant <name> or --seg-model <path>")
    raise SystemExit(1)


# ─── Calibration loader ─────────────────────────────────────────────────────

def load_calibration(path: str) -> list[dict]:
    """Load joint calibration from a TOML file."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    joints = data.get("joints", [])
    if not joints:
        raise ValueError(f"No joints found in calibration file: {path}")
    return [
        {"name": j["name"], "min_rad": j["min_rad"], "max_rad": j["max_rad"]}
        for j in joints
    ]


# ─── YOLO Segmentor ─────────────────────────────────────────────────────────

class YoloSegmentor:
    """Runs a YOLO segmentation model to produce binary masks."""

    def __init__(self, model_path: str, conf: float = 0.5):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = conf

    def segment(self, frame: np.ndarray) -> np.ndarray:
        """Run segmentation on a BGR frame. Returns a binary mask (uint8, 0 or 255)."""
        results = self.model(frame, conf=self.conf, verbose=False)
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if results and results[0].masks is not None:
            for m in results[0].masks.data:
                m_np = m.cpu().numpy().astype(np.uint8)
                m_resized = cv2.resize(m_np, (w, h), interpolation=cv2.INTER_NEAREST)
                mask[m_resized > 0] = 255

        return mask

    def apply_mask(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply binary mask to frame (black background where mask is 0)."""
        masked = frame.copy()
        masked[mask == 0] = 0
        return masked


# ─── WebSocket Controller ───────────────────────────────────────────────────

class WebSocketController:
    """Connects to nuttymoves' WebSocket-based robot controller."""

    def __init__(self, url: str, num_joints: int):
        import websocket as ws_lib
        self.url = url
        self.num_joints = num_joints
        self.ws = ws_lib.WebSocket()
        self.ws.settimeout(5.0)
        print(f"Connecting to controller at {url}...")
        self.ws.connect(url)
        print(f"Connected to WebSocket controller at {url}")

    def set_angles(self, angles_rad: list) -> bool:
        msg = json.dumps({
            "type": "command",
            "positions": [float(a) for a in angles_rad],
        })
        try:
            self.ws.send(msg)
            try:
                self.ws.settimeout(0.5)
                self.ws.recv()
                self.ws.settimeout(5.0)
            except Exception:
                pass
            return True
        except Exception as e:
            print(f"  WebSocket send failed: {e}")
            return False

    def close(self):
        try:
            self.set_angles([0.0] * self.num_joints)
            time.sleep(0.1)
        except Exception:
            pass
        try:
            self.ws.close()
        except Exception:
            pass
        print("WebSocket controller disconnected")


# ─── Pose generation ─────────────────────────────────────────────────────────

def generate_poses(
    joint_ranges: list[dict],
    num_poses: int,
    walk_delta: float = 0.2,
    reset_every: int = 25,
    seed: int = 42,
) -> list:
    """Generate a sequence of poses via random walk with occasional full-random resets.

    Each successive pose changes by at most ``walk_delta`` per joint (clipped to the
    joint's calibrated range). Every ``reset_every`` poses the walk jumps to a fresh
    fully-random pose to ensure coverage across the joint space.
    """
    rng = np.random.default_rng(seed)
    poses = []
    current = [0.0] * len(joint_ranges)
    for i in range(num_poses):
        if i % reset_every == 0:
            pose = [rng.uniform(j["min_rad"], j["max_rad"]) for j in joint_ranges]
        else:
            pose = []
            for c, j in zip(current, joint_ranges):
                step = rng.uniform(-walk_delta, walk_delta)
                pose.append(float(np.clip(c + step, j["min_rad"], j["max_rad"])))
        poses.append(pose)
        current = pose
    return poses


def interpolate_to(
    controller,
    current: list[float],
    target: list[float],
    max_delta: float,
    rate_hz: float,
) -> list[float]:
    """Send interpolated commands from current to target, limited to max_delta per step.

    Returns the final pose (== target).
    """
    dt = 1.0 / rate_hz
    pos = list(current)
    while True:
        remaining = [t - p for t, p in zip(target, pos)]
        max_remaining = max(abs(r) for r in remaining)
        if max_remaining <= max_delta:
            controller.set_angles(target)
            time.sleep(dt)
            return list(target)
        # Take a step of at most max_delta on the largest-moving joint
        scale = max_delta / max_remaining
        pos = [p + r * scale for p, r in zip(pos, remaining)]
        controller.set_angles(pos)
        time.sleep(dt)


# ─── Sweep collection ────────────────────────────────────────────────────────

def resolve_pose_data_dir(args) -> str:
    """Resolve the effective pose data directory: <output-dir>/<variant>."""
    base = args.output_dir
    variant = getattr(args, "variant", None)
    if variant:
        return os.path.join(base, variant)
    # Fall back to the raw output-dir when a direct --seg-model path is used
    return base


def open_camera(args):
    """Open the camera with requested resolution/fps (DirectShow + MJPG)."""
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    rw, rh = (int(x) for x in args.resolution.split("x"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, rw)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rh)
    if getattr(args, "fps", 0) and args.fps > 0:
        cap.set(cv2.CAP_PROP_FPS, args.fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Camera: {actual_w}x{actual_h} @ {actual_fps:.0f} fps")
    return cap


def collect_sweep(args, joint_ranges: list[dict]):
    num_joints = len(joint_ranges)

    pose_dir = resolve_pose_data_dir(args)
    masked_dir = os.path.join(pose_dir, "masked")
    os.makedirs(masked_dir, exist_ok=True)
    labels_path = os.path.join(pose_dir, "labels.csv")
    print(f"Pose data dir: {os.path.abspath(pose_dir)}")

    start_idx = 0
    if args.append and os.path.exists(masked_dir):
        existing = [f for f in os.listdir(masked_dir) if f.endswith(".jpg")]
        if existing:
            start_idx = max(int(os.path.splitext(f)[0]) for f in existing) + 1

    cap = open_camera(args)

    # Load segmentor
    seg_path = resolve_seg_model(args)
    print(f"Loading segmentor from {seg_path}...")
    seg = YoloSegmentor(str(seg_path))

    # Open controller
    controller = WebSocketController(args.ws, num_joints)

    # Generate poses (offset seed by start_idx for variety on append)
    poses = generate_poses(
        joint_ranges,
        args.num_poses,
        walk_delta=args.walk_delta,
        reset_every=args.reset_every,
        seed=args.seed + start_idx,
    )

    print(f"Collecting {len(poses)} poses for {num_joints} joints")
    print(f"  motion: max-delta {args.max_delta} rad/step @ {args.rate}Hz, settle {args.settle}s")
    print(f"  walk: delta {args.walk_delta} rad/pose, reset every {args.reset_every} poses")
    print(f"Joint ranges:")
    for j in joint_ranges:
        print(f"  {j['name']}: [{j['min_rad']:.3f}, {j['max_rad']:.3f}] rad")
    print(f"Starting at index {start_idx}")

    # Move to neutral (all zeros) before starting, using interpolation from unknown position
    neutral = [0.0] * num_joints
    print("Moving to neutral pose before starting...")
    # We don't know the starting position, so send neutral directly and wait.
    controller.set_angles(neutral)
    time.sleep(max(args.settle, 1.0))
    current_pose = list(neutral)

    # CSV setup
    joint_cols = [f"joint_{i+1}" for i in range(num_joints)]
    header = ["frame"] + joint_cols + ["base_roll", "base_pitch"]

    file_mode = "a" if args.append and os.path.exists(labels_path) else "w"
    write_header = file_mode == "w"

    collected = 0
    try:
        with open(labels_path, file_mode, newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)

            for i, pose in enumerate(poses):
                idx = start_idx + i

                try:
                    current_pose = interpolate_to(
                        controller, current_pose, pose, args.max_delta, args.rate
                    )
                except Exception as e:
                    print(f"  Interpolation failed at pose {idx}: {e}")
                    continue

                time.sleep(args.settle)

                # Flush camera buffer and grab fresh frame
                for _ in range(3):
                    cap.grab()
                ret, frame = cap.read()
                if not ret:
                    print("Camera read failed!")
                    break

                # Segment and mask
                mask = seg.segment(frame)
                masked = seg.apply_mask(frame, mask)

                # Save masked frame
                fname = f"{idx:06d}.jpg"
                cv2.imwrite(os.path.join(masked_dir, fname), masked)

                # Write label
                row = [fname] + [f"{a:.6f}" for a in pose] + ["0.0", "0.0"]
                writer.writerow(row)
                f.flush()
                collected += 1

                if i % 50 == 0:
                    print(f"  {i}/{len(poses)} ({100*i/len(poses):.0f}%)")

                # Live preview: original + mask overlay
                overlay = frame.copy()
                overlay[mask > 0] = cv2.addWeighted(
                    frame, 0.5, np.full_like(frame, (0, 180, 0)), 0.5, 0
                )[mask > 0]
                angles_text = " ".join(f"{a:+.2f}" for a in pose)
                cv2.putText(overlay, f"Pose {idx} | {angles_text}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("VIMU v2 Collection", overlay)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Interrupted by user")
                    break

    finally:
        # Return to neutral before disconnecting
        print("Returning to neutral pose...")
        try:
            interpolate_to(
                controller, current_pose, [0.0] * num_joints, args.max_delta, args.rate
            )
            time.sleep(max(args.settle, 1.0))
        except Exception:
            pass
        controller.close()
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nDone! Collected {collected} masked frames -> {pose_dir}")
    print(f"Next step: python train.py --data {pose_dir} --num-joints {num_joints}")


# ─── Tilted base collection ──────────────────────────────────────────────────

def collect_tilted(args, joint_ranges: list[dict]):
    num_joints = len(joint_ranges)
    pose_dir = resolve_pose_data_dir(args)
    masked_dir = os.path.join(pose_dir, "masked")
    os.makedirs(masked_dir, exist_ok=True)
    print(f"Pose data dir: {os.path.abspath(pose_dir)}")

    cap = open_camera(args)

    seg_path = resolve_seg_model(args)
    print(f"Loading segmentor from {seg_path}...")
    seg = YoloSegmentor(str(seg_path))

    existing = [f for f in os.listdir(masked_dir) if f.endswith(".jpg")] if os.path.exists(masked_dir) else []
    start_idx = max((int(os.path.splitext(f)[0]) for f in existing), default=-1) + 1

    labels_path = os.path.join(pose_dir, "labels.csv")
    file_exists = os.path.exists(labels_path)

    print("\n=== Tilted Base Collection ===")
    print("Manually tilt the robot to a known angle.")
    print("SPACE = capture (will ask for angles) | Q = quit")

    try:
        with open(labels_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                joint_cols = [f"joint_{i+1}" for i in range(num_joints)]
                writer.writerow(["frame"] + joint_cols + ["base_roll", "base_pitch"])

            idx = start_idx
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                mask = seg.segment(frame)
                overlay = frame.copy()
                overlay[mask > 0] = cv2.addWeighted(
                    frame, 0.5, np.full_like(frame, (0, 180, 0)), 0.5, 0
                )[mask > 0]
                cv2.putText(overlay, f"Frames: {idx - start_idx} | SPACE=capture Q=quit",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("VIMU v2 Tilted Collection", overlay)

                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" "):
                    masked = seg.apply_mask(frame, mask)
                    fname = f"{idx:06d}.jpg"
                    cv2.imwrite(os.path.join(masked_dir, fname), masked)

                    print(f"\nFrame {idx} captured!")
                    joints = []
                    for i in range(num_joints):
                        val = float(input(f"  Joint {i+1} angle (rad): "))
                        joints.append(val)
                    roll = float(input("  Base roll (radians): "))
                    pitch = float(input("  Base pitch (radians): "))

                    row = [fname] + [f"{a:.6f}" for a in joints] + [f"{roll:.6f}", f"{pitch:.6f}"]
                    writer.writerow(row)
                    f.flush()
                    idx += 1
                    print(f"  Saved! ({idx - start_idx} tilted frames)")

    finally:
        cap.release()
        cv2.destroyAllWindows()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def add_common_args(p):
    """Arguments shared between sweep and tilted modes."""
    p.add_argument("--variant", help="Segmentation model variant name")
    p.add_argument("--seg-model", help="Direct path to YOLO checkpoint (overrides --variant)")
    p.add_argument("--models-dir", default=None, help="Override models directory")
    p.add_argument("--calibration", required=True,
                   help="Path to calibration TOML (from adelino-standalone calibrate)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--resolution", default="640x480", help="Camera resolution WxH (default: 640x480)")
    p.add_argument("--fps", type=float, default=0, help="Camera capture FPS (0 = max supported)")
    p.add_argument("--output-dir", default=os.environ.get("POSE_DATA_DIR", "./pose_data"))


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="VIMU v2: Pose data collection with live segmentation")
    sub = parser.add_subparsers(dest="mode", required=True)

    # Sweep mode
    sweep = sub.add_parser("sweep", help="Automated random pose sweep")
    add_common_args(sweep)
    sweep.add_argument("--ws", default=os.environ.get("CONTROLLER_WS", "ws://localhost:8765"),
                       help="WebSocket URL of robot controller (default: ws://localhost:8765)")
    sweep.add_argument("--num-poses", type=int, default=500)
    sweep.add_argument("--settle", type=float, default=0.2,
                       help="Seconds to wait after arriving at a pose before capture (default: 0.2)")
    sweep.add_argument("--max-delta", type=float, default=0.05,
                       help="Max per-joint change per interpolation sub-step in rad (default: 0.05)")
    sweep.add_argument("--rate", type=float, default=30.0,
                       help="Interpolation command rate in Hz (default: 30)")
    sweep.add_argument("--walk-delta", type=float, default=0.2,
                       help="Max per-joint change between captured poses in rad (default: 0.2)")
    sweep.add_argument("--reset-every", type=int, default=25,
                       help="Jump to a fresh random pose every N poses (default: 25)")
    sweep.add_argument("--seed", type=int, default=42)
    sweep.add_argument("--append", action="store_true",
                       help="Append to existing dataset (for multi-angle collection)")

    # Tilted mode
    tilted = sub.add_parser("tilted", help="Manual tilted base capture")
    add_common_args(tilted)

    args = parser.parse_args()
    joint_ranges = load_calibration(args.calibration)
    print(f"Loaded calibration: {len(joint_ranges)} joints from {args.calibration}")

    if args.mode == "sweep":
        collect_sweep(args, joint_ranges)
    elif args.mode == "tilted":
        collect_tilted(args, joint_ranges)


if __name__ == "__main__":
    main()
