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
    # First camera angle
    python collect_pose.py \
        --calibration ../../projects/adelino/target/release/calibration.toml \
        --seg-model vimu_seg.pt \
        --ws ws://localhost:8765 \
        --camera 0 \
        --num-poses 500 \
        --output-dir ./pose_data

    # Move tripod to a different angle, then append
    python collect_pose.py \
        --calibration ../../projects/adelino/target/release/calibration.toml \
        --seg-model vimu_seg.pt \
        --ws ws://localhost:8765 \
        --camera 0 \
        --num-poses 500 \
        --output-dir ./pose_data \
        --append

    # Tilted base collection (manual)
    python collect_pose.py tilted \
        --calibration ../../projects/adelino/target/release/calibration.toml \
        --seg-model vimu_seg.pt \
        --camera 0 \
        --output-dir ./pose_data

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

import cv2
import numpy as np

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


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

def generate_poses(joint_ranges: list[dict], num_poses: int, seed: int = 42) -> list:
    rng = np.random.default_rng(seed)
    poses = []
    for _ in range(num_poses):
        pose = [rng.uniform(j["min_rad"], j["max_rad"]) for j in joint_ranges]
        poses.append(pose)
    return poses


# ─── Sweep collection ────────────────────────────────────────────────────────

def collect_sweep(args, joint_ranges: list[dict]):
    num_joints = len(joint_ranges)

    # Find start index for appending
    masked_dir = os.path.join(args.output_dir, "masked")
    os.makedirs(masked_dir, exist_ok=True)
    labels_path = os.path.join(args.output_dir, "labels.csv")

    start_idx = 0
    if args.append and os.path.exists(masked_dir):
        existing = [f for f in os.listdir(masked_dir) if f.endswith(".jpg")]
        if existing:
            start_idx = max(int(os.path.splitext(f)[0]) for f in existing) + 1

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Load segmentor
    print(f"Loading segmentor from {args.seg_model}...")
    seg = YoloSegmentor(args.seg_model)

    # Open controller
    controller = WebSocketController(args.ws, num_joints)

    # Generate poses (offset seed by start_idx for variety on append)
    poses = generate_poses(joint_ranges, args.num_poses, seed=args.seed + start_idx)

    print(f"Collecting {len(poses)} poses for {num_joints} joints, settle time {args.settle}s")
    print(f"Joint ranges:")
    for j in joint_ranges:
        print(f"  {j['name']}: [{j['min_rad']:.3f}, {j['max_rad']:.3f}] rad")
    print(f"Starting at index {start_idx}")
    print(f"Estimated time: {len(poses) * args.settle / 60:.1f} minutes")

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

                ok = controller.set_angles(pose)
                if not ok:
                    time.sleep(0.1)
                    ok = controller.set_angles(pose)
                    if not ok:
                        print(f"  Skipping pose {idx}")
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
        controller.close()
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nDone! Collected {collected} masked frames -> {args.output_dir}")
    print(f"Next step: python train.py --data {args.output_dir} --num-joints {num_joints}")


# ─── Tilted base collection ──────────────────────────────────────────────────

def collect_tilted(args, joint_ranges: list[dict]):
    num_joints = len(joint_ranges)
    masked_dir = os.path.join(args.output_dir, "masked")
    os.makedirs(masked_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"Loading segmentor from {args.seg_model}...")
    seg = YoloSegmentor(args.seg_model)

    existing = [f for f in os.listdir(masked_dir) if f.endswith(".jpg")] if os.path.exists(masked_dir) else []
    start_idx = max((int(os.path.splitext(f)[0]) for f in existing), default=-1) + 1

    labels_path = os.path.join(args.output_dir, "labels.csv")
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

def main():
    parser = argparse.ArgumentParser(description="VIMU v2: Pose data collection with live segmentation")
    sub = parser.add_subparsers(dest="mode", required=True)

    # Sweep mode
    sweep = sub.add_parser("sweep", help="Automated random pose sweep")
    sweep.add_argument("--calibration", required=True,
                       help="Path to calibration TOML (from adelino-standalone calibrate)")
    sweep.add_argument("--seg-model", required=True, help="Path to trained YOLO segmentor (vimu_seg.pt)")
    sweep.add_argument("--ws", default="ws://localhost:8765",
                       help="WebSocket URL of robot controller (default: ws://localhost:8765)")
    sweep.add_argument("--camera", type=int, default=0)
    sweep.add_argument("--num-poses", type=int, default=500)
    sweep.add_argument("--settle", type=float, default=0.6)
    sweep.add_argument("--output-dir", default="./pose_data")
    sweep.add_argument("--seed", type=int, default=42)
    sweep.add_argument("--append", action="store_true",
                       help="Append to existing dataset (for multi-angle collection)")

    # Tilted mode
    tilted = sub.add_parser("tilted", help="Manual tilted base capture")
    tilted.add_argument("--calibration", required=True,
                        help="Path to calibration TOML (from adelino-standalone calibrate)")
    tilted.add_argument("--seg-model", required=True, help="Path to trained YOLO segmentor (vimu_seg.pt)")
    tilted.add_argument("--camera", type=int, default=0)
    tilted.add_argument("--output-dir", default="./pose_data")

    args = parser.parse_args()
    joint_ranges = load_calibration(args.calibration)
    print(f"Loaded calibration: {len(joint_ranges)} joints from {args.calibration}")

    if args.mode == "sweep":
        collect_sweep(args, joint_ranges)
    elif args.mode == "tilted":
        collect_tilted(args, joint_ranges)


if __name__ == "__main__":
    main()
