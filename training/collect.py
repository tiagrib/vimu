"""
VIMU Phase 2: Data Collection

Commands the robot through random poses via the Arduino binary protocol,
captures synchronized camera frames, and saves labeled training data.

Usage:
    python collect.py \
        --serial /dev/ttyUSB0 \
        --camera 0 \
        --num-joints 6 \
        --num-poses 1500 \
        --output-dir ./data

Output:
    data/
        frames/000000.jpg, 000001.jpg, ...
        labels.csv  →  frame, joint_1, ..., joint_N, base_roll, base_pitch
"""

import argparse
import csv
import os
import struct
import time

import cv2
import numpy as np
import serial as pyserial


# ─── Arduino binary protocol ─────────────────────────────────────────────────

FRAME_START_TX = 0xAA
FRAME_START_RX = 0xBB
CMD_SET_POSITIONS = 0x01
CMD_DETACH_ALL = 0x04

SERVO_MIN_US = 500
SERVO_MAX_US = 2500


class ServoController:
    """Communicates with the Arduino data-collection firmware."""

    def __init__(self, port: str, baud: int = 500000, num_servos: int = 6):
        self.ser = pyserial.Serial(port, baud, timeout=0.1)
        self.num_servos = num_servos
        time.sleep(2.5)  # Wait for Arduino reset
        self.ser.reset_input_buffer()
        print(f"Arduino connected: {port} @ {baud}")

    def _send(self, cmd: int, payload: bytes = b""):
        frame = bytes([FRAME_START_TX, cmd, len(payload)]) + payload
        checksum = 0
        for b in frame[1:]:
            checksum ^= b
        frame += bytes([checksum])
        self.ser.write(frame)
        self.ser.flush()

    def _read_response(self, timeout: float = 0.5) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.ser.in_waiting >= 4:
                start = self.ser.read(1)
                if start[0] == FRAME_START_RX:
                    status = self.ser.read(1)[0]
                    length = self.ser.read(1)[0]
                    if length > 0:
                        self.ser.read(length)
                    self.ser.read(1)  # checksum
                    return status == 0x00
            time.sleep(0.001)
        return False

    def set_angles(self, angles_rad: list) -> bool:
        """Set servo positions. Angles in radians [-π/2, +π/2]."""
        payload = b""
        for angle in angles_rad:
            norm = (angle + np.pi / 2) / np.pi
            norm = float(np.clip(norm, 0, 1))
            us = int(SERVO_MIN_US + norm * (SERVO_MAX_US - SERVO_MIN_US))
            payload += struct.pack("<H", us)
        self._send(CMD_SET_POSITIONS, payload)
        return self._read_response()

    def detach(self):
        self._send(CMD_DETACH_ALL)
        self._read_response()

    def close(self):
        self.detach()
        self.ser.close()


# ─── Pose generation ─────────────────────────────────────────────────────────

# CUSTOMIZE per joint: (min_rad, max_rad)
# These should match your robot's actual mechanical limits.
DEFAULT_JOINT_RANGES = [
    (-1.2, 1.2),   # Joint 1
    (-0.8, 1.2),   # Joint 2
    (-1.0, 1.0),   # Joint 3
    (-1.2, 1.2),   # Joint 4
    (-1.5, 1.5),   # Joint 5
    (-0.8, 0.8),   # Joint 6
]


def generate_poses(num_joints: int, num_poses: int, seed: int = 42) -> list:
    """Generate random joint configurations within limits."""
    rng = np.random.default_rng(seed)
    ranges = DEFAULT_JOINT_RANGES[:num_joints]
    poses = []
    for _ in range(num_poses):
        pose = []
        for lo, hi in ranges:
            pose.append(rng.uniform(lo, hi))
        poses.append(pose)
    return poses


# ─── Main collection loop ────────────────────────────────────────────────────

def collect(args):
    frames_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Open Arduino
    servo = ServoController(args.serial, args.baud, args.num_joints)

    # Generate poses
    poses = generate_poses(args.num_joints, args.num_poses, seed=args.seed)
    print(f"Collecting {len(poses)} poses, settle time {args.settle}s")
    print(f"Estimated time: {len(poses) * args.settle / 60:.1f} minutes")

    # CSV setup
    labels_path = os.path.join(args.output_dir, "labels.csv")
    joint_cols = [f"joint_{i+1}" for i in range(args.num_joints)]
    # base_roll and base_pitch are 0 when on the table
    header = ["frame"] + joint_cols + ["base_roll", "base_pitch"]

    collected = 0
    try:
        with open(labels_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for idx, pose in enumerate(poses):
                # Command the servos
                ok = servo.set_angles(pose)
                if not ok:
                    print(f"  WARNING: Servo command failed at pose {idx}, retrying...")
                    time.sleep(0.1)
                    ok = servo.set_angles(pose)
                    if not ok:
                        print(f"  Skipping pose {idx}")
                        continue

                # Wait for servos to settle
                time.sleep(args.settle)

                # Flush camera buffer and grab fresh frame
                for _ in range(3):
                    cap.grab()
                ret, frame = cap.read()
                if not ret:
                    print("Camera read failed!")
                    break

                # Save frame
                fname = f"{idx:06d}.jpg"
                cv2.imwrite(os.path.join(frames_dir, fname), frame)

                # Write label row: joints + base state (0 on table)
                row = [fname] + [f"{a:.6f}" for a in pose] + ["0.0", "0.0"]
                writer.writerow(row)
                f.flush()
                collected += 1

                # Progress
                if idx % 50 == 0:
                    print(f"  {idx}/{len(poses)} ({100*idx/len(poses):.0f}%)")

                # Preview
                cv2.imshow("VIMU Collection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Interrupted by user")
                    break

    finally:
        servo.close()
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nDone! Collected {collected} labeled frames → {args.output_dir}")
    print(f"Next step: python train.py --data-dir {args.output_dir} --num-joints {args.num_joints}")


# ─── Tilted base collection ──────────────────────────────────────────────────

def collect_tilted(args):
    """
    Interactive mode for collecting base orientation labels.
    Manually tilt the robot to known angles, press SPACE to capture.
    """
    frames_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Find next frame index
    existing = [f for f in os.listdir(frames_dir) if f.endswith(".jpg")]
    start_idx = len(existing)

    labels_path = os.path.join(args.output_dir, "labels.csv")
    file_exists = os.path.exists(labels_path)

    print("\n=== Tilted Base Collection ===")
    print("Manually tilt the robot to a known angle.")
    print("SPACE = capture (will ask for angles) | Q = quit")

    try:
        with open(labels_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                joint_cols = [f"joint_{i+1}" for i in range(args.num_joints)]
                writer.writerow(["frame"] + joint_cols + ["base_roll", "base_pitch"])

            idx = start_idx
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                display = frame.copy()
                cv2.putText(display, f"Frames: {idx - start_idx} | SPACE=capture Q=quit",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("VIMU Tilted Collection", display)

                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord(" "):
                    fname = f"{idx:06d}.jpg"
                    cv2.imwrite(os.path.join(frames_dir, fname), frame)

                    print(f"\nFrame {idx} captured!")
                    joints = []
                    for i in range(args.num_joints):
                        val = float(input(f"  Joint {i+1} angle (rad, or servo command value): "))
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


def main():
    parser = argparse.ArgumentParser(description="VIMU data collection")
    sub = parser.add_subparsers(dest="mode", required=True)

    # Automated sweep mode
    auto = sub.add_parser("sweep", help="Automated random pose sweep")
    auto.add_argument("--serial", required=True, help="Arduino serial port")
    auto.add_argument("--baud", type=int, default=500000)
    auto.add_argument("--camera", type=int, default=0)
    auto.add_argument("--num-joints", type=int, default=6)
    auto.add_argument("--num-poses", type=int, default=1500)
    auto.add_argument("--settle", type=float, default=0.6, help="Seconds after each command")
    auto.add_argument("--output-dir", default="./data")
    auto.add_argument("--seed", type=int, default=42)

    # Manual tilted base mode
    tilt = sub.add_parser("tilted", help="Manual tilted base capture")
    tilt.add_argument("--camera", type=int, default=0)
    tilt.add_argument("--num-joints", type=int, default=6)
    tilt.add_argument("--output-dir", default="./data")

    args = parser.parse_args()
    if args.mode == "sweep":
        collect(args)
    elif args.mode == "tilted":
        collect_tilted(args)


if __name__ == "__main__":
    main()
