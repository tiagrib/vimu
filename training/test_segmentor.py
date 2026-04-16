"""
VIMU v2: Live segmentor test.

Opens the camera and runs the trained YOLO segmentor in real time,
showing a green mask overlay on the detected robot. Use this to verify
the segmentor works across different angles, lighting, and backgrounds
before moving to Phase 3.

Press 'r' to start/stop recording negative training videos (e.g. remove
the robot from view, hit 'r', move camera around false positives, hit 'r'
again). Videos are saved directly to the configured VIDEO_DIR.

Usage:
    python test_segmentor.py --model vimu_seg.pt
    python test_segmentor.py --model vimu_seg.pt --camera 1

Controls:
    r = start/stop recording
    q = quit
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


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


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="VIMU v2: Live segmentor test")
    parser.add_argument("--model", default="vimu_seg.pt", help="YOLO segmentor checkpoint")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    args = parser.parse_args()

    video_dir = Path(os.environ.get("VIDEO_DIR", "./videos"))

    # Detect available cameras (suppress OpenCV probe noise)
    os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
    available = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            available.append((i, w, h))
            cap.release()

    if not available:
        print("ERROR: No cameras found")
        return

    print(f"Found {len(available)} camera(s):")
    for idx, w, h in available:
        marker = " <-- using" if idx == args.camera else ""
        print(f"  Camera {idx}: {w}x{h}{marker}")
    if args.camera not in [i for i, _, _ in available]:
        print(f"\nERROR: Camera {args.camera} not available. Use --camera <index> to select one.")
        return
    print(f"\nTo use a different camera: python test_segmentor.py --camera <index>")

    from ultralytics import YOLO
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera}")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"\nControls: r = start/stop recording, q = quit")
    print(f"Recordings save to: {video_dir.resolve()}\n")

    cv2.namedWindow("Segmentor Test", cv2.WINDOW_NORMAL)

    prev_time = time.time()
    fps = 0.0
    recording = False
    writer = None
    record_path = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=args.conf, verbose=False)

        # Draw mask overlay
        display = frame.copy()
        for result in results:
            if result.masks is not None:
                for mask_data in result.masks.data:
                    mask = mask_data.cpu().numpy()
                    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    binary = (mask > 0.5).astype(np.uint8)

                    # Green overlay
                    colored = np.zeros_like(frame)
                    colored[:, :, 1] = 180
                    display[binary > 0] = cv2.addWeighted(
                        frame, 0.6, colored, 0.4, 0
                    )[binary > 0]

                    # Contour outline
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(display, contours, -1, (0, 255, 0), 2)

        # FPS
        now = time.time()
        dt = now - prev_time
        prev_time = now
        fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))
        cv2.putText(display, f"FPS: {fps:.0f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Recording indicator
        if recording:
            cv2.circle(display, (frame_w - 25, 25), 10, (0, 0, 255), -1)
            cv2.putText(display, "REC", (frame_w - 70, 33),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            writer.write(frame)  # write raw frame, not the overlay

        # Letterbox to fit window without distorting aspect ratio
        win_rect = cv2.getWindowImageRect("Segmentor Test")
        if win_rect[2] > 0 and win_rect[3] > 0:
            win_w, win_h = win_rect[2], win_rect[3]
            scale = min(win_w / frame_w, win_h / frame_h)
            new_w, new_h = int(frame_w * scale), int(frame_h * scale)
            resized = cv2.resize(display, (new_w, new_h))
            canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
            y_off = (win_h - new_h) // 2
            x_off = (win_w - new_w) // 2
            canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
            display = canvas

        cv2.imshow("Segmentor Test", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("r"):
            if not recording:
                # Start recording
                video_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                record_path = video_dir / f"neg_{stamp}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(record_path), fourcc, camera_fps, (frame_w, frame_h))
                recording = True
                print(f"  Recording started: {record_path}")
            else:
                # Stop recording
                writer.release()
                writer = None
                recording = False
                print(f"  Recording saved: {record_path}")

    if writer is not None:
        writer.release()
        if recording:
            print(f"  Recording saved: {record_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
