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
    python test_segmentor.py --variant large_600frames
    python test_segmentor.py --variant large_600frames --camera 1
    python test_segmentor.py --model /path/to/vimu_seg.pt   # direct path

Controls:
    r = start/stop recording
    q = quit
"""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

os.environ["OPENCV_LOG_LEVEL"] = "FATAL"

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from model_paths import get_model_path, list_variants, get_models_dir


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


def resolve_model(args) -> Path:
    """Resolve model path from --variant or --model."""
    if args.model:
        p = Path(args.model)
        if not p.exists():
            print(f"ERROR: Model file not found: {p}")
            raise SystemExit(1)
        return p

    if args.variant:
        p = get_model_path("segmentation", args.variant, "vimu_seg.pt", args.models_dir)
        if not p.exists():
            print(f"ERROR: No vimu_seg.pt found for variant '{args.variant}'")
            print(f"  Expected: {p}")
            variants = list_variants("segmentation", args.models_dir)
            if variants:
                print(f"  Available variants: {', '.join(variants)}")
            raise SystemExit(1)
        return p

    # No --variant or --model: list available and ask
    variants = list_variants("segmentation", args.models_dir)
    models_dir = get_models_dir(args.models_dir)
    if variants:
        print(f"Available segmentation variants ({models_dir.resolve()}):")
        for v in variants:
            print(f"  --variant {v}")
    else:
        print(f"No segmentation variants found in {models_dir.resolve()}")
    print("\nSpecify --variant <name> or --model <path>")
    raise SystemExit(1)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="VIMU v2: Live segmentor test")
    parser.add_argument("--variant", help="Segmentation model variant name")
    parser.add_argument("--model", help="Direct path to YOLO checkpoint (overrides --variant)")
    parser.add_argument("--models-dir", default=None, help="Override models directory")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--fps", type=float, default=0,
                        help="Camera capture FPS (0 = max supported, default: 0)")
    parser.add_argument("--resolution", default=None,
                        help="Camera resolution as WxH (e.g. 1280x720)")
    args = parser.parse_args()

    model_path = resolve_model(args)
    video_dir = Path(os.environ.get("VIDEO_DIR", "./videos"))

    # Detect available cameras using DirectShow (fast probe, no errors on Windows)
    available = []
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
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
    print(f"Loading model: {model_path}\n")
    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera}")
        return

    # Set MJPEG codec before resolution/fps — many webcams only support
    # higher framerates (e.g. 60fps) in MJPEG mode, not the default YUY2
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if args.resolution:
        rw, rh = (int(x) for x in args.resolution.split("x"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, rw)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rh)
    if args.fps > 0:
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    print(f"Camera: {frame_w}x{frame_h} @ {camera_fps:.0f} fps")
    if args.resolution or args.fps > 0:
        req_parts = []
        if args.resolution:
            req_parts.append(args.resolution)
        if args.fps > 0:
            req_parts.append(f"{args.fps:.0f}fps")
        print(f"  Requested: {', '.join(req_parts)} -> got {frame_w}x{frame_h} @ {camera_fps:.0f} fps")
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
