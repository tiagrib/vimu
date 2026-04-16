"""
VIMU v2: Live segmentor test.

Opens the camera and runs the trained YOLO segmentor in real time,
showing a green mask overlay on the detected robot. Use this to verify
the segmentor works across different angles, lighting, and backgrounds
before moving to Phase 3.

Usage:
    python test_segmentor.py --model vimu_seg.pt
    python test_segmentor.py --model vimu_seg.pt --camera 1

Controls:
    q = quit
"""

import argparse
import time

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="VIMU v2: Live segmentor test")
    parser.add_argument("--model", default="vimu_seg.pt", help="YOLO segmentor checkpoint")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--conf", type=float, default=0.5, help="Detection confidence threshold")
    args = parser.parse_args()

    # Detect available cameras
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
    print(f"\nTo use a different camera: python test_segmentor.py --camera <index>\n")

    from ultralytics import YOLO
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera}")
        return

    print(f"Running segmentor on camera {args.camera}. Press 'q' to quit.")

    prev_time = time.time()
    fps = 0.0

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

        cv2.imshow("Segmentor Test", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
