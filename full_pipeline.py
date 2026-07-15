"""
Full Integrated Pipeline
========================
Combines all three systems:
  1. Detection   → detect vehicles + classify emergency type
  2. Speed       → estimate each vehicle's speed, flag slow-movers
  3. Signal      → detect siren audio + flashing lights

Run on video:
    python pipeline/full_pipeline.py --video ./test_video.mp4 \
        --classifier_weights ./results/detection/classifier_best.pth \
        --siren_weights      ./results/signal/siren_best.pth

Run on webcam:
    python pipeline/full_pipeline.py --video 0
"""

import argparse
import sys
import cv2
import time
import numpy as np
import torch
from pathlib import Path
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))

from detection.model            import VehicleDetectionPipeline
from speed_estimation.tracker   import CentroidTracker
from speed_estimation.speed     import CameraCalibration, SpeedEstimator
from emergency_signal.siren_model import FlashDetector


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--video',               type=str, default='0')
    p.add_argument('--classifier_weights',  type=str, default=None)
    p.add_argument('--siren_weights',       type=str, default=None)
    p.add_argument('--output_dir',          type=str, default='./results')
    p.add_argument('--save_output',         action='store_true')
    p.add_argument('--slow_threshold',      type=float, default=20.0)
    p.add_argument('--mount_height',        type=float, default=1.2)
    p.add_argument('--mount_pitch',         type=float, default=5.0)
    return p.parse_args()


COLORS = {
    'civilian':   (180, 180, 180),
    'police':     (255, 120,   0),
    'ambulance':  (0,   200, 255),
    'fire_truck': (30,   60, 255),
}


def draw_all(frame, det_results, speed_results, flash_results, fps, frame_num):
    """Compose full annotation overlay on frame."""
    out = frame.copy()

    # --- Per-vehicle annotations ---
    for det in det_results:
        x1, y1, x2, y2 = det['box']
        tid   = det.get('track_id')
        color = COLORS.get(det['em_class'], (180,180,180))
        thick = 3 if det['is_emergency'] else 1

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)

        label_parts = [det['em_class']]
        if tid and tid in speed_results:
            spd = speed_results[tid]['speed_kmh']
            slow = speed_results[tid]['is_slow']
            label_parts.append(f"{spd:.0f}km/h" + (" ⚠SLOW" if slow else ""))

        # Flash overlay
        for fr in flash_results:
            fx1,fy1,fx2,fy2 = fr['box']
            if abs(fx1-x1)<30 and fr['is_flashing']:
                label_parts.append("FLASH")

        label = "  ".join(label_parts)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1-22), (x1+tw+6, y1), color, -1)
        cv2.putText(out, label, (x1+3, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    # --- Alert banner ---
    alerts = []
    em_dets = [d for d in det_results if d['is_emergency']]
    if em_dets:
        names = list({d['em_class'] for d in em_dets})
        alerts.append(f"EMERGENCY: {', '.join(names).upper()}")

    slow_ids = [v for v in speed_results.values() if v['is_slow']]
    if slow_ids:
        alerts.append(f"SLOW VEHICLE x{len(slow_ids)}")

    any_flash = any(r['is_flashing'] for r in flash_results)
    if any_flash:
        alerts.append("FLASHING LIGHTS")

    if alerts:
        banner = "  |  ".join(alerts)
        h = frame.shape[0]
        cv2.rectangle(out, (0, h-44), (frame.shape[1], h), (0,0,200), -1)
        cv2.putText(out, f"⚠  {banner}", (10, h-14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # --- Stats ---
    cv2.putText(out, f"FPS: {fps:.1f}  Frame: {frame_num}", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
    cv2.putText(out, f"Vehicles: {len(det_results)}", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)

    return out


def run():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Info] Device: {device}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Build sub-systems ──────────────────────────
    print("[Init] Loading detection pipeline...")
    det_pipeline = VehicleDetectionPipeline(
        yolo_weights='yolov8n.pt',
        classifier_weights=args.classifier_weights,
    )

    print("[Init] Setting up speed estimator...")
    cap_probe = cv2.VideoCapture(
        int(args.video) if args.video.isdigit() else args.video)
    fps = cap_probe.get(cv2.CAP_PROP_FPS) or 30
    W   = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_probe.release()

    calib     = CameraCalibration(image_width_px=W, image_height_px=H,
                                  mount_height_m=args.mount_height,
                                  mount_pitch_deg=args.mount_pitch)
    tracker   = CentroidTracker()
    speed_est = SpeedEstimator(calib, fps, slow_threshold_kmh=args.slow_threshold)

    print("[Init] Setting up flash detector...")
    flash_det = FlashDetector(fps=fps)

    # ── Open video ─────────────────────────────────
    cap = cv2.VideoCapture(int(args.video) if args.video.isdigit() else args.video)
    if not cap.isOpened():
        print(f"[Error] Cannot open: {args.video}"); return

    writer = None
    if args.save_output:
        out_path = str(Path(args.output_dir) / 'full_pipeline_output.mp4')
        writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps, (W, H))
        print(f"[Info] Saving output to: {out_path}")

    print("[Info] Running full pipeline. Press Q to quit.\n")
    frame_num  = 0
    fps_history = deque(maxlen=30)
    VEHICLE_IDS = {2, 3, 5, 7}

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_num += 1
        t0 = time.time()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Detection
        det_results = det_pipeline.predict(frame)

        # 2. Speed estimation via tracker
        tracker_dets = [{'box': d['box'], 'class_id': d['yolo_class']}
                        for d in det_results]
        active_tracks = tracker.update(tracker_dets)
        speed_results = speed_est.update(active_tracks)

        # Map speed back to detection results by nearest centroid
        for det in det_results:
            bx = (det['box'][0]+det['box'][2])//2
            by = (det['box'][1]+det['box'][3])//2
            best_tid, best_dist = None, 1e9
            for t in active_tracks:
                d = np.linalg.norm(t.centroid - np.array([bx, by]))
                if d < best_dist:
                    best_dist, best_tid = d, t.id
            det['track_id'] = best_tid

        # 3. Flash detection
        boxes = [d['box'] for d in det_results]
        flash_results = flash_det.update(gray, boxes)

        # Compute FPS
        fps_history.append(time.time() - t0)
        current_fps = 1.0 / (sum(fps_history) / len(fps_history) + 1e-6)

        # Draw
        annotated = draw_all(frame, det_results, speed_results,
                             flash_results, current_fps, frame_num)

        if writer: writer.write(annotated)
        cv2.imshow('Autonomous CV Pipeline', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

        # Console log every 30 frames
        if frame_num % 30 == 0:
            em = [d for d in det_results if d['is_emergency']]
            slow = [v for v in speed_results.values() if v['is_slow']]
            flash = [r for r in flash_results if r['is_flashing']]
            print(f"Frame {frame_num:5d} | "
                  f"Vehicles:{len(det_results)} | "
                  f"Emergency:{len(em)} | "
                  f"Slow:{len(slow)} | "
                  f"Flash:{len(flash)} | "
                  f"FPS:{current_fps:.1f}")

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    print(f"\n[Done] Processed {frame_num} frames")


if __name__ == '__main__':
    run()
