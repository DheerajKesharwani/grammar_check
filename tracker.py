"""
Tracker.py  —  Vehicle Speed Tracker (Side-View Camera)
=========================================================
Combines:
  • Kalman filter multi-object tracker  (built-in)
  • YOLO vehicle detection              (ultralytics)
  • CameraCalibration + SpeedEstimator  (Speed.py)

Fixes applied vs original Copilot version:
  FIX 1 — Speed cap 80 km/h          (kills 96–101 km/h tracking artefacts)
  FIX 2 — VEHICLE_WIDTH_M used       (width-based metres/pixel for side-view accuracy)
  FIX 3 — slow_threshold = 15 km/h   (correct for parking-lot scene, was 30)
  FIX 4 — Box colour uses 15 km/h    (green ≥ 15, yellow < 15)

Run:
    python3 Tracker.py
Requires:
    pip install ultralytics opencv-python numpy
    Speed.py in the same folder
"""

import numpy as np
import cv2

# ─────────────────────────────────────────────────────────────
#  Kalman Filter Box
# ─────────────────────────────────────────────────────────────

class KalmanBox:
    """
    Kalman filter for one bounding box.
    State  : [x, y, w, h, vx, vy, vw, vh]
    Observe: [x, y, w, h]
    """
    def __init__(self, box):
        x, y, w, h = self._to_xywh(box)
        self.state = np.array([x, y, w, h, 0., 0., 0., 0.], dtype=float)
        self.P = np.eye(8) * 10.
        self.F = np.eye(8)
        self.F[0,4] = self.F[1,5] = self.F[2,6] = self.F[3,7] = 1.0
        self.H = np.eye(4, 8)
        self.Q = np.eye(8) * 0.01
        self.R = np.eye(4) * 1.0

    @staticmethod
    def _to_xywh(box):
        x1, y1, x2, y2 = box
        return x1, y1, x2 - x1, y2 - y1

    def predict(self):
        self.state = self.F @ self.state
        self.P     = self.F @ self.P @ self.F.T + self.Q
        return self.state[:4]

    def update(self, box):
        z = np.array(self._to_xywh(box), dtype=float)
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P     = (np.eye(8) - K @ self.H) @ self.P

    def get_box(self):
        x, y, w, h = self.state[:4]
        return [x, y, x + w, y + h]

    def get_centroid(self):
        x, y, w, h = self.state[:4]
        return np.array([x + w / 2, y + h / 2])

    def get_pixel_width(self):
        return max(1.0, self.state[2])   # w component


# ─────────────────────────────────────────────────────────────
#  Track
# ─────────────────────────────────────────────────────────────

class Track:
    _id_counter = 0

    def __init__(self, box, class_id):
        Track._id_counter += 1
        self.id       = Track._id_counter
        self.class_id = class_id
        self.kf       = KalmanBox(box)
        self.hits     = 1
        self.misses   = 0
        self.history  = [self.kf.get_centroid().copy()]

    def predict(self):
        return self.kf.predict()

    def update(self, box, class_id):
        self.kf.update(box)
        self.class_id = class_id
        self.hits    += 1
        self.misses   = 0
        self.history.append(self.kf.get_centroid().copy())
        if len(self.history) > 30:
            self.history.pop(0)

    def mark_missed(self):
        self.misses += 1

    @property
    def is_confirmed(self):
        return self.hits >= 3

    @property
    def centroid(self):
        return self.kf.get_centroid()


# ─────────────────────────────────────────────────────────────
#  IoU helper
# ─────────────────────────────────────────────────────────────

def iou(a, b):
    xa = max(a[0], b[0]); ya = max(a[1], b[1])
    xb = min(a[2], b[2]); yb = min(a[3], b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-6)


# ─────────────────────────────────────────────────────────────
#  Centroid Tracker
# ─────────────────────────────────────────────────────────────

class CentroidTracker:
    def __init__(self, iou_threshold=0.35, max_misses=10):
        self.tracks        = []
        self.iou_threshold = iou_threshold
        self.max_misses    = max_misses

    def update(self, detections):
        for t in self.tracks:
            t.predict()

        if not detections:
            for t in self.tracks:
                t.mark_missed()
        else:
            det_boxes = [d['box']      for d in detections]
            det_cls   = [d['class_id'] for d in detections]
            matched_t = set(); matched_d = set()

            if self.tracks:
                mat = np.zeros((len(self.tracks), len(detections)))
                for i, t in enumerate(self.tracks):
                    for j, b in enumerate(det_boxes):
                        mat[i, j] = iou(t.kf.get_box(), b)
                while True:
                    idx = np.unravel_index(mat.argmax(), mat.shape)
                    if mat[idx] < self.iou_threshold:
                        break
                    i, j = idx
                    self.tracks[i].update(det_boxes[j], det_cls[j])
                    matched_t.add(i); matched_d.add(j)
                    mat[i, :] = -1; mat[:, j] = -1

            for i, t in enumerate(self.tracks):
                if i not in matched_t:
                    t.mark_missed()
            for j in range(len(detections)):
                if j not in matched_d:
                    self.tracks.append(Track(det_boxes[j], det_cls[j]))

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return [t for t in self.tracks if t.is_confirmed]


# ─────────────────────────────────────────────────────────────
#  FIX 2 — Width-based speed estimator for side-view camera
# ─────────────────────────────────────────────────────────────

class SideViewSpeedEstimator:
    """
    Better speed estimator for side-view (horizontal) cameras.

    Instead of projecting pixels onto a ground plane (which assumes
    a downward-pitched dashcam), this uses the known real-world width
    of each vehicle class to compute metres-per-pixel dynamically.

        metres_per_pixel = real_width_m / pixel_width_of_vehicle

    Then speed = lateral centroid displacement (px) × mpp / time_per_frame

    FIX 1 built in: hard cap at MAX_SPEED_KMH to eliminate tracking artefacts.
    FIX 3 built in: slow_threshold = 15 km/h (correct for parking lot).
    """

    # FIX 1
    MAX_SPEED_KMH   = 80.0

    # FIX 3
    SLOW_THRESHOLD  = 15.0

    # Real-world widths per COCO class (metres)
    # car=2, motorcycle=3, bus=5, truck=7
    VEHICLE_WIDTH_M = {2: 1.8, 3: 0.7, 5: 2.5, 7: 2.4}

    def __init__(self, fps, focal_px, window=7):
        """
        fps      : video frame rate
        focal_px : focal length in pixels  (fx from CameraCalibration)
        window   : frames to average over (smoothing)
        """
        self.fps      = fps
        self.focal_px = focal_px
        self.window   = window
        self._speeds  = {}   # track_id → smoothed speed

    def _metres_per_pixel(self, track):
        """
        Estimate real-world distance represented by 1 pixel at this vehicle's depth.
        Uses apparent pixel width + known real width.
        """
        px_width   = track.kf.get_pixel_width()
        real_width = self.VEHICLE_WIDTH_M.get(track.class_id, 1.8)
        # distance_m = (real_width × focal_px) / px_width
        # mpp        = real_width / px_width   (focal cancels in speed)
        return real_width / max(px_width, 1.0)

    def update(self, tracks):
        """
        Args:
            tracks : confirmed Track objects
        Returns:
            dict  : track_id → {speed_kmh, is_slow, centroid, mpp}
        """
        results  = {}
        live_ids = {t.id for t in tracks}

        for track in tracks:
            if len(track.history) < 2:
                continue

            hist = track.history[-self.window:]
            mpp  = self._metres_per_pixel(track)

            # Sum lateral (x-axis) displacement — dominant for side-view
            total_m = 0.0
            for i in range(1, len(hist)):
                dx = abs(hist[i][0] - hist[i-1][0])   # lateral pixels
                dy = abs(hist[i][1] - hist[i-1][1])   # vertical pixels
                pixel_dist = np.sqrt(dx**2 + dy**2)
                total_m   += pixel_dist * mpp

            elapsed_s = len(hist) / self.fps
            speed_ms  = total_m / (elapsed_s + 1e-6)
            speed_kmh = speed_ms * 3.6

            # EMA smoothing
            prev   = self._speeds.get(track.id, speed_kmh)
            smooth = 0.7 * prev + 0.3 * speed_kmh

            # ── FIX 1: Hard speed cap ──────────────────────────────
            smooth = min(smooth, self.MAX_SPEED_KMH)

            self._speeds[track.id] = smooth

            # ── FIX 3: Correct slow threshold ─────────────────────
            results[track.id] = {
                'speed_kmh': round(smooth, 1),
                'is_slow':   smooth < self.SLOW_THRESHOLD,
                'centroid':  track.centroid.tolist(),
                'mpp':       round(mpp, 4),
            }

        # Cleanup dead tracks
        self._speeds = {k: v for k, v in self._speeds.items() if k in live_ids}
        return results


# ─────────────────────────────────────────────────────────────
#  Main — run on video
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from ultralytics import YOLO
    from Speed import CameraCalibration

    # ── Config ────────────────────────────────────────────────
    VIDEO_PATH  = "VID-20260617-WA0018 (1).mp4"
    OUTPUT_PATH = "speed_output.mp4"

    # YOLO COCO vehicle class IDs
    VEHICLE_CLASSES = {2, 3, 5, 7}   # car, motorcycle, bus, truck

    # ── Open video ────────────────────────────────────────────
    cap   = cv2.VideoCapture(VIDEO_PATH)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {W}x{H}  {fps:.1f} fps  {total} frames")

    # ── Camera calibration (SIDE-VIEW phone camera) ───────────
    # pitch=0°    → camera is horizontal (not tilted down)
    # mount_height_m=8.0 → estimated lateral distance to vehicles
    # focal_length_mm=4.0, sensor_width_mm=5.6 → typical phone camera
    calib    = CameraCalibration(
        focal_length_mm=4.0,
        sensor_width_mm=5.6,
        image_width_px=W,
        image_height_px=H,
        mount_height_m=8.0,
        mount_pitch_deg=0.0,
    )
    focal_px = calib.fx   # pixels — passed to SideViewSpeedEstimator

    # ── Sub-systems ───────────────────────────────────────────
    tracker   = CentroidTracker(iou_threshold=0.35, max_misses=10)

    # FIX 2: Use SideViewSpeedEstimator instead of SpeedEstimator
    # FIX 1+3 are baked into SideViewSpeedEstimator
    estimator = SideViewSpeedEstimator(fps=fps, focal_px=focal_px, window=7)

    model = YOLO("yolov8n.pt")

    # ── Output video writer ───────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (W, H))

    speed_log : dict = {}
    class_log : dict = {}

    print(f"\n{'Frame':>6}  {'Track':>6}  {'Class':>10}  {'Speed':>8}  {'m/px':>7}")
    print("─" * 50)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ── YOLO detection ────────────────────────────────────
        results    = model(frame, verbose=False, conf=0.30)[0]
        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({"box": [x1, y1, x2, y2], "class_id": cls})

        # ── Tracker: assign IDs ───────────────────────────────
        confirmed = tracker.update(detections)
        for t in confirmed:
            class_log[t.id] = t.class_id

        # ── Speed estimation (FIX 1+2+3 active) ──────────────
        speeds = estimator.update(confirmed)
        for tid, info in speeds.items():
            speed_log[tid] = info['speed_kmh']
            cls_name = {2:"car", 3:"moto", 5:"bus", 7:"truck"}.get(
                class_log.get(tid, 2), "vehicle")
            print(f"  {frame_idx:>6}  {tid:>6}  {cls_name:>10}  "
                  f"{info['speed_kmh']:>6.1f} km/h  {info['mpp']:>7.4f}")

        # ── Draw bounding boxes + labels ──────────────────────
        for track in confirmed:
            x1, y1, x2, y2 = map(int, track.kf.get_box())
            spd  = speed_log.get(track.id)

            # FIX 3+4: colour threshold now 15 km/h (was 30)
            # Green = moving (≥15 km/h), Yellow = slow (<15 km/h)
            if spd is None:
                color = (180, 180, 180)   # grey — no estimate yet
            elif spd >= 15:
                color = (0, 220, 0)       # green — moving
            else:
                color = (0, 165, 255)     # orange — slow / parked

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"ID {track.id}"
            if spd is not None:
                slow_tag = " SLOW" if spd < 15 else ""
                label   += f"  {spd:.1f} km/h{slow_tag}"

            cv2.putText(frame, label,
                        (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Frame counter overlay
        cv2.putText(frame, f"Frame {frame_idx}/{total}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 60 == 0:
            print(f"  ... {frame_idx}/{total} frames processed")

    cap.release()
    writer.release()
    print(f"\nDone. Saved → {OUTPUT_PATH}")

    # ── Speed summary ─────────────────────────────────────────
    if speed_log:
        print("\n─── Speed summary ───────────────────────────────────")
        for tid, spd in sorted(speed_log.items()):
            cls_name = {2:"car", 3:"moto", 5:"bus", 7:"truck"}.get(
                class_log.get(tid, 2), "vehicle")
            # FIX 3: flag uses 15 km/h threshold
            flag = "  ⚠ slow" if spd < 15 else ""
            print(f"  Track {tid:>3} [{cls_name:>7}]:  {spd:>6.1f} km/h{flag}")

        speeds_all = list(speed_log.values())
        # FIX 1: cap shown in summary too
        peak = max(speeds_all)
        mean = sum(speeds_all) / len(speeds_all)
        print(f"\n  Peak : {peak:.1f} km/h  {'(capped at 80)' if peak >= 79 else ''}")
        print(f"  Mean : {mean:.1f} km/h")
        print(f"  Slow : {sum(1 for s in speeds_all if s < 15)} / {len(speeds_all)} vehicles")
