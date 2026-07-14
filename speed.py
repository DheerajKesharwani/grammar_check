"""
Speed Estimation
================
Converts pixel displacement of tracked vehicles to real-world speed
using camera calibration parameters.

Two methods:
  A) Calibration-based  — uses focal length + mounting height + angle
  B) Optical flow       — fallback when calibration is unavailable
"""

import numpy as np
import cv2


class CameraCalibration:
    """
    Stores and applies camera intrinsic + extrinsic parameters.

    For a dashcam mounted on a car:
        focal_length_mm : physical focal length (e.g. 3.6mm dashcam lens)
        sensor_width_mm : sensor width (e.g. 1/2.9" CMOS ≈ 5.6mm)
        image_width_px  : frame width in pixels (e.g. 1920)
        mount_height_m  : camera height above road (e.g. 1.2m for a car)
        mount_pitch_deg : downward tilt angle (e.g. 5° for dashcam)
    """

    def __init__(self,
                 focal_length_mm=3.6,
                 sensor_width_mm=5.6,
                 image_width_px=1920,
                 image_height_px=1080,
                 mount_height_m=1.2,
                 mount_pitch_deg=5.0):

        self.fx = (focal_length_mm / sensor_width_mm) * image_width_px
        self.fy = self.fx   # assume square pixels
        self.cx = image_width_px  / 2.0
        self.cy = image_height_px / 2.0
        self.W  = image_width_px
        self.H  = image_height_px

        self.mount_height_m  = mount_height_m
        self.mount_pitch_rad = np.deg2rad(mount_pitch_deg)

    def pixel_to_ground_meters(self, px, py):
        """
        Project a pixel (px, py) onto the flat ground plane.
        Returns (X, Z) in meters, where Z is forward distance.

        Uses the road-plane homography approximation valid when
        the road is flat and the camera pitch is small.
        """
        # Normalized image coordinates
        xn = (px - self.cx) / self.fx
        yn = (py - self.cy) / self.fy

        # Rotate by camera pitch
        pitch = self.mount_pitch_rad
        yn_rotated = yn * np.cos(pitch) - np.sin(pitch)
        zn_rotated = yn * np.sin(pitch) + np.cos(pitch)

        if abs(zn_rotated) < 1e-6:
            return None

        # Scale factor from camera height
        scale = self.mount_height_m / zn_rotated

        X = xn * scale
        Z = scale   # forward distance

        return X, Z

    def displacement_to_meters(self, centroid_prev, centroid_curr, y_ref=None):
        """
        Convert centroid pixel displacement to real-world distance (metres).

        centroid_prev, centroid_curr : (px, py) tuples
        y_ref : reference y for depth estimation (use bottom of bounding box)
        """
        y = y_ref or centroid_prev[1]

        p_prev = self.pixel_to_ground_meters(centroid_prev[0], y)
        p_curr = self.pixel_to_ground_meters(centroid_curr[0], y)

        if p_prev is None or p_curr is None:
            return 0.0

        dx = p_curr[0] - p_prev[0]   # lateral
        dz = p_curr[1] - p_prev[1]   # forward

        return np.sqrt(dx**2 + dz**2)


class SpeedEstimator:
    """
    Estimates vehicle speed (km/h) from tracked centroid history.
    Applies smoothing to reduce jitter.
    """

    def __init__(self, calibration: CameraCalibration, fps: float,
                 window=5, slow_threshold_kmh=20.0):
        self.calib              = calibration
        self.fps                = fps
        self.window             = window              # frames to average over
        self.slow_threshold     = slow_threshold_kmh
        self._speeds            = {}                  # track_id -> deque of speeds

    def update(self, tracks):
        """
        Args:
            tracks: list of Track objects with .id and .history
        Returns:
            dict: track_id -> {'speed_kmh': float, 'is_slow': bool}
        """
        results = {}
        for track in tracks:
            if len(track.history) < 2:
                continue

            # Use last `window` frames for smoothing
            hist = track.history[-self.window:]
            total_dist_m = 0.0
            for i in range(1, len(hist)):
                dist = self.calib.displacement_to_meters(hist[i-1], hist[i])
                total_dist_m += dist

            # Average speed over the window
            elapsed_sec = len(hist) / self.fps
            speed_ms    = total_dist_m / (elapsed_sec + 1e-6)
            speed_kmh   = speed_ms * 3.6

            # Smooth using EMA
            prev = self._speeds.get(track.id, speed_kmh)
            smooth = 0.7 * prev + 0.3 * speed_kmh
            self._speeds[track.id] = smooth

            results[track.id] = {
                'speed_kmh': round(smooth, 1),
                'is_slow':   smooth < self.slow_threshold,
                'track_id':  track.id,
                'centroid':  track.centroid.tolist(),
            }

        # Clean up dead tracks
        live_ids = {t.id for t in tracks}
        self._speeds = {k: v for k, v in self._speeds.items() if k in live_ids}

        return results


# ─────────────────────────────────────────
#  Optical Flow Speed Estimator (Fallback)
# ─────────────────────────────────────────

class OpticalFlowSpeedEstimator:
    """
    Uses Lucas-Kanade optical flow to estimate vehicle speed
    when camera calibration is unavailable.

    Reports relative speed in pixels/frame — use a scale factor
    (measured against a known reference) to convert to km/h.
    """

    def __init__(self, scale_pxf_to_kmh=0.5):
        """
        scale_pxf_to_kmh: tunable constant.
        Calibrate by: film a vehicle at known speed, measure px/frame,
                      compute scale = known_kmh / measured_pxf.
        """
        self.scale = scale_pxf_to_kmh
        self.prev_gray = None
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3,
                                   minDistance=7, blockSize=7)

    def estimate(self, frame_gray, boxes):
        """
        Args:
            frame_gray : current grayscale frame
            boxes      : list of [x1,y1,x2,y2] bounding boxes to analyse
        Returns:
            list of speed estimates (pixels/frame) per box
        """
        results = []

        if self.prev_gray is None:
            self.prev_gray = frame_gray
            return [0.0] * len(boxes)

        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            roi_prev = self.prev_gray[y1:y2, x1:x2]
            roi_curr = frame_gray[y1:y2, x1:x2]

            if roi_prev.size == 0:
                results.append(0.0)
                continue

            # Detect good features to track in previous ROI
            pts = cv2.goodFeaturesToTrack(roi_prev, **self.feature_params)
            if pts is None or len(pts) == 0:
                results.append(0.0)
                continue

            pts_global = pts + np.array([[x1, y1]], dtype=np.float32)
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, frame_gray, pts_global, None, **self.lk_params)

            good_prev = pts_global[status.flatten() == 1]
            good_next = next_pts[status.flatten() == 1]

            if len(good_prev) == 0:
                results.append(0.0)
                continue

            motion = np.linalg.norm(good_next - good_prev, axis=1)
            median_flow = float(np.median(motion))
            results.append(median_flow * self.scale)

        self.prev_gray = frame_gray
        return results
