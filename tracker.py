"""
Multi-Object Tracker (Centroid + Kalman Filter)
================================================
Lightweight tracker similar in spirit to ByteTrack/DeepSORT.
Maintains vehicle identities across frames using:
  - IoU-based box matching
  - Kalman filter for smooth trajectory prediction
"""

import numpy as np
from collections import defaultdict


class KalmanBox:
    """
    Kalman filter for a single bounding box.
    State: [x, y, w, h, vx, vy, vw, vh]
    Observation: [x, y, w, h]
    """
    def __init__(self, box):
        x, y, w, h = self._to_xywh(box)
        # State: position + velocity
        self.state = np.array([x, y, w, h, 0., 0., 0., 0.], dtype=float)
        self.P = np.eye(8) * 10.           # Covariance
        self.F = np.eye(8)                  # State transition
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = self.F[3, 7] = 1.0
        self.H = np.eye(4, 8)               # Observation
        self.Q = np.eye(8) * 0.01          # Process noise
        self.R = np.eye(4) * 1.0           # Measurement noise

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
        self.P = (np.eye(8) - K @ self.H) @ self.P

    def get_box(self):
        x, y, w, h = self.state[:4]
        return [x, y, x + w, y + h]

    def get_centroid(self):
        x, y, w, h = self.state[:4]
        return np.array([x + w / 2, y + h / 2])

    def get_velocity(self):
        return self.state[4:6]   # vx, vy in pixels/frame


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

    @property
    def velocity_px(self):
        """Pixel velocity in (vx, vy) per frame."""
        return self.kf.get_velocity()


def iou(box_a, box_b):
    """Intersection over Union between two boxes [x1,y1,x2,y2]."""
    xa = max(box_a[0], box_b[0]); ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2]); yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
    return inter / (area_a + area_b - inter + 1e-6)


class CentroidTracker:
    """
    IoU + Kalman filter multi-object tracker.
    Drop-in replacement for ByteTrack/DeepSORT for this project.
    """

    def __init__(self, iou_threshold=0.3, max_misses=5):
        self.tracks       = []
        self.iou_threshold = iou_threshold
        self.max_misses   = max_misses

    def update(self, detections):
        """
        Args:
            detections: list of {'box': [x1,y1,x2,y2], 'class_id': int}
        Returns:
            list of Track objects (confirmed tracks only)
        """
        # Predict all existing tracks
        for track in self.tracks:
            track.predict()

        if not detections:
            for track in self.tracks:
                track.mark_missed()
        else:
            det_boxes = [d['box']      for d in detections]
            det_cls   = [d['class_id'] for d in detections]
            matched_tracks = set()
            matched_dets   = set()

            # Greedy IoU matching
            if self.tracks:
                iou_matrix = np.zeros((len(self.tracks), len(detections)))
                for i, track in enumerate(self.tracks):
                    for j, box in enumerate(det_boxes):
                        iou_matrix[i, j] = iou(track.kf.get_box(), box)

                while True:
                    idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                    if iou_matrix[idx] < self.iou_threshold:
                        break
                    i, j = idx
                    self.tracks[i].update(det_boxes[j], det_cls[j])
                    matched_tracks.add(i)
                    matched_dets.add(j)
                    iou_matrix[i, :] = -1
                    iou_matrix[:, j] = -1

            # Unmatched tracks
            for i, track in enumerate(self.tracks):
                if i not in matched_tracks:
                    track.mark_missed()

            # New tracks for unmatched detections
            for j in range(len(detections)):
                if j not in matched_dets:
                    self.tracks.append(Track(det_boxes[j], det_cls[j]))

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        return [t for t in self.tracks if t.is_confirmed]
