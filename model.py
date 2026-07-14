"""
Detection Layer
===============
Stage 1: YOLOv8 detects all vehicles in a frame
Stage 2: Fine-grained classifier crops each vehicle
         and classifies as emergency vs civilian
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np


# ─────────────────────────────────────────
#  Emergency Vehicle Classifier (Stage 2)
# ─────────────────────────────────────────

class EmergencyVehicleClassifier(nn.Module):
    """
    Lightweight MobileNetV3-based classifier.
    Input : cropped vehicle image [B, 3, 224, 224]
    Output: logits over [civilian, police, ambulance, fire_truck]
    """

    CLASSES = ['civilian', 'police', 'ambulance', 'fire_truck']
    NUM_CLASSES = len(CLASSES)

    def __init__(self, pretrained=True, dropout=0.3):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)

        # Replace classifier head
        in_features = backbone.classifier[3].in_features
        backbone.classifier[3] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, self.NUM_CLASSES)
        )
        self.backbone = backbone

        # Visual cue attention branch (focuses on light-bar region = top 1/3)
        self.attention = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, self.NUM_CLASSES)
        )

    def forward(self, x):
        main_out    = self.backbone(x)
        # Attention branch only sees top third (where light-bars live)
        top_third   = x[:, :, :x.shape[2]//3, :]
        attn_out    = self.attention(top_third)
        return main_out + 0.3 * attn_out   # fuse


# ─────────────────────────────────────────
#  Full Detection Pipeline
# ─────────────────────────────────────────

class VehicleDetectionPipeline:
    """
    Two-stage pipeline:
      1. YOLOv8 → bounding boxes of all vehicles
      2. EmergencyVehicleClassifier → type per crop

    Usage:
        pipeline = VehicleDetectionPipeline(
            yolo_weights='yolov8n.pt',
            classifier_weights='./results/detection/classifier_best.pth'
        )
        results = pipeline.predict(frame_bgr)
    """

    def __init__(self, yolo_weights='yolov8n.pt',
                 classifier_weights=None, device=None,
                 conf_threshold=0.4):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf_threshold = conf_threshold

        # Stage 1 — YOLOv8 (via ultralytics)
        try:
            from ultralytics import YOLO
            self.yolo = YOLO(yolo_weights)
            self.yolo_available = True
        except ImportError:
            print("[Warning] ultralytics not installed. YOLO stage disabled.")
            print("          Run: pip install ultralytics")
            self.yolo_available = False

        # Stage 2 — Fine-grained classifier
        self.classifier = EmergencyVehicleClassifier(pretrained=False).to(self.device)
        if classifier_weights:
            ckpt = torch.load(classifier_weights, map_location=self.device)
            state = ckpt.get('model_state', ckpt)
            self.classifier.load_state_dict(state)
            print(f"[Info] Classifier loaded: {classifier_weights}")
        self.classifier.eval()

        self.preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(self, frame_bgr):
        """
        Args:
            frame_bgr: numpy array (H, W, 3) in BGR (OpenCV format)
        Returns:
            list of dicts: {box, class_id, class_name, conf, is_emergency}
        """
        import cv2
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        results = []

        if self.yolo_available:
            yolo_out = self.yolo(frame_bgr, conf=self.conf_threshold, verbose=False)
            boxes = yolo_out[0].boxes
        else:
            # Fallback: treat whole frame as single vehicle crop for demo
            boxes = None

        if boxes is not None and len(boxes):
            # COCO vehicle class IDs: 2=car, 3=motorcycle, 5=bus, 7=truck
            vehicle_ids = {2, 3, 5, 7}
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id not in vehicle_ids:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                conf = float(box.conf[0])

                # Crop and classify
                crop = Image.fromarray(frame_rgb[y1:y2, x1:x2])
                em_cls, em_conf = self._classify_crop(crop)

                results.append({
                    'box':          [x1, y1, x2, y2],
                    'yolo_class':   cls_id,
                    'conf':         conf,
                    'em_class_id':  em_cls,
                    'em_class':     EmergencyVehicleClassifier.CLASSES[em_cls],
                    'em_conf':      em_conf,
                    'is_emergency': em_cls > 0,
                })

        return results

    @torch.no_grad()
    def _classify_crop(self, pil_crop):
        tensor = self.preprocess(pil_crop).unsqueeze(0).to(self.device)
        logits = self.classifier(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        cls_id = probs.argmax().item()
        return cls_id, probs[cls_id].item()

    def draw(self, frame_bgr, detections):
        """Draw bounding boxes and labels on frame."""
        import cv2
        COLORS = {
            'civilian':   (180, 180, 180),
            'police':     (255, 100,   0),
            'ambulance':  (0,   200, 255),
            'fire_truck': (0,    60, 255),
        }
        out = frame_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = det['box']
            color = COLORS[det['em_class']]
            label = f"{det['em_class']} {det['em_conf']:.0%}"
            thickness = 3 if det['is_emergency'] else 1
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(out, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return out
