"""
Test Detection Pipeline
========================
Test on images:
    python detection/test.py --source ./test_images/ --classifier_weights ./results/detection/classifier_best.pth

Test on video:
    python detection/test.py --source ./test_video.mp4 --classifier_weights ./results/detection/classifier_best.pth

Test on webcam:
    python detection/test.py --source 0
"""

import argparse
import cv2
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from model import VehicleDetectionPipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--source',              type=str, default='0')
    p.add_argument('--yolo_weights',        type=str, default='yolov8n.pt')
    p.add_argument('--classifier_weights',  type=str, default=None)
    p.add_argument('--output_dir',          type=str, default='./results/detection')
    p.add_argument('--conf',                type=float, default=0.4)
    p.add_argument('--save_output',         action='store_true')
    return p.parse_args()


def process_image(pipeline, img_path, out_dir, args):
    frame = cv2.imread(str(img_path))
    if frame is None:
        print(f"[Error] Could not read: {img_path}")
        return

    detections = pipeline.predict(frame)
    annotated  = pipeline.draw(frame, detections)

    print(f"\n{img_path.name}: {len(detections)} vehicles detected")
    for i, det in enumerate(detections):
        tag = "🚨 EMERGENCY" if det['is_emergency'] else "civilian"
        print(f"  [{i+1}] {tag} | class={det['em_class']} conf={det['em_conf']:.0%} | box={det['box']}")

    if args.save_output:
        out_path = Path(out_dir) / f"detected_{img_path.name}"
        cv2.imwrite(str(out_path), annotated)
        print(f"  Saved: {out_path}")
    else:
        cv2.imshow('Detections', annotated)
        cv2.waitKey(0)


def process_video(pipeline, source, out_dir, args):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        print(f"[Error] Cannot open: {source}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None

    if args.save_output:
        out_path = str(Path(out_dir) / 'detected_video.mp4')
        writer   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frame_count = 0
    total_time  = 0.0
    print(f"[Info] Processing video: {source}")
    print("[Info] Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0         = time.time()
        detections = pipeline.predict(frame)
        elapsed    = time.time() - t0
        total_time += elapsed
        frame_count += 1

        annotated = pipeline.draw(frame, detections)

        # FPS overlay
        inf_fps = frame_count / total_time
        cv2.putText(annotated, f"FPS: {inf_fps:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Emergency alert overlay
        emergency = [d for d in detections if d['is_emergency']]
        if emergency:
            cv2.putText(annotated,
                        f"⚠ EMERGENCY: {', '.join(d['em_class'] for d in emergency)}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if writer:
            writer.write(annotated)

        cv2.imshow('Detection', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"\n[Done] {frame_count} frames | avg FPS: {frame_count/total_time:.1f}")


def test():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    pipeline = VehicleDetectionPipeline(
        yolo_weights=args.yolo_weights,
        classifier_weights=args.classifier_weights,
        conf_threshold=args.conf,
    )

    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}
    src = Path(args.source)

    if src.is_dir():
        imgs = [p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS]
        print(f"[Info] Testing on {len(imgs)} images in {src}")
        for img_path in sorted(imgs):
            process_image(pipeline, img_path, args.output_dir, args)
    elif src.is_file() and src.suffix.lower() in IMG_EXTS:
        process_image(pipeline, src, args.output_dir, args)
    else:
        process_video(pipeline, args.source, args.output_dir, args)


if __name__ == '__main__':
    test()
