import json
from pathlib import Path
import cv2
from ultralytics import YOLO
from src.postprocess import Detection, TemporalViolationTracker, apply_nms_and_filter


def run_inference(
    weights_path: str,
    source_path: str,
    output_dir: str = "reports/inference_demo",
    conf_thresh: float = 0.30,
    iou_thresh: float = 0.45,
    min_streak: int = 3,
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model = YOLO(weights_path)
    tracker = TemporalViolationTracker(min_consecutive_frames=min_streak)

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open source: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_out_file = out_path / f"annotated_{Path(source_path).stem}.mp4"
    writer = cv2.VideoWriter(
        str(video_out_file),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_logs = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.15, verbose=False)[0]
        raw_dets = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            raw_dets.append(Detection(cls_id, cls_name, conf, (x1, y1, x2, y2)))

        clean_dets = apply_nms_and_filter(raw_dets, conf_threshold=conf_thresh, iou_threshold=iou_thresh)
        alarm_state = tracker.update(clean_dets)

        for d in clean_dets:
            x1, y1, x2, y2 = map(int, d.bbox)
            is_violation = d.class_name in tracker.VIOLATION_CLASSES
            color = (0, 0, 255) if is_violation else (0, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{d.class_name} {d.confidence:.2f}"
            cv2.putText(frame, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if alarm_state["is_alarm"]:
            banner = "ALARM! Confirmed violations: " + ", ".join(alarm_state["confirmed_alerts"])
            cv2.rectangle(frame, (10, 10), (width - 10, 50), (0, 0, 200), -1)
            cv2.putText(frame, banner, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        writer.write(frame)

        if alarm_state["raw_violations"] or alarm_state["is_alarm"]:
            frame_logs.append({
                "frame": frame_idx,
                "detections_count": len(clean_dets),
                **alarm_state,
            })

        frame_idx += 1

    cap.release()
    writer.release()

    log_file = out_path / "violation_events_log.json"
    log_file.write_text(json.dumps(frame_logs, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Processed frames: {frame_idx} | Frames with PPE events: {len(frame_logs)}")

if __name__ == "__main__":
    summary = json.loads(Path("reports/metrics_summary.json").read_text(encoding="utf-8"))
    best_exp = summary["_selected_model"]["experiment"] if "_selected_model" in summary else summary["selected_model"]["experiment"]
    best_weights = summary[best_exp]["hweights_path"] if "hweights_path" in summary[best_exp] else summary[best_exp]["weights_path"]
    run_inference(best_weights, "source_files/hardhat.mp4")
