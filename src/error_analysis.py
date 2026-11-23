import json
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from src.postprocess import compute_iou


def load_gt_boxes(lbl_path: Path, img_w: int, img_h: int):
    boxes = []
    if not lbl_path.exists():
        return boxes
    for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split()
        cls_id = int(parts[0])
        xc, yc, w, h = map(float, parts[1:])
        x1 = (xc - w / 2) * img_w
        y1 = (yc - h / 2) * img_h
        x2 = (xc + w / 2) * img_w
        y2 = (yc + h / 2) * img_h
        boxes.append((cls_id, (x1, y1, x2, y2)))
    return boxes


def main():
    summary = json.loads(Path("reports/metrics_summary.json").read_text(encoding="utf-8"))
    best_exp = summary["selected_model"]["experiment"]
    weights = summary[best_exp]["weights_path"]
    model = YOLO(weights)

    test_imgs = sorted(Path("data/processed/test/images").glob("*.jpg"))
    test_lbls = Path("data/processed/test/labels")
    gallery_dir = Path("reports/error_gallery")
    gallery_dir.mkdir(parents=True, exist_ok=True)

    # Дополнительно сохраним PR/F1 кривые для отчёта, если их не было
    val_res = model.val(data="configs/ppe.yaml", split="test", imgsz=640, batch=8, device=0, plots=True, verbose=False)
    save_dir = Path(val_res.save_dir)
    for fig in ["PR_curve.png", "F1_curve.png", "P_curve.png", "R_curve.png"]:
        if (save_dir / fig).exists():
            Path(f"reports/figures/FINAL_TEST_{fig}").write_bytes((save_dir / fig).read_bytes())

    error_records = []
    for img_path in test_imgs:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        gt = load_gt_boxes(test_lbls / f"{img_path.stem}.txt", w, h)
        pred_res = model.predict(img, conf=0.30, verbose=False)[0]
        preds = []
        for b in pred_res.boxes:
            preds.append((int(b.cls[0]), float(b.conf[0]), tuple(map(float, b.xyxy[0].tolist()))))

        matched_gt = set()
        fp_count = 0
        for p_cls, p_conf, p_box in preds:
            best_iou, best_j = 0.0, -1
            for j, (g_cls, g_box) in enumerate(gt):
                if j not in matched_gt and g_cls == p_cls:
                    iou = compute_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
            if best_iou >= 0.5:
                matched_gt.add(best_j)
            else:
                fp_count += 1

        fn_count = len(gt) - len(matched_gt)
        total_err = fp_count + fn_count
        if total_err > 0:
            error_records.append((total_err, fp_count, fn_count, img_path, img, gt, preds, pred_res.names))

    error_records.sort(key=lambda x: x[0], reverse=True)
    gallery_manifest = []

    for idx, (tot, fp, fn, img_path, img, gt, preds, names) in enumerate(error_records[:12], start=1):
        vis = img.copy()
        for g_cls, (x1, y1, x2, y2) in gt:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0), 2)
            cv2.putText(vis, f"GT:{names[g_cls]}", (int(x1), max(15, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
        for p_cls, p_conf, (x1, y1, x2, y2) in preds:
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(vis, f"PR:{names[p_cls]} {p_conf:.2f}", (int(x1), min(vis.shape[0] - 5, int(y2) + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

        out_name = f"err_{idx:02d}_{img_path.stem}.jpg"
        cv2.imwrite(str(gallery_dir / out_name), vis)
        gallery_manifest.append({
            "id": idx,
            "file": out_name,
            "source_image": img_path.name,
            "false_positives": fp,
            "false_negatives": fn,
            "gt_classes": [names[c] for c, _ in gt],
            "pred_classes": [names[c] for c, _, _ in preds],
        })

    Path("reports/error_gallery/gallery_manifest.json").write_text(
        json.dumps(gallery_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved {len(gallery_manifest)} error gallery images to reports/error_gallery/")


if __name__ == "__main__":
    main()
