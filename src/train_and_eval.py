import json
import shutil
from pathlib import Path
from ultralytics import YOLO


def extract_metrics(val_results) -> dict:
    """Извлекает основные метрики детекции из объекта результатов Ultralytics."""
    box = val_results.box
    return {
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "mAP50": round(float(box.map50), 4),
        "mAP50_95": round(float(box.map), 4),
    }


def main():
    data_yaml = str(Path("configs/ppe.yaml").resolve())
    base_weights = "models/yolov8n.pt" if Path("models/yolov8n.pt").exists() else "yolov8n.pt"

    experiments = {
        "E1_baseline": {
            "description": "YOLOv8n baseline (imgsz=640, epochs=30, batch=8, без mosaic/mixup)",
            "train_kwargs": dict(
                data=data_yaml,
                epochs=30,
                imgsz=640,
                batch=8,
                device=0,
                seed=42,
                workers=2,
                project="runs",
                name="E1_baseline",
                exist_ok=True,
                mosaic=0.0,
                mixup=0.0,
                degrees=0.0,
                scale=0.1,
                hsv_s=0.2,
                hsv_v=0.1,
                cos_lr=False,
            ),
        },
        "E2_improved": {
            "description": "YOLOv8n improved (imgsz=640, epochs=30, batch=8, mosaic=1.0, mixup=0.1, cos_lr=True)",
            "train_kwargs": dict(
                data=data_yaml,
                epochs=30,
                imgsz=640,
                batch=8,
                device=0,
                seed=42,
                workers=2,
                project="runs",
                name="E2_improved",
                exist_ok=True,
                mosaic=1.0,
                mixup=0.1,
                degrees=10.0,
                scale=0.5,
                hsv_s=0.6,
                hsv_v=0.4,
                cos_lr=True,
            ),
        },
    }

    summary = {}
    best_exp_name = None
    best_val_map = -1.0

    for exp_name, cfg in experiments.items():
        print(f"\n{'='*60}\nЗапуск эксперимента: {exp_name}\n{cfg['description']}\n{'='*60}")
        model = YOLO(base_weights)
        model.train(**cfg["train_kwargs"])

        best_pt = Path(f"runs/{exp_name}/weights/best.pt")
        best_model = YOLO(str(best_pt))

        print(f"\nВалидация {exp_name} на split='val'...")
        val_res = best_model.val(data=data_yaml, split="val", imgsz=640, batch=8, device=0, verbose=False)
        val_metrics = extract_metrics(val_res)

        summary[exp_name] = {
            "description": cfg["description"],
            "weights_path": str(best_pt),
            "val_metrics": val_metrics,
        }
        print(f"Результаты {exp_name} на VAL: {val_metrics}")

        if val_metrics["mAP50_95"] > best_val_map:
            best_val_map = val_metrics["mAP50_95"]
            best_exp_name = exp_name

    # Финальная оценка выбранной по val модели на замороженной выборке test
    print(f"\n{'='*60}\nЛучшая конфигурация по val: {best_exp_name} (mAP50-95={best_val_map})")
    print("Запускаем однократную финальную оценку на замороженном TEST...\n{'='*60}")

    winner_weights = summary[best_exp_name]["weights_path"]
    winner_model = YOLO(winner_weights)
    test_res = winner_model.val(
        data=data_yaml,
        split="test",
        imgsz=640,
        batch=8,
        device=0,
        project="runs",
        name=f"{best_exp_name}_final_test",
        exist_ok=True,
        verbose=False,
    )
    test_metrics = extract_metrics(test_res)

    # Сохраняем поклассовые метрики на test
    per_class_test = {}
    names = test_res.names
    for idx, cls_idx in enumerate(test_res.box.ap_class_index):
        cls_name = names[int(cls_idx)]
        per_class_test[cls_name] = {
            "precision": round(float(test_res.box.p[idx]), 4),
            "recall": round(float(test_res.box.r[idx]), 4),
            "mAP50": round(float(test_res.box.ap50[idx]), 4),
            "mAP50_95": round(float(test_res.box.ap[idx]), 4),
        }

    summary["selected_model"] = {
        "experiment": best_exp_name,
        "test_metrics": test_metrics,
        "per_class_test_metrics": per_class_test,
    }

    # Копируем ключевые графики в reports/figures/ для документации
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    for exp_name in experiments:
        for fig_name in ["results.png", "confusion_matrix.png", "PR_curve.png", "F1_curve.png"]:
            src_fig = Path(f"runs/{exp_name}/{fig_name}")
            if src_fig.exists():
                shutil.copy2(src_fig, figures_dir / f"{exp_name}_{fig_name}")

    for fig_name in ["confusion_matrix.png", "PR_curve.png", "F1_curve.png", "val_batch0_pred.jpg"]:
        src_fig = Path(f"runs/{best_exp_name}_final_test/{fig_name}")
        if src_fig.exists():
            shutil.copy2(src_fig, figures_dir / f"FINAL_TEST_{fig_name}")

    Path("reports/metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nГотово! Все метрики сохранены в reports/metrics_summary.json, графики — в reports/figures/.")


if __name__ == "__main__":
    main()
