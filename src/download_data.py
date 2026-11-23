import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

CLASS_NAMES = [
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
    "machinery",
    "vehicle",
]

def convert_bbox_to_yolo(bbox, img_w, img_h):
    """Конвертирует [x_min, y_min, width, height] в нормированный YOLO [x_center, y_center, w, h]."""
    x_min, y_min, w, h = bbox
    x_center = (x_min + w / 2.0) / img_w
    y_center = (y_min + h / 2.0) / img_h
    norm_w = w / img_w
    norm_h = h / img_h

    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    norm_w = max(0.0, min(1.0, norm_w))
    norm_h = max(0.0, min(1.0, norm_h))
    return x_center, y_center, norm_w, norm_h


def main():
    raw_images_dir = Path("data/raw/images")
    raw_labels_dir = Path("data/raw/labels")
    raw_images_dir.mkdir(parents=True, exist_ok=True)
    raw_labels_dir.mkdir(parents=True, exist_ok=True)

    print("Загружаем датасет keremberke/construction-safety-object-detection (Parquet)...")
    ds = load_dataset(
        "keremberke/construction-safety-object-detection",
        name="default",
        revision="refs/convert/parquet",
    )

    total_saved = 0
    for split_name in ds.keys():
        split_data = ds[split_name]
        print(f"Обработка части '{split_name}' ({len(split_data)} кадров)...")
        for idx, item in enumerate(tqdm(split_data)):
            image = item["image"].convert("RGB")
            img_w, img_h = image.size
            image_id = item.get("image_id", f"{split_name}_{idx}")
            file_stem = f"{split_name}_{image_id:05d}" if isinstance(image_id, int) else f"{split_name}_{image_id}"

            img_path = raw_images_dir / f"{file_stem}.jpg"
            lbl_path = raw_labels_dir / f"{file_stem}.txt"

            image.save(img_path, format="JPEG", quality=95)

            objects = item["objects"]
            lines = []
            for cat_id, bbox in zip(objects["category"], objects["bbox"]):
                xc, yc, nw, nh = convert_bbox_to_yolo(bbox, img_w, img_h)
                if nw > 0 and nh > 0:
                    lines.append(f"{cat_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

            lbl_path.write_text("\n".join(lines), encoding="utf-8")
            total_saved += 1

    print(f"\nГотово! Сохранено {total_saved} изображений и файлов разметки в data/raw/.")


if __name__ == "__main__":
    main()
