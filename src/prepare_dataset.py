import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

CLASS_NAMES = [
    "barricade",
    "dumpster",
    "excavators",
    "gloves",
    "hardhat",
    "mask",
    "no-hardhat",
    "no-mask",
    "no-safety vest",
    "person",
    "safety net",
    "safety shoes",
    "safety vest",
    "dump truck",
    "mini-van",
    "truck",
    "wheel loader",
]


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_dhash(img_path: Path, hash_size: int = 8) -> str:
    """Вычисляет перцептивный dHash для группировки визуально близких кадров одной сцены."""
    with Image.open(img_path) as img:
        gray = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray)
        diff = pixels[:, 1:] > pixels[:, :-1]
        bits = "".join("1" if b else "0" for b in diff.flatten()[:24])
        return f"{int(bits, 2):06x}"


def validate_label_file(lbl_path: Path, num_classes: int = 17) -> list[int]:
    """Проверяет корректность разметки YOLO и возвращает список ID классов на кадре."""
    if not lbl_path.exists():
        return []
    text = lbl_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    classes_present = []
    for line in text.splitlines():
        parts = line.strip().split()
        assert len(parts) == 5, f"Неверный формат строки в {lbl_path}: {line}"
        cls_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
        assert 0 <= cls_id < num_classes, f"Недопустимый класс {cls_id} в {lbl_path}"
        assert all(0.0 <= c <= 1.0 for c in coords), f"Координаты вне [0, 1] в {lbl_path}"
        classes_present.append(cls_id)
    return classes_present


def main():
    raw_images = sorted(Path("data/raw/images").glob("*.jpg"))
    raw_labels_dir = Path("data/raw/labels")
    processed_dir = Path("data/processed")

    if processed_dir.exists():
        shutil.rmtree(processed_dir)

    seen_hashes = set()
    records = []
    duplicates_removed = 0

    for img_path in raw_images:
        lbl_path = raw_labels_dir / f"{img_path.stem}.txt"
        sha = compute_sha256(img_path)
        if sha in seen_hashes:
            duplicates_removed += 1
            continue
        seen_hashes.add(sha)

        classes_present = validate_label_file(lbl_path, len(CLASS_NAMES))
        scene_group = compute_dhash(img_path)
        records.append({
            "stem": img_path.stem,
            "img_path": img_path,
            "lbl_path": lbl_path,
            "sha256": sha,
            "scene_group": scene_group,
            "classes": classes_present,
        })

    print(f"Всего кадров в raw: {len(raw_images)} | Удалено точных дубликатов: {duplicates_removed} | Уникальных: {len(records)}")

    indices = np.arange(len(records))
    groups = np.array([r["scene_group"] for r in records])

    # 1. Отделяем 70% train и 30% temp (val + test) с группировкой по сценам
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(gss1.split(indices, groups=groups))

    # 2. Делим temp пополам (по 15% от общего числа на val и test) с группировкой по сценам
    temp_groups = groups[temp_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    val_rel_idx, test_rel_idx = next(gss2.split(temp_idx, groups=temp_groups))
    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]

    splits = {"train": train_idx, "val": val_idx, "test": test_idx}

    manifest = {
        "random_seed": 42,
        "split_ratio_target": "70/15/15",
        "duplicates_removed": duplicates_removed,
        "classes": CLASS_NAMES,
        "splits": {},
    }

    for split_name, idxs in splits.items():
        img_out = processed_dir / split_name / "images"
        lbl_out = processed_dir / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        class_counts = {name: 0 for name in CLASS_NAMES}
        split_items = []

        for i in idxs:
            rec = records[i]
            shutil.copy2(rec["img_path"], img_out / f"{rec['stem']}.jpg")
            if rec["lbl_path"].exists():
                shutil.copy2(rec["lbl_path"], lbl_out / f"{rec['stem']}.txt")
            else:
                (lbl_out / f"{rec['stem']}.txt").write_text("", encoding="utf-8")

            for c_id in rec["classes"]:
                class_counts[CLASS_NAMES[c_id]] += 1

            split_items.append({
                "stem": rec["stem"],
                "sha256": rec["sha256"],
                "scene_group": rec["scene_group"],
            })

        manifest["splits"][split_name] = {
            "num_images": len(idxs),
            "class_object_counts": class_counts,
            "items": split_items,
        }
        non_zero = {k: v for k, v in class_counts.items() if v > 0}
        print(f"Split '{split_name}': {len(idxs)} изображений | Активные классы: {non_zero}")

    # Жёсткая самопроверка отсутствия утечек между сплитами
    for s1 in ["train", "val", "test"]:
        for s2 in ["train", "val", "test"]:
            if s1 >= s2:
                continue
            items1 = manifest["splits"][s1]["items"]
            items2 = manifest["splits"][s2]["items"]
            assert not ({x["stem"] for x in items1} & {x["stem"] for x in items2}), f"Пересечение имён {s1} и {s2}"
            assert not ({x["sha256"] for x in items1} & {x["sha256"] for x in items2}), f"Пересечение хэшей {s1} и {s2}"
            assert not ({x["scene_group"] for x in items1} & {x["scene_group"] for x in items2}), f"Пересечение групп {s1} и {s2}"

    Path("data/split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\nПроверка утечек пройдена успешно! Манифест сохранён в data/split_manifest.json.")


if __name__ == "__main__":
    main()
