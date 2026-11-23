from dataclasses import dataclass, field


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)


def compute_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Вычисляет Intersection over Union (IoU) для двух прямоугольников (x1, y1, x2, y2)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def apply_nms_and_filter(
    detections: list[Detection],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> list[Detection]:
    """Фильтрует детекции по порогу уверенности и применяет class-aware NMS."""
    filtered = [d for d in detections if d.confidence >= conf_threshold]
    filtered.sort(key=lambda d: d.confidence, reverse=True)

    kept: list[Detection] = []
    for candidate in filtered:
        suppress = False
        for existing in kept:
            if candidate.class_id == existing.class_id:
                if compute_iou(candidate.bbox, existing.bbox) > iou_threshold:
                    suppress = True
                    break
        if not suppress:
            kept.append(candidate)
    return kept


class TemporalViolationTracker:
    """Агрегатор нарушений СИЗ по последовательности кадров для подавления одиночных выбросов."""

    VIOLATION_CLASSES = {"no-hardhat", "no-mask", "no-safety vest"}

    def __init__(self, min_consecutive_frames: int = 3):
        if min_consecutive_frames < 1:
            raise ValueError("min_consecutive_frames должен быть >= 1")
        self.min_consecutive_frames = min_consecutive_frames
        self._streaks: dict[str, int] = {cls_name: 0 for cls_name in self.VIOLATION_CLASSES}

    def update(self, detections: list[Detection]) -> dict:
        """Обновляет счётчики серий нарушений на текущем кадре и возвращает подтверждённые тревоги."""
        present_violations = {
            d.class_name for d in detections if d.class_name in self.VIOLATION_CLASSES
        }

        confirmed_alerts = []
        for cls_name in sorted(self.VIOLATION_CLASSES):
            if cls_name in present_violations:
                self._streaks[cls_name] += 1
            else:
                self._streaks[cls_name] = 0

            if self._streaks[cls_name] >= self.min_consecutive_frames:
                confirmed_alerts.append(cls_name)

        return {
            "raw_violations": sorted(present_violations),
            "streaks": dict(self._streaks),
            "confirmed_alerts": confirmed_alerts,
            "is_alarm": len(confirmed_alerts) > 0,
        }
