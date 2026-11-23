import pytest
from src.postprocess import Detection, TemporalViolationTracker, apply_nms_and_filter, compute_iou


def test_compute_iou():
    box1 = (0.0, 0.0, 10.0, 10.0)
    box2 = (5.0, 0.0, 15.0, 10.0)
    # Пересечение = 50, объединение = 100 + 100 - 50 = 150 -> IoU = 1/3
    assert pytest.approx(compute_iou(box1, box2), rel=1e-5) == 1.0 / 3.0
    assert compute_iou(box1, (20.0, 20.0, 30.0, 30.0)) == 0.0


def test_apply_nms_and_filter():
    dets = [
        Detection(4, "hardhat", 0.90, (10, 10, 50, 50)),
        Detection(4, "hardhat", 0.75, (12, 12, 52, 52)),  # Сильно перекрывается с первым -> должен подавиться
        Detection(6, "no-hardhat", 0.80, (12, 12, 52, 52)),  # Другой класс -> должен остаться
        Detection(9, "person", 0.15, (0, 0, 100, 200)),  # Ниже порога conf=0.25 -> должен отфильтроваться
    ]
    result = apply_nms_and_filter(dets, conf_threshold=0.25, iou_threshold=0.45)
    assert len(result) == 2
    assert result[0].class_name == "hardhat"
    assert result[1].class_name == "no-hardhat"


def test_temporal_violation_tracker():
    tracker = TemporalViolationTracker(min_consecutive_frames=3)
    det_violation = [Detection(6, "no-hardhat", 0.85, (10, 10, 40, 40))]
    det_clean = [Detection(4, "hardhat", 0.90, (10, 10, 40, 40))]

    # Кадр 1 и 2: нарушение есть, но тревога ещё не подтверждена
    assert tracker.update(det_violation)["is_alarm"] is False
    assert tracker.update(det_violation)["is_alarm"] is False

    # Кадр 3: серия достигла 3 кадров -> тревога подтверждена
    res3 = tracker.update(det_violation)
    assert res3["is_alarm"] is True
    assert "no-hardhat" in res3["confirmed_alerts"]

    # Кадр 4: нарушение исчезло -> серия сбросилась
    res4 = tracker.update(det_clean)
    assert res4["is_alarm"] is False
    assert res4["streaks"]["no-hardhat"] == 0
