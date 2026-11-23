# PPE Detection (YOLOv8) - Lab 3

## 1. Environment
- GPU: NVIDIA GeForce GTX 1650 Mobile (4 GB VRAM, CUDA 13.0)
- Stack: Python 3.12, PyTorch, Ultralytics YOLOv8, OpenCV, scikit-learn, pytest

## 2. Dataset & Anti-Leakage Split (70/15/15)
- Source: keremberke/construction-safety-object-detection (17 classes)
- Deduplication: removed 6 exact SHA-256 duplicates (398 -> 392 unique images)
- Scene-grouped split via dHash + GroupShuffleSplit (seed=42): 274 train / 59 val / 59 test
- Manifest saved in data/split_manifest.json

## 3. Validation Comparison (YOLOv8n, imgsz=640, epochs=30, batch=8)
- E1_baseline (no mosaic/mixup): P=0.7375, R=0.4161, mAP50=0.4718, mAP50-95=0.3257
- E2_improved (mosaic=1.0, mixup=0.1, cos_lr): P=0.8204, R=0.4508, mAP50=0.4470, mAP50-95=0.2656
- Winner selected by val mAP50-95: E1_baseline

## 4. Final Test Evaluation & Error Analysis
- E1_baseline on frozen test (59 images, 182 objects): P=0.8059, R=0.3439, mAP50=0.3901, mAP50-95=0.2469
- Strong classes: wheel loader (mAP50=0.9683), excavators (0.8378), hardhat (0.7515), no-hardhat (0.6990), dump truck (0.6168)
- Low macro-Recall is caused by severe class imbalance on rare/small PPE classes (gloves, no-mask, safety shoes, no-safety vest)
- Future improvements: oversampling rare PPE classes, 80-100 epochs with close_mosaic=15, imgsz=800

## 5. Post-processing & Temporal Tracking
- Class-aware NMS (conf >= 0.30, IoU <= 0.45) + TemporalViolationTracker (K >= 3 consecutive frames)
- Unit tests in tests/test_postprocess.py (pytest -v)

## 6. Reproduction
1. python src/download_data.py
2. python src/prepare_dataset.py
3. pytest -v
4. python src/train_and_eval.py
5. python -m src.infer
