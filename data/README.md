# Описание набора данных и правила разметки (PPE Detection)

## 1. Источник и состав данных
- **Базовый набор:** Construction Site Safety (`keremberke/construction-safety-object-detection`), 17 классов объектов строительной площадки и СИЗ.
- **Схема классов (ID 0..16):** `barricade`, `dumpster`, `excavators`, `gloves`, `hardhat`, `mask`, `no-hardhat`, `no-mask`, `no-safety vest`, `person`, `safety net`, `safety shoes`, `safety vest`, `dump truck`, `mini-van`, `truck`, `wheel loader`.
- **Конфиденциальность:** исходные и обработанные изображения исключены из Git (`.gitignore`), в репозитории зафиксирован только манифест разбиения `data/split_manifest.json` с контрольными суммами `SHA-256`.

## 2. Правила разметки (YOLO Bounding Boxes)
- Формат аннотаций: `<class_id> <x_center> <y_center> <width> <height>`, координаты нормированы в диапазон `[0.0, 1.0]`.
- Разметка разделяет наличие СИЗ (`hardhat`, `safety vest`, `mask`, `gloves`, `safety shoes`) и явные нарушения (`no-hardhat`, `no-safety vest`, `no-mask`) как самостоятельные классы.
- Валидация разметки (`src/prepare_dataset.py`) проверяет принадлежность `class_id` диапазону `0..16`, корректность координат и отсутствие пустых/повреждённых рамок.

## 3. Очистка и разбиение без утечек (70 / 15 / 15)
- Удалено **6 точных дубликатов** по `SHA-256` (из 398 исходных файлов оставлено **392 уникальных кадра**).
- Выполнено групповое разбиение по перцептивному хэшу сцен (`dHash`) через `GroupShuffleSplit` (`seed=42`): `train` (274) / `val` (59) / `test` (59).
