# Traffic Violation Detection System

Production-structured implementation of the automated traffic violation
detection pipeline: preprocessing → detection → tracking → seven violation
checks → license plate recognition → evidence generation → storage →
analytics dashboard.

This implements the finalized project plan exactly as scoped — all seven
violations required by the problem statement are present (helmet, seatbelt,
triple riding, wrong-side driving, stop-line, red-light, illegal parking),
with seatbelt and illegal parking deliberately using the narrower, lower-effort
designs the plan specifies (an "indeterminate" fallback for seatbelt, and a
polygon+dwell-timer for parking, with no camera calibration/perspective
correction).

## Project Structure

```
traffic_violation_system/
├── config/
│   ├── config.yaml                 # the single source of truth for every setting
│   └── zones/cam_01_zones.json     # per-camera geometry (stop line, polygons, etc.)
├── data/
│   ├── models/                     # .pt weight files go here (not committed)
│   ├── training/                   # labeled crops for the two classifiers
│   ├── evidence/                   # annotated evidence images, written at runtime
│   ├── db/                         # SQLite database file, written at runtime
│   └── eval/                       # ground-truth files for scripts/evaluate.py
├── scripts/
│   ├── run_pipeline.py             # main entry point — runs the live pipeline
│   ├── train_helmet_classifier.py
│   ├── train_seatbelt_classifier.py
│   ├── calibrate_camera.py         # interactive tool to produce zone JSON files
│   └── evaluate.py                 # mAP / Precision/Recall/F1 / OCR accuracy
├── src/traffic_system/
│   ├── preprocessing/              # Stage 1
│   ├── detection/                  # Stage 2 (vehicle detector + plate detector)
│   ├── tracking/                   # Stage 3 (ByteTrack-style tracker)
│   ├── violations/                 # Stage 4 (all 7 checks + confidence routing)
│   ├── ocr/                        # Stage 5 (plate localization -> SR -> OCR -> validation)
│   ├── evidence/                   # Stage 6 (annotated frames + metadata)
│   ├── storage/                    # Stage 7 (SQLAlchemy models + repository)
│   ├── dashboard/                  # Streamlit analytics/search/review UI
│   ├── training/                   # shared dataset + training loop for classifiers
│   ├── utils/                      # config loader, types, logging, video source
│   └── pipeline.py                 # orchestrates all stages for one camera
└── tests/                          # unit tests for the geometry/rule-based logic
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # makes `traffic_system` importable as a package
```

### Models you need to provide

This repository ships **no pretrained weights** (you said you'll bring your
own dataset). Place the following files before running the pipeline:

| File | Used by | How to get it |
|---|---|---|
| `data/models/yolov10_traffic.pt` | Vehicle/person detection | Fine-tune YOLOv10 on your vehicle dataset (Ultralytics training CLI/API) |
| `data/models/plate_yolov10.pt` | Plate localization | Fine-tune a YOLOv10 model on a plate-bounding-box dataset |
| `data/models/helmet_classifier.pt` | Helmet check | Run `scripts/train_helmet_classifier.py` (see below) |
| `data/models/seatbelt_classifier.pt` | Seatbelt check | Run `scripts/train_seatbelt_classifier.py` (see below) |

Update the corresponding `*_weights` / `*_path` keys in `config/config.yaml`
if you store these files elsewhere — nothing in the code hardcodes these
paths outside of config.yaml.

## Configuration

Every threshold, path, and toggle lives in `config/config.yaml`. Common
things you'll want to change first:

- `sources` — your camera list (file path, RTSP URL, or webcam index per entry)
- `detection.weights_path`, `detection.classes` — must match your trained model
- `violations.*.confidence_threshold` — tune per your validation results
- `storage.backend` — `sqlite` for local dev, `postgresql` for production

## Per-camera zone calibration

Stop lines, red-light zones, no-parking polygons, and the traffic-light ROI
are **not hardcoded** — they're drawn once per camera using:

```bash
python scripts/calibrate_camera.py \
  --source data/sample_videos/intersection_01.mp4 \
  --camera-id cam_01 \
  --output config/zones/cam_01_zones.json
```

Then reference that output file in `config.yaml` under the matching
`sources[].zone_config` entry.

## Training the two classifiers

Helmet and seatbelt checks use a small CNN trained on cropped images you
label as `compliant` / `non_compliant`. Organize your data as:

```
data/training/helmet/compliant/*.jpg
data/training/helmet/non_compliant/*.jpg

data/training/seatbelt/compliant/*.jpg
data/training/seatbelt/non_compliant/*.jpg
```

Then run:

```bash
python scripts/train_helmet_classifier.py --config config/config.yaml
python scripts/train_seatbelt_classifier.py --config config/config.yaml
```

Hyperparameters (epochs, batch size, learning rate, image size) come from
`config.yaml`'s `training` section — edit there, not the scripts.

## Running the pipeline

```bash
# Run every camera listed in config.yaml
python scripts/run_pipeline.py --config config/config.yaml

# Run just one camera
python scripts/run_pipeline.py --config config/config.yaml --camera cam_01

# Process a bounded number of frames (useful for a quick smoke test)
python scripts/run_pipeline.py --config config/config.yaml --max-frames 200
```

Violations are written to the configured database (SQLite by default at
`data/db/violations.db`) and annotated evidence images to `data/evidence/`.

### Live signal state

The red-light/stop-line checks need to know the traffic signal's current
state. By default (`violations.red_light.signal_state_source: manual`) you
push it in programmatically:

```python
camera_pipeline.set_manual_signal_state("red")
```

Wire this to your actual signal controller feed in production. Setting
`signal_state_source: classifier` falls back to an HSV-heuristic color read
of the configured `traffic_light_roi` — adequate for a demo, explicitly not
a substitute for a trained classifier (the code logs a warning every time
this fallback path is used).

## Dashboard

```bash
streamlit run src/traffic_system/dashboard/app.py -- --config config/config.yaml
```

Provides: an overview of violation counts by type, time-series trend
charts, a searchable/exportable record table, and a human review queue for
confirming or rejecting low-confidence violations.

## Evaluation

```bash
# Detection mAP (needs an Ultralytics-format dataset YAML, separate from config.yaml)
python scripts/evaluate.py --config config/config.yaml --task detection --val-data-yaml path/to/dataset.yaml

# Violation classification Precision/Recall/F1 (needs a manually labeled ground-truth JSON)
python scripts/evaluate.py --config config/config.yaml --task violations --ground-truth data/eval/violations_gt.json

# OCR accuracy (needs a CSV of predicted_text,ground_truth_text)
python scripts/evaluate.py --config config/config.yaml --task ocr --ground-truth data/eval/plates_gt.csv
```

## Running tests

```bash
pytest tests/ -v
```

Tests cover the parts of the system that don't require trained model
weights to verify: geometry (point-in-polygon, line-crossing), IoU math,
plate format validation, and confidence routing.

## Troubleshooting

**PaddleOCR initialization is slow or fails with a download error.** This is
expected behavior in network-restricted environments — PaddleOCR downloads
its model weights on first use. The system runs OCR in an isolated worker
*process* (not just a thread) specifically because a stuck/failed import of
PaddlePaddle's native C++ extension has been observed to corrupt process
state badly enough to hang or kill the *entire* pipeline if it happened
in-process. With this isolation, a failed/slow OCR init only disables plate
recognition for that run (logged clearly) — vehicle detection, tracking,
and all seven violation checks continue working normally, and violations
are still stored with `plate_text` set to a not-read marker. To fix it
properly: pre-download PaddleOCR's model files on a machine with network
access and point the `PADDLEOCR_HOME` environment variable at them before
starting the pipeline.

## Design notes carried over from the project plan

- **Preprocessing is deliberately classical** (CLAHE + a token rain/blur
  mitigation filter) — no learned deraining/deblurring network, matching
  the plan's explicit scope decision.
- **Seatbelt only runs on `car`-class vehicles**, and returns
  `seatbelt_indeterminate` (routed straight to human review) when the
  windshield crop is too small/low-resolution to trust, rather than
  guessing or silently skipping.
- **Illegal parking uses polygon containment + a dwell timer** on top of
  existing tracker IDs — no camera calibration or perspective correction.
- **Confidence routing is one shared module** used by all seven checks
  (`violations/confidence_router.py`), not duplicated per-violation logic.
- **OCR only runs on violations that already passed the confidence
  gate** — never on every vehicle in every frame.
