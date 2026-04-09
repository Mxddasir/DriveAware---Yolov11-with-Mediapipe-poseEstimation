# Drive Aware

**AI-powered computer vision system for detecting active mobile phone use while driving.**

Drive Aware fuses YOLO object detection with MediaPipe BlazePose body landmark estimation to identify phone use in images, video, and live webcam streams. Rather than flagging every phone that appears in frame, it requires contextual evidence — hands raised to the face, phone held in hand — to confirm active use, dramatically reducing false positives.

---

## How It Works

Detection runs as a two-stage pipeline:

1. **YOLO at high confidence (0.75)** — if a phone is found with high certainty, it is accepted immediately.
2. **If no high-confidence detection**, MediaPipe BlazePose estimates body landmarks. If a hand is within a normalised distance threshold of the face, the pose is flagged as *suspicious*.
3. **YOLO at low confidence (0.25)** — run only on suspicious poses. A phone found here, combined with the suspicious pose, is accepted.
4. Any detection can be further filtered by requiring the phone bounding box to be near a hand landmark (`require_hand_proximity` option).

```
Input frame
    │
    ▼
YOLO (high conf 0.75) ──── found ──────────────────────► Accept
    │ not found
    ▼
MediaPipe BlazePose
    │
    ├── hands NOT near face ──────────────────────────► Reject
    │
    └── hands near face (suspicious)
            │
            ▼
        YOLO (low conf 0.25) ── found ───────────────► Accept
                               not found ────────────► Reject
```

---

## Features

| Mode | Description |
|---|---|
| **Image Analysis** | Upload images or choose from sample library with search and pagination |
| **Video Analysis** | Upload a video or choose a sample clip; frame-by-frame analysis with progress bar |
| **Live Webcam** | Real-time detection stream; sustained phone use triggers a visual alert |
| **Evaluation** | Run the full pipeline against a labelled test set; compare YOLO v11, v8, and v5 side by side with accuracy, precision, recall, F1, confusion matrix, and per-stage timing |

All modes generate a **Drive Report** — a safety score, incident timeline, and key insights, with optional PDF export.

---

## Project Structure

```
drive-aware/
├── streamlit_fusion_app.py      # Main web application (all UI and pipeline orchestration)
│
├── models/
│   ├── phone_detector.py        # PhoneDetector — wraps YOLO (Ultralytics)
│   └── pose_estimator.py        # PoseEstimator — wraps MediaPipe BlazePose
│
├── logic/
│   ├── decision_engine.py       # DecisionEngine — two-stage confidence rule
│   └── proximity_analyzer.py    # ProximityAnalyzer — hand-face and hand-phone distances
│
├── processors/
│   ├── image_processor.py       # Single-image pipeline
│   └── video_processor.py       # Video/webcam pipeline
│
├── utils/
│   ├── visualization.py         # Pose skeleton and status overlay drawing
│   └── constants.py             # MediaPipe landmark indices and pose connections
│
├── config/
│   └── settings.py              # DetectionConfig dataclass with defaults
│
├── weights/
│   ├── best.pt                  # YOLO v11 weights (primary)
│   ├── weightsv8/best.pt        # YOLO v8 weights
│   └── weightsv5/best.pt        # YOLO v5 weights
│
├── pose_landmarker_full.task    # MediaPipe BlazePose model
├── Testing images/              # Labelled test set (image + .txt label file pairs)
├── Testing videos/              # Sample video clips
└── requirements.txt
```

---

## Installation

**Requirements:** Python 3.10+, a webcam (for live mode)

```bash
# 1. Clone the repository
git clone https://github.com/mxddasir/drive-aware.git
cd drive-aware

# 2. Install dependencies
pip install -r requirements.txt
```

**Model files** — two files are required that are not included in the repository due to size:

| File | Location | Source |
|---|---|---|
| YOLO v11 weights | `weights/best.pt` | Trained model (contact repo owner) |
| MediaPipe pose model | `pose_landmarker_full.task` | [MediaPipe Solutions](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) — download `pose_landmarker_full.task` |

Optional YOLO v8/v5 weights go in `weights/weightsv8/best.pt` and `weights/weightsv5/best.pt` respectively (only needed for the Evaluation comparison tab).

---

## Running the App

```bash
streamlit run streamlit_fusion_app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Configuration Parameters

These are adjustable from the sidebar in the web app:

| Parameter | Default | Description |
|---|---|---|
| `conf_high` | `0.75` | Minimum YOLO confidence to immediately accept a phone detection |
| `conf_low` | `0.25` | Fallback confidence used only when the pose is suspicious |
| `hand_face_thresh` | `0.18` | Normalised distance threshold for hand-to-face proximity |
| `hand_phone_thresh` | `0.12` | Normalised distance threshold for phone-to-hand proximity |
| `require_hand_proximity` | `False` | Reject detections where the phone is not near a hand |
| `draw_pose` | `False` | Overlay the BlazePose skeleton on output frames |

---

## Evaluation

The Evaluation tab runs the full pipeline against the `Testing images/` folder. Ground truth is determined by the presence of a `.txt` label file with the same name as the image — images with a label file are expected to show Phone Use; those without are expected to show No Phone Use.

Results include:

- **Summary** — accuracy, precision, recall, F1, TP/TN/FP/FN, and average inference time per model
- **Stage Timing** — average milliseconds for each pipeline stage (YOLO high-conf, MediaPipe, proximity, YOLO low-conf, decision)
- **Per-image Results** — searchable and paginated table comparing all models on every image
- **Errors** — incorrect predictions with expandable image previews, paginated per model

Metrics are computed as:

```
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × (Precision × Recall) / (Precision + Recall)
```

---

## Dependencies

```
streamlit
ultralytics
opencv-python-headless==4.10.0.84
torch
mediapipe>=0.10.0
reportlab          # optional — enables PDF report export
```


