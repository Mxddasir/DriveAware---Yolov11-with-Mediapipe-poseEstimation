# Phone Usage Detection System

A professional, modular computer vision system that detects active phone use by combining YOLO-based phone detection with MediaPipe pose estimation. The system analyzes hand-face and hand-phone proximity relationships to make intelligent decisions about whether phone use is occurring.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Components Explained](#components-explained)
- [Workflow](#workflow)
- [Streamlit Web Application](#streamlit-web-application)

## 🎯 Overview

This project implements a sophisticated phone detection system that goes beyond simple object detection. It uses a **fusion approach** combining:

1. **YOLO Phone Detection**: Fast, high-accuracy phone object detection
2. **MediaPipe Pose Estimation**: Human pose landmark detection (BlazePose)
3. **Proximity Analysis**: Spatial relationship analysis between hands, face, and phone
4. **Decision Engine**: Intelligent rule-based decision making

The system is designed to reduce false positives by requiring contextual signals (hands near face, phone near hands) before declaring phone use as "active".

## 🏗️ Architecture

The project follows a clean, modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   main.py    │  │ streamlit_   │  │  Other Apps │      │
│  │  (CLI)       │  │ fusion_app.py│  │             │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
┌─────────┼──────────────────┼──────────────────┼─────────────┐
│         │                  │                  │              │
│  ┌──────▼──────────────────▼──────────────────▼──────┐     │
│  │           Processor Layer                          │     │
│  │  ┌────────────────┐  ┌────────────────┐          │     │
│  │  │ImageProcessor  │  │VideoProcessor   │          │     │
│  │  └───────┬────────┘  └────────┬────────┘          │     │
│  └──────────┼────────────────────┼────────────────────┘     │
│             │                    │                          │
│  ┌──────────▼────────────────────▼────────────────────┐     │
│  │              Logic Layer                           │     │
│  │  ┌────────────────┐  ┌────────────────┐          │     │
│  │  │DecisionEngine  │  │ProximityAnalyzer│          │     │
│  │  └────────────────┘  └────────────────┘          │     │
│  └────────────────────────────────────────────────────┘     │
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │              Model Layer                           │     │
│  │  ┌────────────────┐  ┌────────────────┐          │     │
│  │  │PhoneDetector   │  │PoseEstimator   │          │     │
│  │  │  (YOLO)        │  │  (MediaPipe)   │          │     │
│  │  └────────────────┘  └────────────────┘          │     │
│  └────────────────────────────────────────────────────┘     │
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │              Utility Layer                         │     │
│  │  ┌────────────────┐  ┌────────────────┐          │     │
│  │  │Visualizer      │  │Constants       │          │     │
│  │  └────────────────┘  └────────────────┘          │     │
│  └────────────────────────────────────────────────────┘     │
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │              Config Layer                          │     │
│  │  ┌────────────────────────────────────────────┐   │     │
│  │  │         DetectionConfig                    │   │     │
│  │  └────────────────────────────────────────────┘   │     │
│  └────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
project/
├── main.py                      # CLI entry point with argparse
├── streamlit_fusion_app.py      # Streamlit web application
├── fusion_mediapipe_tasks.py    # Legacy monolithic script (backward compatibility)
│
├── models/                      # ML Model Wrappers
│   ├── __init__.py
│   ├── phone_detector.py        # PhoneDetector class (YOLO wrapper)
│   └── pose_estimator.py        # PoseEstimator class (MediaPipe wrapper)
│
├── logic/                       # Business Logic
│   ├── __init__.py
│   ├── decision_engine.py       # DecisionEngine: Active/Not Active logic
│   └── proximity_analyzer.py    # ProximityAnalyzer: Hand-face, hand-phone analysis
│
├── utils/                       # Utilities
│   ├── __init__.py
│   ├── visualization.py         # Visualizer: Drawing functions
│   └── constants.py             # Constants: Landmark indices, pose connections
│
├── processors/                  # Processing Pipelines
│   ├── __init__.py
│   ├── video_processor.py       # VideoProcessor: Video/webcam processing
│   └── image_processor.py       # ImageProcessor: Single image processing
│
└── config/                      # Configuration
    ├── __init__.py
    └── settings.py              # DetectionConfig: Configuration dataclass
```

## 🔄 How It Works

### High-Level Flow

1. **Input**: Image or video frame (BGR format)
2. **Phone Detection**: YOLO model detects phones at high confidence (default: 0.75)
3. **Conditional Pose Estimation**: 
   - Only runs if: pose visualization requested, hand proximity filtering enabled, OR high-confidence detection failed
4. **Proximity Analysis**:
   - **Hand-Face**: Checks if hands are near face landmarks (nose, mouth, ears)
   - **Hand-Phone**: Checks if detected phone boxes are near hand landmarks
5. **Decision Making**:
   - **High-confidence phone found**: Accept (with optional hand proximity filter)
   - **Low-confidence phone + suspicious pose**: Accept (with optional hand proximity filter)
   - **Otherwise**: Reject
6. **Visualization**: Annotate frame with bounding boxes, pose skeleton, status overlay
7. **Output**: Annotated image/video with detection results

### Detection Strategy

The system uses a **two-stage confidence approach**:

```
┌─────────────────────────────────────────────────┐
│         Input Frame                             │
└───────────────┬─────────────────────────────────┘
                │
                ▼
    ┌───────────────────────┐
    │ YOLO High Conf (0.75) │
    └───────────┬───────────┘
                │
        ┌───────┴───────┐
        │                │
    Found?            Not Found?
        │                │
        ▼                ▼
   ┌────────┐    ┌─────────────────┐
   │Accept  │    │ Check Suspicious│
   │(with   │    │ Pose (hands near│
   │filter) │    │ face?)          │
   └────────┘    └────────┬────────┘
                          │
                    ┌─────┴─────┐
                    │            │
                Yes?          No?
                    │            │
                    ▼            ▼
            ┌─────────────┐  ┌────────┐
            │YOLO Low Conf│  │ Reject │
            │(0.25)       │  └────────┘
            └──────┬──────┘
                   │
            ┌──────┴──────┐
            │             │
        Found?        Not Found?
            │             │
            ▼             ▼
       ┌────────┐    ┌────────┐
       │Accept  │    │ Reject │
       │(with   │    └────────┘
       │filter) │
       └────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for YOLO)
- Webcam (for live detection)

### Dependencies

```bash
pip install ultralytics opencv-python mediapipe numpy streamlit pillow reportlab
```

### Model Files Required

1. **YOLO Phone Model**: Place your trained YOLO model at `weights/best.pt`
2. **MediaPipe Pose Model**: Download `pose_landmarker_full.task` from [MediaPipe Solutions](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)

## 🚀 Usage

### Command Line Interface

```bash
# Webcam detection
python main.py --source 0

# Video file
python main.py --source video.mp4 --save output.mp4

# Single image
python main.py --source image.jpg --save annotated.jpg --draw_pose

# With custom thresholds
python main.py --source 0 \
    --conf_high 0.8 \
    --conf_low 0.3 \
    --hand_face_thresh 0.15 \
    --require_hand_proximity
```

### Streamlit Web Application

```bash
streamlit run streamlit_fusion_app.py
```

Features:
- **Upload Images**: Test detection on uploaded images
- **Upload Video**: Analyze video files with progress tracking
- **Live Webcam**: Real-time detection with incident tracking
- **Drive Reports**: Professional PDF reports with statistics and recommendations

### Programmatic Usage

```python
from models.phone_detector import PhoneDetector
from processors.image_processor import ImageProcessor

# Initialize detector
detector = PhoneDetector("weights/best.pt")

# Create processor
processor = ImageProcessor(detector)

# Process image
annotated, is_active = processor.process(
    image_path="test.jpg",
    pose_task_path="pose_landmarker_full.task",
    conf_high=0.75,
    conf_low=0.25,
    hand_face_thresh=0.18,
    hand_phone_thresh=0.12,
    require_hand_proximity=False,
    draw_pose=True
)
```

## 🧩 Components Explained

### Models Layer (`models/`)

#### `PhoneDetector`
- **Purpose**: Wraps YOLO model for phone detection
- **Key Methods**:
  - `detect(frame, confidence)`: Returns detection results, boxes, confidences
- **Usage**: Load once, use for multiple frames

#### `PoseEstimator`
- **Purpose**: Wraps MediaPipe pose landmarker
- **Key Methods**:
  - `detect_image(frame_rgb)`: Single image pose detection
  - `detect_video(frame_rgb, timestamp_ms)`: Video frame pose detection
  - `close()`: Resource cleanup
- **Context Manager**: Supports `with` statement for automatic cleanup

### Logic Layer (`logic/`)

#### `ProximityAnalyzer`
- **Purpose**: Analyzes spatial relationships
- **Key Methods**:
  - `hands_near_face(pose_result, threshold)`: Checks if hands are near face
  - `get_hand_points_px(pose_result, width, height)`: Extracts hand landmarks
  - `phone_close_to_hand(boxes, hand_points, threshold)`: Checks phone-hand proximity
- **All Static**: Stateless, pure functions

#### `DecisionEngine`
- **Purpose**: Makes final active/not-active decisions
- **Key Methods**:
  - `decide_phone_use(...)`: Returns (is_active, reason_string)
- **Strategy**: Two-stage confidence approach with proximity filtering

### Processors Layer (`processors/`)

#### `ImageProcessor`
- **Purpose**: Complete pipeline for single images
- **Workflow**:
  1. Load image
  2. Run phone detection
  3. Conditionally run pose estimation
  4. Analyze proximity
  5. Make decision
  6. Annotate and return

#### `VideoProcessor`
- **Purpose**: Complete pipeline for video/webcam
- **Features**:
  - Frame-by-frame processing
  - FPS calculation
  - Optional video saving
  - Lazy pose estimator initialization

### Utils Layer (`utils/`)

#### `Visualizer`
- **Purpose**: Drawing and annotation functions
- **Methods**:
  - `draw_pose_landmarks()`: Draws pose skeleton
  - `annotate_status()`: Draws status overlay bar

#### `constants.py`
- **Purpose**: Centralized constants
- **Contents**:
  - MediaPipe landmark indices (NOSE, L_WRIST, etc.)
  - Pose skeleton connections for visualization

### Config Layer (`config/`)

#### `DetectionConfig`
- **Purpose**: Centralized configuration management
- **Features**:
  - Dataclass with defaults
  - Automatic path resolution
  - Type hints for all parameters

## 🔄 Workflow: How Components Work Together

### Example: Processing a Single Image

```python
# 1. Initialize models
detector = PhoneDetector("weights/best.pt")  # Loads YOLO model
processor = ImageProcessor(detector)         # Wraps detector

# 2. Process image
annotated, active = processor.process(...)

# Inside processor.process():
#   2a. detector.detect() → PhoneDetector runs YOLO
#   2b. PoseEstimator() → MediaPipe pose detection (if needed)
#   2c. ProximityAnalyzer.hands_near_face() → Analyze hand-face distance
#   2d. ProximityAnalyzer.phone_close_to_hand() → Analyze phone-hand distance
#   2e. DecisionEngine.decide_phone_use() → Make final decision
#   2f. Visualizer.draw_pose_landmarks() → Draw skeleton (if requested)
#   2g. Visualizer.annotate_status() → Draw status overlay
```

### Data Flow Diagram

```
Input Frame (BGR)
    │
    ├─→ PhoneDetector.detect()
    │       │
    │       └─→ YOLO Model → (boxes, confidences)
    │
    ├─→ [If needed] PoseEstimator.detect_image()
    │       │
    │       └─→ MediaPipe → (pose_landmarks)
    │               │
    │               ├─→ ProximityAnalyzer.hands_near_face()
    │               │       └─→ (suspicious, distance)
    │               │
    │               └─→ ProximityAnalyzer.get_hand_points_px()
    │                       └─→ (hand_points_list)
    │
    ├─→ ProximityAnalyzer.phone_close_to_hand()
    │       └─→ (hand_ok, distance)
    │
    ├─→ DecisionEngine.decide_phone_use()
    │       └─→ (is_active, reason)
    │
    └─→ Visualizer.annotate_status()
            └─→ Annotated Frame (BGR)
```

## 🌐 Streamlit Web Application

The Streamlit app (`streamlit_fusion_app.py`) provides a user-friendly interface with:

### Features

1. **Image Upload Mode**
   - Upload single or multiple images
   - Batch processing with results display
   - Annotated output visualization

2. **Video Upload Mode**
   - Upload video files (mp4, mov, avi, etc.)
   - Progress tracking during analysis
   - Frame-by-frame preview
   - **Drive Report Generation**: Professional PDF reports with:
     - Safety scores
     - Incident timeline
     - Phone use statistics
     - Recommendations

3. **Live Webcam Mode**
   - Real-time detection
   - Consecutive frame tracking (5+ frames = alert)
   - Audio alerts (macOS: text-to-speech)
   - **Drive Report**: Generated at session end

### Report Features

- **Session Information**: Date, duration, start time
- **Safety Score**: 0-100 based on phone use percentage
- **Phone Use Summary**: Total time, percentage, incident count
- **Incident Timeline**: Visual bar chart of incidents
- **Incident Details**: Timestamped list of each incident
- **Key Insights**: Contextual analysis based on usage patterns
- **Recommendations**: Actionable advice for improvement
- **PDF Export**: Professional formatted reports

## 🎓 Key Design Principles

1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Reusability**: Classes can be imported and used independently
3. **Testability**: Pure functions and isolated classes are easy to test
4. **Maintainability**: Clear structure makes it easy to find and modify code
5. **Extensibility**: New features can be added without modifying existing code
6. **Documentation**: Comprehensive docstrings and type hints throughout

## 🔧 Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `conf_high` | 0.75 | High confidence threshold for phone detection |
| `conf_low` | 0.25 | Low confidence threshold (fallback) |
| `hand_face_thresh` | 0.18 | Normalized distance for hand-face proximity |
| `hand_phone_thresh` | 0.12 | Normalized distance for phone-hand proximity |
| `require_hand_proximity` | False | Filter detections without hand proximity |
| `draw_pose` | False | Draw pose skeleton overlay |

## 📝 Notes

- **Performance**: Pose estimation is computationally expensive, so it's only run when necessary
- **Accuracy**: The fusion approach significantly reduces false positives compared to phone detection alone
- **Thresholds**: May need tuning based on your specific use case and camera setup
- **Models**: Ensure you have the correct model files in the expected locations

## 🤝 Contributing

When adding new features:
1. Follow the existing modular structure
2. Add comprehensive docstrings
3. Update this README if adding new components
4. Maintain backward compatibility with the legacy `fusion_mediapipe_tasks.py`

## 📄 License

[Add your license here]

---

**Built with**: Python, YOLO (Ultralytics), MediaPipe, OpenCV, Streamlit