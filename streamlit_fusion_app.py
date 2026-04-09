import math
import os
import platform
import tempfile
import time
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from models.phone_detector import PhoneDetector
from models.pose_estimator import PoseEstimator
from logic.proximity_analyzer import ProximityAnalyzer
from logic.decision_engine import DecisionEngine
from utils.visualization import Visualizer
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.image import Image as MpImage, ImageFormat as MpImageFormat

# importing reportlab for pdf generation 
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# Sample images directory (relative to this file)
SAMPLE_IMAGES_DIR = Path(__file__).resolve().parent / "Testing images"
SAMPLE_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# weight paths for Evaluation tab (v11 = main weights, v8 = weightsv8, v5 = weightsv5)
EVAL_MODELS = {
    "YOLO v11": "weights/best.pt",
    "YOLO v8": "weights/weightsv8/best.pt",
    "YOLO v5": "weights/weightsv5/best.pt",
}

SAMPLE_VIDEOS_DIR = Path(__file__).resolve().parent / "Testing videos"
SAMPLE_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _get_sample_image_paths() -> List[Path]:
    """Return sorted list of image paths in SAMPLE_IMAGES_DIR, or empty if missing/empty."""
    if not SAMPLE_IMAGES_DIR.is_dir():
        return []
    paths = [
        p for p in SAMPLE_IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SAMPLE_IMAGE_EXTENSIONS
    ]
    return sorted(paths, key=lambda p: p.name)


def _get_sample_images_with_labels() -> List[Tuple[Path, str]]:
    """Return list of (path, friendly_name) for sample images. Names are 'Photo 1', 'Photo 2', etc."""
    paths = _get_sample_image_paths()
    return [(p, f"Photo {i + 1}") for i, p in enumerate(paths)]


def _load_thumbnail(path: Path, max_width: int = 160) -> np.ndarray:
    """Load image from path and resize to thumbnail (RGB for display). Returns None on failure."""
    frame = cv2.imread(str(path))
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _load_video_thumbnail(path: Path, max_width: int = 160) -> np.ndarray:
    """Read first frame of a video and return as an RGB thumbnail. Returns None on failure."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        frame = cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _get_sample_video_paths() -> List[Path]:
    """Return sorted list of video paths in SAMPLE_VIDEOS_DIR, or empty if missing/empty."""
    if not SAMPLE_VIDEOS_DIR.is_dir():
        return []
    paths = [
        p for p in SAMPLE_VIDEOS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SAMPLE_VIDEO_EXTENSIONS
    ]
    return sorted(paths, key=lambda p: p.name)


def _get_sample_videos_with_labels() -> List[Tuple[Path, str]]:
    """Return list of (path, friendly_name) for sample videos. Names are 'Video 1', 'Video 2', etc."""
    paths = _get_sample_video_paths()
    return [(p, f"Video {i + 1}") for i, p in enumerate(paths)]


class _SampleVideo:
    # wraps a Path so it looks like an uploaded file to _analyze_video_file_streamlit
    # (which expects .name and .getvalue())

    def __init__(self, path: Path):
        self.name = path.name
        self._path = path

    def getvalue(self) -> bytes:
        return self._path.read_bytes()


# single image pipeline — takes a BGR numpy array, runs the full detection pipeline,
# returns (annotated_bgr, is_active, timing_dict)

def _process_single_image_bgr(
    frame_bgr: np.ndarray,
    phone_model: PhoneDetector,
    pose_task_path: str,
    conf_high: float,
    conf_low: float,
    hand_face_thresh: float,
    hand_phone_thresh: float,
    require_hand_proximity: bool,
    draw_pose: bool,
):
    H, W = frame_bgr.shape[:2]

    # first pass: YOLO at high confidence
    _t = time.time()
    det_high, boxes_high, _ = phone_model.detect(frame_bgr, conf_high)
    yolo_high_ms = (time.time() - _t) * 1000
    found_high = boxes_high.shape[0] > 0

    # defaults — overwritten below if pose runs
    suspicious = False
    face_d = 999.0
    hand_pts_px: List[tuple[int, int]] = []
    hand_ok = False
    hand_d = 999.0
    used_conf = conf_high
    det_to_plot = det_high

    # pose only runs when needed: drawing requested, hand filter on, or YOLO missed
    need_pose = draw_pose or require_hand_proximity or (not found_high)
    pose_result = None
    pose_ms = 0.0
    pose_ran = False
    proximity_ms = 0.0
    yolo_low_ms = 0.0

    if need_pose:
        # run MediaPipe BlazePose to get body landmarks
        _t = time.time()
        landmarker = PoseEstimator(pose_task_path, vision.RunningMode.IMAGE)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pose_result = landmarker.detect_image(frame_rgb)
        landmarker.close()
        pose_ms = (time.time() - _t) * 1000
        pose_ran = True

        # check if any hand landmark is close to the face
        _t = time.time()
        suspicious, _best_hand_norm, face_d = ProximityAnalyzer.hands_near_face(
            pose_result, hand_face_thresh
        )
        hand_pts_px = ProximityAnalyzer.get_hand_points_px(pose_result, W, H)
        proximity_ms += (time.time() - _t) * 1000

    if found_high:
        # phone found with high confidence — optionally check it's near a hand
        if require_hand_proximity and hand_pts_px:
            _t = time.time()
            hand_ok, hand_d = ProximityAnalyzer.phone_close_to_hand(
                boxes_high, hand_pts_px, W, H, hand_phone_thresh
            )
            proximity_ms += (time.time() - _t) * 1000
        det_to_plot = det_high
        used_conf = conf_high
        found_low = False
    else:
        # YOLO missed at high conf — try again at low conf only if pose looks suspicious
        if suspicious:
            _t = time.time()
            det_low, boxes_low, _ = phone_model.detect(frame_bgr, conf_low)
            yolo_low_ms = (time.time() - _t) * 1000
            found_low = boxes_low.shape[0] > 0
            det_to_plot = det_low
            used_conf = conf_low

            if require_hand_proximity and hand_pts_px and found_low:
                _t = time.time()
                hand_ok, hand_d = ProximityAnalyzer.phone_close_to_hand(
                    boxes_low, hand_pts_px, W, H, hand_phone_thresh
                )
                proximity_ms += (time.time() - _t) * 1000
        else:
            found_low = False

    # final decision: active or not
    _t = time.time()
    active, reason = DecisionEngine.decide_phone_use(
        found_high=found_high,
        found_low=found_low,
        suspicious=suspicious,
        hand_ok=hand_ok,
        require_hand_proximity=require_hand_proximity,
    )
    decision_ms = (time.time() - _t) * 1000

    # draw everything onto the frame
    annotated = det_to_plot.plot()

    if draw_pose and pose_result is not None:
        annotated = Visualizer.draw_pose_landmarks(annotated, pose_result, draw_connections=True)

    annotated = Visualizer.annotate_status(
        annotated,
        active=active,
        suspicious=suspicious,
        used_conf=used_conf,
        reason=reason,
        face_dist=face_d,
        hand_dist=hand_d,
    )

    timing = {
        "yolo_high_ms": yolo_high_ms,
        "pose_ms": pose_ms,
        "pose_ran": pose_ran,
        "proximity_ms": proximity_ms,
        "yolo_low_ms": yolo_low_ms,
        "decision_ms": decision_ms,
        "total_ms": yolo_high_ms + pose_ms + proximity_ms + yolo_low_ms + decision_ms,
    }

    return annotated, active, timing


def _play_alert_sound():
    # sound output varies by OS; failures are silently ignored
    system = platform.system()
    try:
        if system == "Darwin":
            os.system('say "phone active" &')
        elif system == "Windows":
            import winsound
            winsound.MessageBeep()
        else:
            os.system("printf '\\a'")
    except Exception:
        pass


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    if minutes == 0:
        return f"{sec} seconds"
    label = "minute" if minutes == 1 else "minutes"
    return f"{minutes} {label} {sec:02d} seconds"


def _prepare_report_data(
    session_start: datetime,
    session_end: datetime,
    incidents: List[Tuple[datetime, datetime]],
):
    # calculate totals and safety score from the incident list

    total_duration_s = (session_end - session_start).total_seconds()
    phone_use_s = sum((end - start).total_seconds() for start, end in incidents if end > start)
    phone_use_pct = (phone_use_s / total_duration_s * 100) if total_duration_s > 0 else 0.0

    # 100 = no phone use; score drops by 1.2x the usage percentage, minimum 0
    safety_score = int(max(0, 100 - phone_use_pct * 1.2))

    date_str = session_start.strftime("%B %d, %Y")
    start_time_str = session_start.strftime("%H:%M:%S")
    duration_str = _format_duration(total_duration_s)

    # build a list of incidents with timestamps and durations
    incident_items = []
    for idx, (start, end) in enumerate(incidents, start=1):
        dur_s = max(0, (end - start).total_seconds())
        item = {
            "index": idx,
            "start_dt": start,
            "end_dt": end,
            "time_str": start.strftime("%H:%M:%S"),
            "duration_s": dur_s,
            "duration_str": _format_duration(dur_s),
        }
        incident_items.append(item)

    # text bar chart for the markdown report — length scaled to the longest incident
    timeline_bars = ""
    timeline_segments = []
    if incident_items:
        max_inc_s = max(item["duration_s"] for item in incident_items)
        for item in incident_items:
            dur_s = item["duration_s"]
            bar_len = int(20 * (dur_s / max_inc_s)) if max_inc_s > 0 else 1
            timeline_bars += (
                f"{item['index']}: " + "█" * max(1, bar_len) + f" ({item['duration_str']})\n"
            )
            start_offset_s = (item["start_dt"] - session_start).total_seconds()
            end_offset_s = (item["end_dt"] - session_start).total_seconds()
            timeline_segments.append({
                "start_s": max(0.0, start_offset_s),
                "end_s": max(0.0, end_offset_s),
            })
    else:
        timeline_bars = "No incidents recorded."

    # choose an insight message based on severity of phone use
    if phone_use_s == 0:
        insights = (
            "No phone use was detected during this session. This is the safest possible behaviour. "
            "Maintaining this habit significantly reduces your risk of distraction-related incidents."
        )
    elif phone_use_pct < 5:
        insights = (
            "Phone use during this session was low, but not zero. Even short periods of distraction "
            "can be enough to miss critical events in front of you. Aim to eliminate active phone use "
            "entirely while the vehicle is in motion."
        )
    elif phone_use_pct < 20:
        insights = (
            "Phone use occupied a noticeable portion of this session. This level of distraction "
            "meaningfully increases your risk on the road. Treat each incident as a signal to adjust "
            "your habits before they become routine."
        )
    else:
        insights = (
            "Phone use was frequent and sustained during this session. This pattern is unsafe and "
            "would be considered high-risk in professional driving contexts. You should treat this "
            "as a serious warning and take concrete steps to remove the phone from your driving routine."
        )

    # structured list used in the PDF; plain markdown version used in the Streamlit report
    recommendations_list = [
        {
            "title": "Pattern Analysis",
            "description": "Most incidents occurred at traffic stops. Phone use at stoplights remains unsafe and is illegal in many jurisdictions."
        },
        {
            "title": "Hands-Free Solution",
            "description": "Consider using a dashboard mount for your phone if navigation is required during drives."
        },
        {
            "title": "Enable Do Not Disturb",
            "description": "Activate 'Do Not Disturb While Driving' mode to automatically silence notifications and reduce temptation."
        },
        {
            "title": "Goal Setting",
            "description": "Set a goal to achieve zero phone use incidents in your next five drives. Consistent improvement reduces risk."
        }
    ]

    # markdown version of recommendations for streamlit display
    recommendations = (
        "- **Before driving**: Configure your phone with driving focus or Do Not Disturb modes so "
        "notifications do not tempt you.\n"
        "- **Physical setup**: If navigation is required, use a fixed, eye-level mount and avoid "
        "holding the phone in your hand.\n"
        "- **During stops**: Avoid the habit of checking messages at every stoplight; this easily "
        "spills over into motion.\n"
        "- **After each drive**: Review where incidents cluster in time and ask what triggered "
        "each one (boredom, notifications, calls). Remove those triggers where possible."
    )

    report_data = {
        "date_str": date_str,
        "start_time_str": start_time_str,
        "duration_str": duration_str,
        "total_duration_s": total_duration_s,
        "phone_use_s": phone_use_s,
        "phone_use_pct": phone_use_pct,
        "safety_score": safety_score,
        "incidents": incident_items,
        "timeline_bars": timeline_bars,
        "timeline_segments": timeline_segments,
        "insights": insights,
        "recommendations": recommendations,
        "recommendations_list": recommendations_list,
    }
    return report_data


# builds a formatted PDF report using reportlab

def _build_pdf_from_report_data(report_data: dict) -> bytes:
    # all drawing is done with reportlab primitives; returns the PDF as bytes

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # helper to draw section headings
    def section_title(y, text, accent_bar=False):
        if accent_bar:
            bar_width = 4
            bar_height = 16
            c.setFillColor(colors.HexColor("#66B2B2"))
            c.rect(40, y - bar_height, bar_width, bar_height, fill=1, stroke=0)
            text_x = 40 + bar_width + 8
        else:
            text_x = 40
        c.setFillColor(colors.HexColor("#003366"))
        c.setFont("Helvetica-Bold", 12)
        c.drawString(text_x, y, text.upper())
        c.setStrokeColor(colors.HexColor("#003366"))
        c.setLineWidth(1)
        c.line(40, y - 4, width - 40, y - 4)
        return y - 20

    # title
    c.setFillColor(colors.HexColor("#003366"))
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2, height - 40, "DRIVER SAFETY REPORT")
    c.setLineWidth(2)
    c.line(40, height - 50, width - 40, height - 50)

    y = height - 80

    # session info section
    y = section_title(y, "Session information")

    c.setFont("Helvetica", 10)
    box_height = 40
    box_width = (width - 80) / 3 - 10
    x0 = 40
    fields = [
        ("DATE", report_data["date_str"]),
        ("START TIME", report_data["start_time_str"]),
        ("DURATION", report_data["duration_str"]),
    ]
    for i, (label, value) in enumerate(fields):
        x = x0 + i * (box_width + 10)
        c.setFillColor(colors.whitesmoke)
        c.roundRect(x, y - box_height, box_width, box_height, 4, stroke=0, fill=1)
        c.setFillColor(colors.gray)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(x + 8, y - 15, label)
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 10)
        c.drawString(x + 8, y - 30, value)

    y -= box_height + 25

    # safety score section
    y = section_title(y, "Safety score")

    score = report_data["safety_score"]
    phone_pct = report_data["phone_use_pct"]
    safe_pct = max(0.0, 100.0 - phone_pct)
    incidents = report_data["incidents"]

    # big score box on the left
    score_box_w = 130
    score_box_h = 70
    x_left = 40
    c.setFillColor(colors.HexColor("#FFF4E5"))
    c.roundRect(x_left, y - score_box_h, score_box_w, score_box_h, 4, stroke=0, fill=1)

    c.setFillColor(colors.HexColor("#FF8C00"))
    c.setFont("Helvetica-Bold", 20)
    c.drawString(x_left + 15, y - 25, f"{score}/100")

    if score >= 90:
        label = "EXCELLENT"
    elif score >= 80:
        label = "GOOD"
    else:
        label = "NEEDS IMPROVEMENT"

    c.setFont("Helvetica-Bold", 10)
    c.drawString(x_left + 15, y - 45, label)

    # breakdown box on the right
    x_right = x_left + score_box_w + 15
    box_w = width - 40 - x_right
    box_h = score_box_h
    c.setFillColor(colors.HexColor("#F5F5F5"))
    c.roundRect(x_right, y - box_h, box_w, box_h, 4, stroke=0, fill=1)

    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x_right + 10, y - 18, "Score breakdown")
    c.setFont("Helvetica", 9)
    c.drawString(x_right + 10, y - 32, f"Safe driving time: {safe_pct:.1f}% of session")
    c.drawString(x_right + 10, y - 44, f"Phone use time: {phone_pct:.1f}% of session")
    c.drawString(x_right + 10, y - 56, f"Total incidents: {len(incidents)}")

    y -= score_box_h + 25

    # phone use warning section
    y = section_title(y, "Phone use detected")

    warning_h = 50
    c.setFillColor(colors.HexColor("#FDECEA"))
    c.roundRect(40, y - warning_h, width - 80, warning_h, 4, stroke=0, fill=1)
    c.setFillColor(colors.HexColor("#B71C1C"))
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y - 18, "Warning: phone use was detected during this session.")
    c.setFont("Helvetica", 9)
    c.drawString(
        50, y - 32,
        f"Total phone use: {_format_duration(report_data['phone_use_s'])} "
        f"({phone_pct:.1f}% of session), incidents: {len(incidents)}.",
    )

    y -= warning_h + 25

    # timeline bar
    y = section_title(y, "Incident timeline")

    bar_y = y - 25
    bar_h = 14
    left = 60
    right = width - 60

    # green background bar
    c.setFillColor(colors.HexColor("#E0F2F1"))
    c.roundRect(left, bar_y, right - left, bar_h, 7, stroke=0, fill=1)

    # overlay red segments where incidents happened
    total_s = max(report_data["total_duration_s"], 1.0)
    c.setFillColor(colors.HexColor("#FF7043"))
    for seg in report_data["timeline_segments"]:
        start_x = left + (seg["start_s"] / total_s) * (right - left)
        end_x = left + (seg["end_s"] / total_s) * (right - left)
        if end_x <= start_x:
            continue
        c.roundRect(start_x, bar_y, end_x - start_x, bar_h, 7, stroke=0, fill=1)

    c.setFillColor(colors.gray)
    c.setFont("Helvetica", 8)
    c.drawString(left, bar_y - 10, "0 min")
    c.drawRightString(right, bar_y - 10, f"{int(total_s // 60)} min")

    y = bar_y - 30

    # individual incident details
    y = section_title(y, "Incident details")

    box_h = 55
    for item in incidents:
        # new page if running out of space
        if y - box_h < 60:
            c.showPage()
            width, height = A4
            y = height - 60
            y = section_title(y, "Incident details (continued)")

        c.setFillColor(colors.HexColor("#FFF8E1"))
        c.roundRect(40, y - box_h, width - 80, box_h, 4, stroke=0, fill=1)

        c.setFillColor(colors.HexColor("#FF8F00"))
        c.setFont("Helvetica-Bold", 10)
        c.drawString(50, y - 18, f"Incident {item['index']}")

        c.setFillColor(colors.black)
        c.setFont("Helvetica", 9)
        c.drawString(50, y - 30, f"Time: {item['time_str']}")
        c.drawString(200, y - 30, f"Duration: {item['duration_str']}")
        c.drawString(50, y - 42, "Status: Active phone use detected")

        y -= box_h + 10

    # performance trends placeholder
    if y < 160:
        c.showPage()
        width, height = A4
        y = height - 60
    y = section_title(y, "Performance trends")
    c.setFont("Helvetica", 9)
    c.drawString(50, y - 16, "Comparison vs last week: trend data is not yet available in this prototype.")
    c.drawString(50, y - 30, "Comparison vs your average: trend data is not yet available in this prototype.")

    y -= 60

    # recommendations section
    if y < 200:
        c.showPage()
        width, height = A4
        y = height - 60
    y = section_title(y, "Recommendations", accent_bar=True)

    recommendations_list = report_data.get("recommendations_list", [])
    box_spacing = 12
    accent_bar_width = 4

    for idx, rec in enumerate(recommendations_list, start=1):
        if y < 80:
            c.showPage()
            width, height = A4
            y = height - 60

        # work out how tall the box needs to be
        desc_text = rec["description"]
        desc_lines = max(1, (len(desc_text) + 75) // 75)
        box_height = 20 + 12 + (desc_lines * 11) + 10

        box_x = 40
        box_y = y - box_height
        box_width = width - 80

        # light blue background
        c.setFillColor(colors.HexColor("#E0F2F7"))
        c.roundRect(box_x, box_y, box_width, box_height, 4, stroke=0, fill=1)

        # accent bar on the left
        c.setFillColor(colors.HexColor("#66B2B2"))
        c.rect(box_x, box_y, accent_bar_width, box_height, fill=1, stroke=0)

        text_x = box_x + accent_bar_width + 12
        text_y = y - 18

        # title
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(text_x, text_y, f"{idx}. {rec['title']}")

        # description with basic word wrapping
        desc_y = text_y - 14
        text_obj = c.beginText()
        text_obj.setTextOrigin(text_x, desc_y)
        text_obj.setFont("Helvetica", 9)
        words = desc_text.split()
        line = ""
        for word in words:
            test_line = line + (" " if line else "") + word
            if len(test_line) * 5.5 > box_width - (text_x - box_x) - 20:
                if line:
                    text_obj.textLine(line)
                line = word
            else:
                line = test_line
        if line:
            text_obj.textLine(line)
        c.drawText(text_obj)

        y = box_y - box_spacing

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


# renders the drive report as markdown into a Streamlit placeholder

def _render_drive_report(
    report_placeholder,
    session_start: datetime,
    session_end: datetime,
    incidents: List[Tuple[datetime, datetime]],
):
    report_data = _prepare_report_data(session_start, session_end, incidents)

    # build the markdown string for streamlit
    report_md = f"""
### Drive Report

**Date:** {report_data['date_str']}  
**Session duration:** {report_data['duration_str']}  

**Safety score:** {report_data['safety_score']}/100  

---

### Phone use summary

**Total phone use:** {_format_duration(report_data['phone_use_s'])} ({report_data['phone_use_pct']:.1f}% of session)  
**Number of incidents:** {len(report_data['incidents'])} event{'s' if len(report_data['incidents']) != 1 else ''}  

---

### Timeline of phone-use incidents

```text
{report_data['timeline_bars'].strip()}
```

---

### Incident details

{chr(10).join(
    f"{item['index']}. {item['time_str']} - {item['duration_str']} - Active use"
    for item in report_data['incidents']
) if report_data['incidents'] else "No phone-use incidents detected."}

---

### Key insights

{report_data['insights']}

---

### Recommendations

{report_data['recommendations']}
"""

    report_placeholder.markdown(report_md)
    return report_data


# reads frames from the webcam and runs detection until the time limit is reached

def _run_live_webcam_streamlit(
    phone_model: YOLO,
    pose_task_path: str,
    conf_high: float,
    conf_low: float,
    hand_face_thresh: float,
    hand_phone_thresh: float,
    require_hand_proximity: bool,
    draw_pose: bool,
    max_seconds: int,
    frame_placeholder,
    info_placeholder,
    report_placeholder,
):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        info_placeholder.error("Could not open webcam (device 0).")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0

    landmarker = None
    running_mode = vision.RunningMode.VIDEO

    frame_idx = 0
    t0 = time.time()
    session_start = datetime.now()

    consecutive_active = 0
    alert_triggered = False

    # tracking for the report
    active_threshold = 5
    incident_active = False
    incident_start_time: datetime | None = None
    last_active_time: datetime | None = None
    incidents: List[Tuple[datetime, datetime]] = []

    try:
        while True:
            now = time.time()
            if now - t0 > max_seconds:
                info_placeholder.info("Live session finished (time limit reached).")
                break

            ok, frame_bgr = cap.read()
            if not ok:
                info_placeholder.warning("Webcam frame grab failed; stopping.")
                break

            H, W = frame_bgr.shape[:2]

            # high conf yolo pass
            det_high, boxes_high, _confs_high = phone_model.detect(frame_bgr, conf_high)
            found_high = boxes_high.shape[0] > 0

            suspicious = False
            face_d = 999.0
            hand_pts_px: List[tuple[int, int]] = []
            hand_ok = False
            hand_d = 999.0
            used_conf = conf_high
            det_to_plot = det_high

            need_pose = draw_pose or require_hand_proximity or (not found_high)
            pose_result = None

            if need_pose:
                if landmarker is None:
                    landmarker = PoseEstimator(pose_task_path, running_mode)

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                timestamp_ms = int((frame_idx / fps) * 1000)
                pose_result = landmarker.detect_video(frame_rgb, timestamp_ms)

                suspicious, _best_hand_norm, face_d = ProximityAnalyzer.hands_near_face(
                    pose_result, hand_face_thresh
                )
                hand_pts_px = ProximityAnalyzer.get_hand_points_px(pose_result, W, H)

            if found_high:
                if require_hand_proximity and hand_pts_px:
                    hand_ok, hand_d = ProximityAnalyzer.phone_close_to_hand(
                        boxes_high, hand_pts_px, W, H, hand_phone_thresh
                    )
                det_to_plot = det_high
                used_conf = conf_high
                found_low = False
            else:
                if suspicious:
                    det_low, boxes_low, _confs_low = phone_model.detect(frame_bgr, conf_low)
                    found_low = boxes_low.shape[0] > 0
                    det_to_plot = det_low
                    used_conf = conf_low

                    if require_hand_proximity and hand_pts_px and found_low:
                        hand_ok, hand_d = ProximityAnalyzer.phone_close_to_hand(
                            boxes_low, hand_pts_px, W, H, hand_phone_thresh
                        )
                else:
                    found_low = False

            active, reason = DecisionEngine.decide_phone_use(
                found_high=found_high,
                found_low=found_low,
                suspicious=suspicious,
                hand_ok=hand_ok,
                require_hand_proximity=require_hand_proximity,
            )

            # track how many frames in a row phone is active
            if active:
                consecutive_active += 1
            else:
                consecutive_active = 0
                alert_triggered = False

                # close any ongoing incident
                if incident_active and incident_start_time and last_active_time:
                    incidents.append((incident_start_time, last_active_time))
                incident_active = False
                incident_start_time = None
                last_active_time = None

            # draw annotations
            annotated = det_to_plot.plot()

            if draw_pose and pose_result is not None:
                annotated = Visualizer.draw_pose_landmarks(
                    annotated, pose_result, draw_connections=True
                )

            annotated = Visualizer.annotate_status(
                annotated,
                active=active,
                suspicious=suspicious,
                used_conf=used_conf,
                reason=reason,
                face_dist=face_d,
                hand_dist=hand_d,
            )

            # fps counter
            dt = time.time() - t0
            if dt > 0:
                fps_now = (frame_idx + 1) / dt
                cv2.putText(
                    annotated, f"FPS: {fps_now:.1f}", (15, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                )

            # if phone active for 5+ frames, show warning and play sound
            if consecutive_active >= active_threshold:
                if not incident_active:
                    incident_active = True
                    incident_start_time = datetime.now()

                last_active_time = datetime.now()

                cv2.putText(
                    annotated, "PHONE ACTIVE (5+ frames)", (15, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
                )
                if not alert_triggered:
                    _play_alert_sound()
                    alert_triggered = True
            else:
                cv2.putText(
                    annotated, f"Active frames in a row: {consecutive_active}", (15, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                )

            # show the frame in streamlit
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(annotated_rgb, channels="RGB")
            info_placeholder.text(
                f"Consecutive ACTIVE frames: {consecutive_active} | reason: {reason}"
            )

            frame_idx += 1
            time.sleep(0.03)  # small delay so we dont hog the cpu

    finally:
        cap.release()
        if landmarker is not None:
            landmarker.close()

        # close any trailing incident
        if incident_active and incident_start_time and last_active_time:
            incidents.append((incident_start_time, last_active_time))

        session_end = datetime.now()
        report_data = _render_drive_report(
            report_placeholder=report_placeholder,
            session_start=session_start,
            session_end=session_end,
            incidents=incidents,
        )
        st.session_state["drive_report_data"] = report_data


# processes an uploaded video file frame by frame and streams progress to Streamlit

def _analyze_video_file_streamlit(
    uploaded_file,
    phone_model: YOLO,
    pose_task_path: str,
    conf_high: float,
    conf_low: float,
    hand_face_thresh: float,
    hand_phone_thresh: float,
    require_hand_proximity: bool,
    draw_pose: bool,
    progress_placeholder,
    status_placeholder,
    frame_placeholder,
    report_placeholder,
):
    # OpenCV needs a real file path, so write the upload to a temp file first
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        status_placeholder.error("Could not open video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = -1

    landmarker = None
    running_mode = vision.RunningMode.VIDEO

    frame_idx = 0
    session_start = datetime.now()
    consecutive_active = 0

    # incident tracking (in seconds relative to video start)
    active_threshold = 5
    incident_active = False
    incident_start_s: float | None = None
    last_active_s: float | None = None
    incidents_s: List[Tuple[float, float]] = []

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            current_time_s = frame_idx / fps
            H, W = frame_bgr.shape[:2]

            # high conf yolo
            det_high, boxes_high, _confs_high = phone_model.detect(frame_bgr, conf_high)
            found_high = boxes_high.shape[0] > 0

            suspicious = False
            face_d = 999.0
            hand_pts_px: List[tuple[int, int]] = []
            hand_ok = False
            hand_d = 999.0
            used_conf = conf_high
            det_to_plot = det_high

            need_pose = draw_pose or require_hand_proximity or (not found_high)
            pose_result = None

            if need_pose:
                if landmarker is None:
                    landmarker = PoseEstimator(pose_task_path, running_mode)

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                timestamp_ms = int((frame_idx / fps) * 1000)
                pose_result = landmarker.detect_video(frame_rgb, timestamp_ms)

                suspicious, _best_hand_norm, face_d = ProximityAnalyzer.hands_near_face(
                    pose_result, hand_face_thresh
                )
                hand_pts_px = ProximityAnalyzer.get_hand_points_px(pose_result, W, H)

            if found_high:
                if require_hand_proximity and hand_pts_px:
                    hand_ok, hand_d = ProximityAnalyzer.phone_close_to_hand(
                        boxes_high, hand_pts_px, W, H, hand_phone_thresh
                    )
                det_to_plot = det_high
                used_conf = conf_high
                found_low = False
            else:
                if suspicious:
                    det_low, boxes_low, _confs_low = phone_model.detect(frame_bgr, conf_low)
                    found_low = boxes_low.shape[0] > 0
                    det_to_plot = det_low
                    used_conf = conf_low

                    if require_hand_proximity and hand_pts_px and found_low:
                        hand_ok, hand_d = ProximityAnalyzer.phone_close_to_hand(
                            boxes_low, hand_pts_px, W, H, hand_phone_thresh
                        )
                else:
                    found_low = False

            active, reason = DecisionEngine.decide_phone_use(
                found_high=found_high,
                found_low=found_low,
                suspicious=suspicious,
                hand_ok=hand_ok,
                require_hand_proximity=require_hand_proximity,
            )

            # track consecutive active frames
            if active:
                consecutive_active += 1
            else:
                consecutive_active = 0
                if incident_active and incident_start_s is not None and last_active_s is not None:
                    incidents_s.append((incident_start_s, last_active_s))
                incident_active = False
                incident_start_s = None
                last_active_s = None

            # start a new incident if we just crossed the threshold
            if active and consecutive_active >= active_threshold:
                if not incident_active:
                    incident_active = True
                    incident_start_s = current_time_s
                last_active_s = current_time_s

            # annotate for preview
            annotated = det_to_plot.plot()

            if draw_pose and pose_result is not None:
                annotated = Visualizer.draw_pose_landmarks(
                    annotated, pose_result, draw_connections=True
                )

            annotated = Visualizer.annotate_status(
                annotated,
                active=active,
                suspicious=suspicious,
                used_conf=used_conf,
                reason=reason,
                face_dist=face_d,
                hand_dist=hand_d,
            )

            # update the preview roughly every half second to avoid flooding the UI
            if frame_idx % max(1, int(fps // 2)) == 0:
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(
                    annotated_rgb, channels="RGB",
                    caption=f"Annotated frame at {current_time_s:.1f} seconds",
                )

            # update progress bar
            if total_frames > 0:
                progress = min(1.0, (frame_idx + 1) / total_frames)
                progress_placeholder.progress(progress)
                status_placeholder.text(
                    f"Processing frame {frame_idx + 1} of {total_frames} "
                    f"({current_time_s:.1f} seconds) | ACTIVE frames in a row: {consecutive_active}"
                )
            else:
                status_placeholder.text(
                    f"Processing frame {frame_idx + 1} "
                    f"({current_time_s:.1f} seconds) | ACTIVE frames in a row: {consecutive_active}"
                )

            frame_idx += 1

        # close any trailing incident
        if incident_active and incident_start_s is not None and last_active_s is not None:
            incidents_s.append((incident_start_s, last_active_s))

        # incidents are tracked in video-seconds; convert to datetimes for the report functions
        if total_frames > 0:
            total_duration_s = total_frames / fps
        else:
            total_duration_s = frame_idx / fps if frame_idx > 0 else 0.0

        session_end = session_start + timedelta(seconds=total_duration_s)
        incidents_dt: List[Tuple[datetime, datetime]] = []
        for start_s, end_s in incidents_s:
            incidents_dt.append(
                (session_start + timedelta(seconds=start_s), session_start + timedelta(seconds=end_s))
            )

        report_data = _render_drive_report(
            report_placeholder=report_placeholder,
            session_start=session_start,
            session_end=session_end,
            incidents=incidents_dt,
        )
        st.session_state["drive_report_data"] = report_data

    finally:
        cap.release()
        try:
            os.remove(video_path)
        except Exception:
            pass


# cached so the model is only loaded once per session, not on every page rerun

@st.cache_resource(show_spinner="Loading YOLO phone model...")
def load_phone_model(phone_model_path: str) -> PhoneDetector:
    return PhoneDetector(phone_model_path)


def _run_evaluation(
    phone_model_path: str,
    pose_task_path: str,
    conf_high: float,
    conf_low: float,
    hand_face_thresh: float,
    hand_phone_thresh: float,
    require_hand_proximity: bool,
    draw_pose: bool,
):
    import pandas as pd
    from pathlib import Path

    st.subheader("System Evaluation")
    st.markdown(
        "Run the complete Drive Aware pipeline against your labelled test images and record "
        "classification results. Select one or more YOLO models to compare on the same test set."
    )

    # ── folder path input ────────────────────────────────────────────────────
    default_test_dir = str(Path(__file__).resolve().parent / "Testing images")
    test_dir_input = st.text_input(
        "Test images folder path",
        value=default_test_dir,
        help="Folder containing test images. Images with a matching .txt label file are expected "
        "to show Phone Use. Images without a label file are expected to show No Phone Use.",
    )

    test_dir = Path(test_dir_input)

    if not test_dir.is_dir():
        st.error(f"Folder not found: {test_dir_input}")
        return

    # ── discover images and determine expected labels ─────────────────────────
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(
        [p for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in img_exts],
        key=lambda p: p.name,
    )

    if not image_paths:
        st.warning("No images found in the selected folder.")
        return

    def get_expected(img_path: Path) -> str:
        label_path = img_path.with_suffix(".txt")
        return "Phone Use" if label_path.exists() else "No Phone Use"

    total_images = len(image_paths)
    phone_count = sum(1 for p in image_paths if get_expected(p) == "Phone Use")
    no_phone_count = total_images - phone_count

    col1, col2, col3 = st.columns(3)
    col1.metric("Total images", total_images)
    col2.metric("Expected Phone Use", phone_count)
    col3.metric("Expected No Phone Use", no_phone_count)

    st.markdown("---")

    # ── model selection: paths + multiselect ───────────────────────────────────
    st.subheader("Models to compare")

    with st.expander("Model paths", expanded=False):
        st.caption("Override the default weight file locations if your models are stored elsewhere.")
        path_v11 = st.text_input("YOLO v11 weights path", value=EVAL_MODELS["YOLO v11"], key="eval_path_v11")
        path_v8 = st.text_input("YOLO v8 weights path", value=EVAL_MODELS["YOLO v8"], key="eval_path_v8")
        path_v5 = st.text_input("YOLO v5 weights path", value=EVAL_MODELS["YOLO v5"], key="eval_path_v5")

    models_to_compare = st.multiselect(
        "Models to compare",
        options=["YOLO v11", "YOLO v8", "YOLO v5"],
        default=["YOLO v11", "YOLO v8", "YOLO v5"],
        help="Run evaluation for each selected model. Same images and thresholds for all.",
    )

    path_by_model = {
        "YOLO v11": path_v11.strip() or EVAL_MODELS["YOLO v11"],
        "YOLO v8": path_v8.strip() or EVAL_MODELS["YOLO v8"],
        "YOLO v5": path_v5.strip() or EVAL_MODELS["YOLO v5"],
    }

    st.markdown("---")

    if "eval_results_by_model" not in st.session_state:
        st.session_state["eval_results_by_model"] = {}
    if "eval_timing_by_model" not in st.session_state:
        st.session_state["eval_timing_by_model"] = {}

    if st.button("Run Evaluation", type="primary"):
        if not models_to_compare:
            st.error("Select at least one model to compare.")
        else:
            st.session_state["eval_results_by_model"] = {}
            st.session_state["eval_timing_by_model"] = {}
            progress_bar = st.progress(0)
            status_text = st.empty()
            preview_col, _ = st.columns([1, 1])
            total_models = len(models_to_compare)
            total_steps = total_models * total_images
            step = 0

            for model_name in models_to_compare:
                path = path_by_model[model_name]
                try:
                    detector = load_phone_model(path)
                except Exception as e:
                    st.error(f"Could not load **{model_name}** from `{path}`: {e}")
                    continue

                results = []
                timings = []  # per-image timing dicts collected for this model
                for i, img_path in enumerate(image_paths):
                    status_text.text(f"{model_name}: {i + 1}/{total_images} — {img_path.name}")
                    frame_bgr = cv2.imread(str(img_path))
                    if frame_bgr is None:
                        results.append({
                            "Image": img_path.name,
                            "Expected": get_expected(img_path),
                            "Actual": "Error",
                            "Confidence": "N/A",
                            "Reason": "Could not read image",
                            "Inference (ms)": "N/A",
                            "Correct": "Error",
                            "Model": model_name,
                        })
                        step += 1
                        progress_bar.progress(step / total_steps)
                        continue

                    t_start = time.time()
                    annotated_bgr, active, timing = _process_single_image_bgr(
                        frame_bgr=frame_bgr,
                        phone_model=detector,
                        pose_task_path=pose_task_path,
                        conf_high=conf_high,
                        conf_low=conf_low,
                        hand_face_thresh=hand_face_thresh,
                        hand_phone_thresh=hand_phone_thresh,
                        require_hand_proximity=require_hand_proximity,
                        draw_pose=draw_pose,
                    )
                    inference_ms = (time.time() - t_start) * 1000
                    timings.append(timing)
                    expected = get_expected(img_path)
                    actual = "Phone Use" if active else "No Phone Use"
                    correct = "Yes" if expected == actual else "No"

                    H, W = frame_bgr.shape[:2]
                    det_high, boxes_high, confs_high = detector.detect(frame_bgr, conf_high)
                    found_high = boxes_high.shape[0] > 0
                    if found_high and len(confs_high) > 0:
                        conf_val = f"{float(confs_high[0]):.2f}"
                        reason_val = "HighConf phone"
                    else:
                        conf_val = f"{conf_high:.2f}"
                        reason_val = "No high-conf detection"

                    results.append({
                        "Image": img_path.name,
                        "Expected": expected,
                        "Actual": actual,
                        "Confidence": conf_val,
                        "Reason": reason_val,
                        "Inference (ms)": f"{inference_ms:.0f}",
                        "Correct": correct,
                        "Model": model_name,
                    })

                    if i % 5 == 0 and model_name == models_to_compare[0]:
                        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                        preview_col.image(
                            annotated_rgb,
                            caption=f"{img_path.name} | {model_name} | {actual}",
                            use_container_width=True,
                        )
                    step += 1
                    progress_bar.progress(step / total_steps)

                st.session_state["eval_results_by_model"][model_name] = results
                st.session_state["eval_timing_by_model"][model_name] = timings

            status_text.text("Evaluation complete.")

    # ── display comparison if available ──────────────────────────────────────
    if st.session_state["eval_results_by_model"]:
        results_by_model = st.session_state["eval_results_by_model"]
        model_names = list(results_by_model.keys())

        # pre-compute summary rows (shared across sub-tabs)
        summary_rows = []
        for model_name in model_names:
            df = pd.DataFrame(results_by_model[model_name])
            total = len(df)
            correct_count = len(df[df["Correct"] == "Yes"])
            incorrect_count = len(df[df["Correct"] == "No"])
            # accuracy = (TP + TN) / total — proportion of all predictions that are correct
            accuracy = correct_count / total if total > 0 else 0.0
            # confusion matrix counts
            tp = len(df[(df["Expected"] == "Phone Use") & (df["Actual"] == "Phone Use")])   # correctly flagged
            fp = len(df[(df["Expected"] == "No Phone Use") & (df["Actual"] == "Phone Use")]) # false alarm
            fn = len(df[(df["Expected"] == "Phone Use") & (df["Actual"] == "No Phone Use")]) # missed detection
            tn = len(df[(df["Expected"] == "No Phone Use") & (df["Actual"] == "No Phone Use")]) # correctly clear
            # precision = TP / (TP + FP) — of all "phone use" predictions, how many were correct
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            # recall = TP / (TP + FN) — of all actual phone-use cases, how many did we catch
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            # F1 = 2 * (precision * recall) / (precision + recall) — harmonic mean, balances both
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            try:
                avg_ms = df["Inference (ms)"].replace("N/A", np.nan).astype(float).mean()
            except Exception:
                avg_ms = float("nan")
            summary_rows.append({
                "Model": model_name,
                "Accuracy": f"{accuracy:.1%}",
                "Precision": f"{precision:.1%}",
                "Recall": f"{recall:.1%}",
                "F1 Score": f"{f1:.1%}",
                "Avg inference (ms)": f"{avg_ms:.0f}" if not np.isnan(avg_ms) else "N/A",
                "True Positive": tp,
                "True Negative": tn,
                "False Positive": fp,
                "False Negative": fn,
                "Correct": correct_count,
                "Incorrect": incorrect_count,
            })

        res_tab_summary, res_tab_timing, res_tab_perim, res_tab_errors = st.tabs(
            ["Summary", "Stage Timing", "Per-image Results", "Errors"]
        )

        # ── Summary sub-tab ──
        with res_tab_summary:
            st.subheader("Summary comparison")
            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df.set_index("Model"), use_container_width=True, height=50 + 36 * len(summary_rows))

            st.markdown("---")
            all_rows = []
            for model_name in model_names:
                for r in results_by_model[model_name]:
                    all_rows.append(r)
            combined_df = pd.DataFrame(all_rows)
            csv_bytes = combined_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download all results as CSV (with Model column)",
                data=csv_bytes,
                file_name="driveaware_evaluation_comparison.csv",
                mime="text/csv",
            )
            total = len(combined_df)
            correct_total = len(combined_df[combined_df["Correct"] == "Yes"])
            st.caption(f"Evaluation ran on {len(model_names)} model(s) × {total_images} images | Total rows: {total} | Correct: {correct_total}")

        # ── Stage Timing sub-tab ──
        with res_tab_timing:
            st.subheader("Stage timing breakdown")
            st.caption(
                "Average milliseconds per pipeline stage across all test images. "
                "MediaPipe (conditional) averages only over images where pose estimation actually ran "
                "(skipped when YOLO fires at high confidence). "
                "Total (stages) is the sum of all timed stages; Wall-clock is the outer timer "
                "that wraps the whole function — any gap indicates overhead outside the four stages "
                "(e.g. image loading, BGR conversion, annotation drawing)."
            )
            timing_rows = []
            timing_by_model = st.session_state.get("eval_timing_by_model", {})
            for mn in model_names:
                t_list = timing_by_model.get(mn, [])
                if not t_list:
                    continue
                n = len(t_list)
                pose_ran_list = [t["pose_ms"] for t in t_list if t["pose_ran"]]

                avg = lambda key: sum(t[key] for t in t_list) / n  # noqa: E731
                timing_rows.append({
                    "Model": mn,
                    "YOLO high-conf (ms)": f"{avg('yolo_high_ms'):.1f}",
                    "YOLO low-conf (ms)": f"{avg('yolo_low_ms'):.1f}",
                    "MediaPipe — all images (ms)": f"{avg('pose_ms'):.1f}",
                    "MediaPipe — when ran (ms)": f"{sum(pose_ran_list) / len(pose_ran_list):.1f}" if pose_ran_list else "N/A",
                    "Proximity (ms)": f"{avg('proximity_ms'):.1f}",
                    "Decision (ms)": f"{avg('decision_ms'):.1f}",
                    "Total stages (ms)": f"{avg('total_ms'):.1f}",
                    "Images with pose": f"{len(pose_ran_list)}/{n}",
                })
            if timing_rows:
                timing_df = pd.DataFrame(timing_rows)
                st.dataframe(timing_df.set_index("Model"), use_container_width=True)
            else:
                st.info("Run evaluation to see stage timing data.")

        # ── Per-image Results sub-tab ──
        with res_tab_perim:
            st.subheader("Per-image comparison")
            first_model = model_names[0]
            base = results_by_model[first_model]
            comp_dict = {
                "Image": [r["Image"] for r in base],
                "Expected": [r["Expected"] for r in base],
            }
            for mn in model_names:
                comp_dict[f"Actual ({mn})"] = [r["Actual"] for r in results_by_model[mn]]
                comp_dict[f"Correct ({mn})"] = [r["Correct"] for r in results_by_model[mn]]
            comp_df = pd.DataFrame(comp_dict)

            # ── Search + page-size controls ──
            pcol_search, pcol_size = st.columns([3, 1])
            with pcol_search:
                perim_search = st.text_input(
                    "Filter by image name",
                    value="",
                    placeholder="e.g. img_123…",
                    label_visibility="collapsed",
                    key="perim_search_input",
                )
            with pcol_size:
                perim_page_size = st.selectbox("Per page", [10, 25, 50, 100], index=1, key="perim_page_size_sel")

            # filter rows
            if perim_search.strip():
                mask = comp_df["Image"].str.contains(perim_search.strip(), case=False, na=False)
                filtered_comp_df = comp_df[mask]
            else:
                filtered_comp_df = comp_df

            total_perim = len(filtered_comp_df)
            n_perim_pages = max(1, math.ceil(total_perim / perim_page_size))

            # reset page when search or page size changes
            if (perim_search != st.session_state.get("_perim_prev_search", "")
                    or perim_page_size != st.session_state.get("_perim_prev_size", perim_page_size)):
                st.session_state["perim_page"] = 0
            st.session_state["_perim_prev_search"] = perim_search
            st.session_state["_perim_prev_size"] = perim_page_size

            if "perim_page" not in st.session_state:
                st.session_state["perim_page"] = 0
            perim_page = max(0, min(st.session_state["perim_page"], n_perim_pages - 1))
            st.session_state["perim_page"] = perim_page

            p_start = perim_page * perim_page_size
            p_end = min(p_start + perim_page_size, total_perim)
            page_comp_df = filtered_comp_df.iloc[p_start:p_end]

            st.caption(f"Showing rows {p_start + 1}–{p_end} of {total_perim} · Page {perim_page + 1} of {n_perim_pages}")
            st.dataframe(page_comp_df, use_container_width=True, hide_index=True)

            pbtn_prev, _, pbtn_next = st.columns([1, 5, 1])
            with pbtn_prev:
                if st.button("← Previous", disabled=(perim_page == 0), key="perim_prev_btn"):
                    st.session_state["perim_page"] = perim_page - 1
                    st.rerun()
            with pbtn_next:
                if st.button("Next →", disabled=(perim_page >= n_perim_pages - 1), key="perim_next_btn"):
                    st.session_state["perim_page"] = perim_page + 1
                    st.rerun()

            st.markdown("---")
            for model_name in model_names:
                df = pd.DataFrame(results_by_model[model_name])
                with st.expander(f"Full results: {model_name}"):
                    def highlight_correct(row):
                        if row.get("Correct") == "Yes":
                            return ["background-color: #d4edda"] * len(row)
                        if row.get("Correct") == "No":
                            return ["background-color: #f8d7da"] * len(row)
                        return [""] * len(row)
                    cols_show = [c for c in df.columns if c != "Model"]
                    st.dataframe(df[cols_show].style.apply(highlight_correct, axis=1), use_container_width=True, height=300)

        # ── Errors sub-tab ──
        with res_tab_errors:
            st.subheader("Incorrect predictions")
            st.caption(
                "Rows where the model predicted wrong. Expand a model to see only its incorrect cases, "
                "and click an image to view it."
            )
            cols_incorrect = ["Image", "Expected", "Actual", "Confidence", "Reason", "Inference (ms)"]
            ERR_IMG_PAGE_SIZE = 10
            for model_name in model_names:
                df = pd.DataFrame(results_by_model[model_name])
                incorrect_df = df[df["Correct"] == "No"]
                if incorrect_df.empty:
                    st.write(f"**{model_name}**: No incorrect predictions.")
                else:
                    with st.expander(f"{model_name} — {len(incorrect_df)} incorrect"):
                        cols_show = [c for c in cols_incorrect if c in incorrect_df.columns]
                        st.dataframe(
                            incorrect_df[cols_show],
                            use_container_width=True,
                            height=min(260, 50 + 30 * len(incorrect_df)),
                        )

                        st.markdown("---")
                        # pagination for image previews
                        total_err = len(incorrect_df)
                        n_err_pages = max(1, math.ceil(total_err / ERR_IMG_PAGE_SIZE))
                        err_page_key = f"err_page_{model_name.replace(' ', '_')}"
                        if err_page_key not in st.session_state:
                            st.session_state[err_page_key] = 0
                        err_page = max(0, min(st.session_state[err_page_key], n_err_pages - 1))
                        st.session_state[err_page_key] = err_page

                        e_start = err_page * ERR_IMG_PAGE_SIZE
                        e_end = min(e_start + ERR_IMG_PAGE_SIZE, total_err)
                        st.caption(f"Image previews: {e_start + 1}–{e_end} of {total_err} · Page {err_page + 1} of {n_err_pages}")

                        ebtn_prev_key = f"err_prev_{model_name.replace(' ', '_')}"
                        ebtn_next_key = f"err_next_{model_name.replace(' ', '_')}"
                        ecol_prev, _, ecol_next = st.columns([1, 5, 1])
                        with ecol_prev:
                            if st.button("← Previous", disabled=(err_page == 0), key=ebtn_prev_key):
                                st.session_state[err_page_key] = err_page - 1
                                st.rerun()
                        with ecol_next:
                            if st.button("Next →", disabled=(err_page >= n_err_pages - 1), key=ebtn_next_key):
                                st.session_state[err_page_key] = err_page + 1
                                st.rerun()

                        page_incorrect = incorrect_df.iloc[e_start:e_end]
                        for _, row in page_incorrect.iterrows():
                            img_name = row.get("Image")
                            if not img_name:
                                continue
                            img_path = test_dir / img_name
                            with st.expander(
                                f"{img_name} — Expected: {row.get('Expected')} | Actual: {row.get('Actual')}"
                            ):
                                frame_bgr = cv2.imread(str(img_path))
                                if frame_bgr is None:
                                    st.warning(f"Could not read image: {img_name}")
                                else:
                                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                                    st.image(
                                        frame_rgb,
                                        caption=(
                                            f"{img_name} | {model_name} | "
                                            f"Expected: {row.get('Expected')} | Actual: {row.get('Actual')}"
                                        ),
                                        use_container_width=True,
                                    )


# entry point — sets up the page, sidebar, and the four main tabs

def main():
    st.set_page_config(
        page_title="Drive Aware",
        layout="wide",
    )

    st.title("Drive Aware")
    st.markdown(
        "AI-based computer vision system for detecting mobile phone use while driving."
    )

    # all detection parameters are controlled from the sidebar
    with st.sidebar:
        st.markdown("## Drive Aware")
        st.header("Settings")

        default_phone_model = "weights/best.pt"
        default_pose_task = "pose_landmarker_full.task"

        phone_model_path = st.text_input("YOLO phone model path", default_phone_model)
        pose_task_path = st.text_input("MediaPipe pose .task path", default_pose_task)

        st.subheader("Thresholds")
        conf_high = st.slider(
            "YOLO conf (normal)", 0.0, 1.0, 0.75, 0.01,
            help="Minimum confidence for the detector to flag a phone. Higher = fewer false alarms.",
        )
        conf_low = st.slider(
            "YOLO conf (when suspicious)", 0.0, 1.0, 0.25, 0.01,
            help="Fallback confidence used only when the pose looks suspicious. Lower catches more borderline cases.",
        )
        hand_face_thresh = st.slider(
            "Hand–face distance (normalized)", 0.05, 0.5, 0.18, 0.01,
            help="How close a hand must be to the face to count as suspicious. Higher = more sensitive.",
        )
        hand_phone_thresh = st.slider(
            "Phone–hand distance (normalized)", 0.05, 0.5, 0.12, 0.01,
            help="How close a detected phone must be to a hand. Only used when hand-proximity filtering is on.",
        )

        st.subheader("Logic & Visuals")
        require_hand_proximity = st.checkbox(
            "Require phone near hand / suspicious face", value=False,
            help="If enabled, a phone is only considered ACTIVE when it's near a hand or the hands are near the face.",
        )
        draw_pose = st.checkbox("Draw pose skeleton", value=True)

    # load the model before setting up tabs so any path error shows at the top
    try:
        phone_model = load_phone_model(phone_model_path)
    except Exception as e:
        st.error(f"Could not load YOLO model from `{phone_model_path}`.\n\nError: {e}")
        return

    tab_img, tab_vid, tab_live, tab_eval = st.tabs(
        ["Image Analysis", "Video Analysis", "Live Webcam", "Evaluation"]
    )

    # ── Image Analysis tab ──
    with tab_img:
        st.subheader("Image Analysis")
        image_source = st.radio(
            "Image source",
            ["Upload image(s)", "Choose from sample images", "Take a photo"],
            help="Upload your own files, pick from sample images, or use your camera.",
        )

        if image_source == "Upload image(s)":
            uploaded_files = st.file_uploader(
                "Upload one or more images",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                accept_multiple_files=True,
            )

            if uploaded_files:
                for up in uploaded_files:
                    st.markdown(f"**File:** `{up.name}`")
                    with st.spinner(f"Processing {up.name}..."):
                        pil_img = Image.open(up).convert("RGB")
                        frame_rgb = np.array(pil_img)
                        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                        annotated_bgr, _, _ = _process_single_image_bgr(
                            frame_bgr=frame_bgr,
                            phone_model=phone_model,
                            pose_task_path=pose_task_path,
                            conf_high=conf_high,
                            conf_low=conf_low,
                            hand_face_thresh=hand_face_thresh,
                            hand_phone_thresh=hand_phone_thresh,
                            require_hand_proximity=require_hand_proximity,
                            draw_pose=draw_pose,
                        )
                        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                        st.image(annotated_rgb, caption=f"Detection result - {up.name}", use_container_width=True)

        elif image_source == "Choose from sample images":
            samples = _get_sample_images_with_labels()
            if not samples:
                st.info("No sample images found in the Testing images folder.")
            else:
                # ── Search + page-size controls ──
                col_search, col_size = st.columns([3, 1])
                with col_search:
                    search_term = st.text_input(
                        "Search images",
                        value="",
                        placeholder="Filter by name…",
                        label_visibility="collapsed",
                        key="img_sample_search_input",
                    )
                with col_size:
                    page_size = st.selectbox("Per page", [10, 20, 30, 50], index=1, key="img_sample_page_size_sel")

                # filter by search term
                if search_term.strip():
                    term = search_term.strip().lower()
                    filtered = [(p, lbl) for p, lbl in samples if term in lbl.lower() or term in p.name.lower()]
                else:
                    filtered = samples

                total = len(filtered)
                n_pages = max(1, math.ceil(total / page_size))

                # reset page when search or page size changes
                prev_search = st.session_state.get("_img_prev_search", "")
                prev_size = st.session_state.get("_img_prev_size", page_size)
                if search_term != prev_search or page_size != prev_size:
                    st.session_state["img_sample_page"] = 0
                st.session_state["_img_prev_search"] = search_term
                st.session_state["_img_prev_size"] = page_size

                if "img_sample_page" not in st.session_state:
                    st.session_state["img_sample_page"] = 0
                page = max(0, min(st.session_state["img_sample_page"], n_pages - 1))
                st.session_state["img_sample_page"] = page

                start = page * page_size
                end = min(start + page_size, total)
                page_samples = filtered[start:end]

                st.caption(f"Showing {start + 1}–{end} of {total} image(s) · Page {page + 1} of {n_pages}")

                # thumbnail grid for current page
                cols_per_row = 4
                for i in range(0, len(page_samples), cols_per_row):
                    row = st.columns(cols_per_row)
                    for j, col in enumerate(row):
                        idx = i + j
                        if idx >= len(page_samples):
                            break
                        path, label = page_samples[idx]
                        thumb = _load_thumbnail(path)
                        if thumb is not None:
                            col.image(thumb, caption=label, use_container_width=True)
                        else:
                            col.caption(label)

                # Previous / Next pagination buttons
                btn_prev, _, btn_next = st.columns([1, 5, 1])
                with btn_prev:
                    if st.button("← Previous", disabled=(page == 0), key="img_prev_btn"):
                        st.session_state["img_sample_page"] = page - 1
                        st.rerun()
                with btn_next:
                    if st.button("Next →", disabled=(page >= n_pages - 1), key="img_next_btn"):
                        st.session_state["img_sample_page"] = page + 1
                        st.rerun()

                # multiselect from ALL filtered results (not just current page)
                friendly_names = [lbl for _, lbl in filtered]
                path_by_name = {lbl: p for p, lbl in filtered}
                selected = st.multiselect(
                    "Select one or more sample images to analyse",
                    friendly_names,
                    help=f"{total} image(s) match the current filter.",
                )
                if selected:
                    for label in selected:
                        path = path_by_name.get(label)
                        if path is None or not path.is_file():
                            continue
                        st.markdown(f"**{label}**")
                        with st.spinner(f"Processing {label}..."):
                            frame_bgr = cv2.imread(str(path))
                            if frame_bgr is None:
                                st.warning(f"Could not read image: {label}")
                                continue
                            annotated_bgr, _, _ = _process_single_image_bgr(
                                frame_bgr=frame_bgr,
                                phone_model=phone_model,
                                pose_task_path=pose_task_path,
                                conf_high=conf_high,
                                conf_low=conf_low,
                                hand_face_thresh=hand_face_thresh,
                                hand_phone_thresh=hand_phone_thresh,
                                require_hand_proximity=require_hand_proximity,
                                draw_pose=draw_pose,
                            )
                            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                            st.image(annotated_rgb, caption=f"Detection result — {label}", use_container_width=True)

        else:  # Take a photo
            cam_img = st.camera_input("Take a photo to analyse")
            if cam_img is not None:
                with st.spinner("Running detection..."):
                    pil_img = Image.open(cam_img).convert("RGB")
                    frame_rgb = np.array(pil_img)
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    annotated_bgr, _, _ = _process_single_image_bgr(
                        frame_bgr=frame_bgr,
                        phone_model=phone_model,
                        pose_task_path=pose_task_path,
                        conf_high=conf_high,
                        conf_low=conf_low,
                        hand_face_thresh=hand_face_thresh,
                        hand_phone_thresh=hand_phone_thresh,
                        require_hand_proximity=require_hand_proximity,
                        draw_pose=draw_pose,
                    )
                    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, caption="Detection result", use_container_width=True)

    # ── Video Analysis tab ──
    with tab_vid:
        st.subheader("Video Analysis")
        st.markdown(
            "Upload a video or choose from sample clips to analyse. "
            "The system will identify periods of active phone use and generate a safety report."
        )

        video_source = st.radio(
            "Video source",
            ["Upload video", "Choose from sample videos"],
        )

        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        frame_placeholder = st.empty()
        report_placeholder = st.empty()

        if video_source == "Upload video":
            video_file = st.file_uploader(
                "Upload a video file",
                type=["mp4", "mov", "avi", "mkv", "webm"],
                accept_multiple_files=False,
            )

            if video_file is not None and st.button("Start video analysis"):
                with st.spinner("Analysing video. This may take a while for long files."):
                    _analyze_video_file_streamlit(
                        uploaded_file=video_file,
                        phone_model=phone_model,
                        pose_task_path=pose_task_path,
                        conf_high=conf_high,
                        conf_low=conf_low,
                        hand_face_thresh=hand_face_thresh,
                        hand_phone_thresh=hand_phone_thresh,
                        require_hand_proximity=require_hand_proximity,
                        draw_pose=draw_pose,
                        progress_placeholder=progress_placeholder,
                        status_placeholder=status_placeholder,
                        frame_placeholder=frame_placeholder,
                        report_placeholder=report_placeholder,
                    )

        else:  # Choose from sample videos
            sample_videos = _get_sample_videos_with_labels()
            if not sample_videos:
                st.info("No sample videos available.")
            else:
                VID_PAGE_SIZE = 5
                total_vids = len(sample_videos)
                n_vid_pages = max(1, math.ceil(total_vids / VID_PAGE_SIZE))

                if "vid_sample_page" not in st.session_state:
                    st.session_state["vid_sample_page"] = 0
                vid_page = max(0, min(st.session_state["vid_sample_page"], n_vid_pages - 1))
                st.session_state["vid_sample_page"] = vid_page

                v_start = vid_page * VID_PAGE_SIZE
                v_end = min(v_start + VID_PAGE_SIZE, total_vids)
                page_videos = sample_videos[v_start:v_end]

                st.caption(f"Showing {v_start + 1}–{v_end} of {total_vids} video(s) · Page {vid_page + 1} of {n_vid_pages}")

                # first-frame thumbnail grid for current page
                cols_per_row = 4
                for i in range(0, len(page_videos), cols_per_row):
                    row = st.columns(cols_per_row)
                    for j, col in enumerate(row):
                        idx = i + j
                        if idx >= len(page_videos):
                            break
                        path, label = page_videos[idx]
                        thumb = _load_video_thumbnail(path)
                        if thumb is not None:
                            col.image(thumb, caption=label, use_container_width=True)
                        else:
                            col.caption(f"{label}\n(no preview)")

                # Previous / Next pagination buttons
                vbtn_prev, _, vbtn_next = st.columns([1, 5, 1])
                with vbtn_prev:
                    if st.button("← Previous", disabled=(vid_page == 0), key="vid_prev_btn"):
                        st.session_state["vid_sample_page"] = vid_page - 1
                        st.rerun()
                with vbtn_next:
                    if st.button("Next →", disabled=(vid_page >= n_vid_pages - 1), key="vid_next_btn"):
                        st.session_state["vid_sample_page"] = vid_page + 1
                        st.rerun()

                # select from visible page only
                page_video_labels = [lbl for _, lbl in page_videos]
                video_path_by_label = {lbl: p for p, lbl in page_videos}
                selected_video = st.selectbox(
                    "Select a sample video to analyse",
                    page_video_labels,
                    help=f"Showing {len(page_video_labels)} video(s) on this page.",
                )
                if selected_video and st.button("Analyse selected video"):
                    video_path = video_path_by_label[selected_video]
                    with st.spinner(f"Analysing {selected_video}..."):
                        _analyze_video_file_streamlit(
                            uploaded_file=_SampleVideo(video_path),
                            phone_model=phone_model,
                            pose_task_path=pose_task_path,
                            conf_high=conf_high,
                            conf_low=conf_low,
                            hand_face_thresh=hand_face_thresh,
                            hand_phone_thresh=hand_phone_thresh,
                            require_hand_proximity=require_hand_proximity,
                            draw_pose=draw_pose,
                            progress_placeholder=progress_placeholder,
                            status_placeholder=status_placeholder,
                            frame_placeholder=frame_placeholder,
                            report_placeholder=report_placeholder,
                        )

        # offer pdf download if report exists
        if REPORTLAB_AVAILABLE and "drive_report_data" in st.session_state:
            report_data = st.session_state["drive_report_data"]
            pdf_bytes = _build_pdf_from_report_data(report_data)
            st.download_button(
                label="Download drive report as PDF",
                data=pdf_bytes,
                file_name="drive_report.pdf",
                mime="application/pdf",
                key="pdf_download_video",
            )
        elif "drive_report_data" in st.session_state and not REPORTLAB_AVAILABLE:
            st.info("To enable PDF downloads, install reportlab: pip install reportlab")

    # ── Live Webcam tab ──
    with tab_live:
        st.subheader("Live Webcam")
        st.markdown(
            "Click **Start live detection** to begin streaming from your webcam. "
            "The session ends automatically after the selected time limit. "
            "Sustained phone use triggers a visual alert and an audio notification."
        )

        max_seconds = st.slider(
            "Max live duration (seconds)", 5, 300, 60, 5,
            help="The webcam session stops automatically after this many seconds.",
        )
        frame_placeholder = st.empty()
        info_placeholder = st.empty()
        report_placeholder = st.empty()

        if st.button("Start live detection"):
            _run_live_webcam_streamlit(
                phone_model=phone_model,
                pose_task_path=pose_task_path,
                conf_high=conf_high,
                conf_low=conf_low,
                hand_face_thresh=hand_face_thresh,
                hand_phone_thresh=hand_phone_thresh,
                require_hand_proximity=require_hand_proximity,
                draw_pose=draw_pose,
                max_seconds=max_seconds,
                frame_placeholder=frame_placeholder,
                info_placeholder=info_placeholder,
                report_placeholder=report_placeholder,
            )

        # offer pdf download if report exists
        if REPORTLAB_AVAILABLE and "drive_report_data" in st.session_state:
            report_data = st.session_state["drive_report_data"]
            pdf_bytes = _build_pdf_from_report_data(report_data)
            st.download_button(
                label="Download drive report as PDF",
                data=pdf_bytes,
                file_name="drive_report.pdf",
                mime="application/pdf",
                key="pdf_download_webcam",
            )
        elif "drive_report_data" in st.session_state and not REPORTLAB_AVAILABLE:
            st.info("To enable PDF downloads, install reportlab: pip install reportlab")

    # ── Evaluation tab ──
    with tab_eval:
        _run_evaluation(
            phone_model_path=phone_model_path,
            pose_task_path=pose_task_path,
            conf_high=conf_high,
            conf_low=conf_low,
            hand_face_thresh=hand_face_thresh,
            hand_phone_thresh=hand_phone_thresh,
            require_hand_proximity=require_hand_proximity,
            draw_pose=draw_pose,
        )


if __name__ == "__main__":
    main()