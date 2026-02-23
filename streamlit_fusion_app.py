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

# try importing reportlab for pdf generation - its optional
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


# ─── single image pipeline (streamlit version, no cv2 windows) ───

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

    # high confidence yolo pass
    det_high, boxes_high, _ = phone_model.detect(frame_bgr, conf_high)
    found_high = boxes_high.shape[0] > 0

    # defaults
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
        landmarker = PoseEstimator(pose_task_path, vision.RunningMode.IMAGE)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pose_result = landmarker.detect_image(frame_rgb)

        suspicious, _best_hand_norm, face_d = ProximityAnalyzer.hands_near_face(
            pose_result, hand_face_thresh
        )
        hand_pts_px = ProximityAnalyzer.get_hand_points_px(pose_result, W, H)
        landmarker.close()

    if found_high:
        # phone found at high conf
        if require_hand_proximity and hand_pts_px:
            hand_ok, hand_d = ProximityAnalyzer.phone_close_to_hand(
                boxes_high, hand_pts_px, W, H, hand_phone_thresh
            )
        det_to_plot = det_high
        used_conf = conf_high
        found_low = False
    else:
        # fallback to low conf if pose is suspicious
        if suspicious:
            det_low, boxes_low, _ = phone_model.detect(frame_bgr, conf_low)
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

    cv2.putText(
        annotated, "Streamlit single-image mode", (15, 105),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )

    return annotated, active


# ─── alert sound ───

def _play_alert_sound():
    # tries to play a sound when phone is detected - different per OS
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


# ─── report helpers ───

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
    # crunch all the numbers for the drive report

    total_duration_s = (session_end - session_start).total_seconds()
    phone_use_s = sum((end - start).total_seconds() for start, end in incidents if end > start)
    phone_use_pct = (phone_use_s / total_duration_s * 100) if total_duration_s > 0 else 0.0

    # safety score: 100 = no phone use, goes down the more you use it
    safety_score = int(max(0, 100 - phone_use_pct * 1.2))

    date_str = session_start.strftime("%B %d, %Y")
    start_time_str = session_start.strftime("%H:%M:%S")
    duration_str = _format_duration(total_duration_s)

    # build incident list with formatted info
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

    # text-based timeline bars for the markdown report
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

    # pick an insight message based on how much phone use there was
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

    # recommendations for the pdf report
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


# ─── pdf generation ───

def _build_pdf_from_report_data(report_data: dict) -> bytes:
    # builds a nice looking pdf report using reportlab

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


# ─── markdown report renderer ───

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


# ─── live webcam loop ───

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


# ─── video file analysis ───

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
    # save the upload to a temp file so opencv can read it
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

            # show a preview frame every half second or so
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

        # convert second-based incidents to datetimes for the report
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


# ─── model loading (cached so it only loads once) ───

@st.cache_resource(show_spinner="Loading YOLO phone model...")
def load_phone_model(phone_model_path: str) -> PhoneDetector:
    return PhoneDetector(phone_model_path)


# ─── main streamlit app ───

def main():
    st.set_page_config(
        page_title="Phone Usage Detector (YOLO + MediaPipe)",
        layout="wide",
    )

    st.title("Phone Usage Detector")
    st.markdown(
        "Fusion of **YOLO phone detection** with **MediaPipe pose landmarks** "
        "for more robust 'phone use' detection."
    )

    # sidebar settings
    with st.sidebar:
        st.header("Settings")

        default_phone_model = "weights/best.pt"
        default_pose_task = "pose_landmarker_full.task"

        phone_model_path = st.text_input("YOLO phone model path", default_phone_model)
        pose_task_path = st.text_input("MediaPipe pose .task path", default_pose_task)

        st.subheader("Thresholds")
        conf_high = st.slider("YOLO conf (normal)", 0.0, 1.0, 0.75, 0.01)
        conf_low = st.slider("YOLO conf (when suspicious)", 0.0, 1.0, 0.25, 0.01)
        hand_face_thresh = st.slider("Hand–face distance (normalized)", 0.05, 0.5, 0.18, 0.01)
        hand_phone_thresh = st.slider("Phone–hand distance (normalized)", 0.05, 0.5, 0.12, 0.01)

        st.subheader("Logic & Visuals")
        require_hand_proximity = st.checkbox(
            "Require phone near hand / suspicious face", value=False,
            help="If enabled, a phone is only considered ACTIVE when it's near a hand or the hands are near the face.",
        )
        draw_pose = st.checkbox("Draw pose skeleton", value=True)

    # input mode selector
    mode = st.radio(
        "Choose input mode",
        ["Upload image(s)", "Upload video", "Live webcam (external window)"],
        help="Upload lets you test your own images or videos. Live mode streams directly from your webcam.",
    )

    # load yolo model (cached)
    try:
        phone_model = load_phone_model(phone_model_path)
    except Exception as e:
        st.error(f"Could not load YOLO model from `{phone_model_path}`.\n\nError: {e}")
        return

    # ── image upload mode ──
    if mode == "Upload image(s)":
        st.subheader("Upload images to test")
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

                    annotated_bgr, _ = _process_single_image_bgr(
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

    # ── video upload mode ──
    elif mode == "Upload video":
        st.subheader("Upload a recorded video")
        st.markdown(
            "Upload a driving or classroom recording. The system will analyse the entire video, "
            "identify periods of active phone use, and generate a drive-style report at the end."
        )

        video_file = st.file_uploader(
            "Upload a video file",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            accept_multiple_files=False,
        )

        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        frame_placeholder = st.empty()
        report_placeholder = st.empty()

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

        # offer pdf download if report exists
        if REPORTLAB_AVAILABLE and "drive_report_data" in st.session_state:
            report_data = st.session_state["drive_report_data"]
            pdf_bytes = _build_pdf_from_report_data(report_data)
            st.download_button(
                label="Download drive report as PDF",
                data=pdf_bytes,
                file_name="drive_report.pdf",
                mime="application/pdf",
            )
        elif "drive_report_data" in st.session_state and not REPORTLAB_AVAILABLE:
            st.info("To enable PDF downloads, install reportlab: pip install reportlab")

    # ── live webcam mode ──
    elif mode == "Live webcam (external window)":
        st.subheader("Live webcam")
        st.markdown(
            "- Click **Start live detection** to stream frames in this page.\n"
            "- The session stops automatically after the selected time limit.\n"
            "- If the phone is **ACTIVE** for more than **5 frames in a row**, the overlay shows "
            "*PHONE ACTIVE (5+ frames)* and your machine plays a sound."
        )

        max_seconds = st.slider(
            "Max live duration (seconds)", 5, 300, 60, 5,
            help="Safety limit for the live loop."
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
            )
        elif "drive_report_data" in st.session_state and not REPORTLAB_AVAILABLE:
            st.info("To enable PDF downloads, install reportlab: pip install reportlab")


if __name__ == "__main__":
    main()