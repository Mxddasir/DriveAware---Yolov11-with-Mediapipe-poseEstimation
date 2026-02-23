import cv2
import numpy as np
import time
from pathlib import Path
from typing import Optional

from models.phone_detector import PhoneDetector
from models.pose_estimator import PoseEstimator
from logic.proximity_analyzer import ProximityAnalyzer
from logic.decision_engine import DecisionEngine
from utils.visualization import Visualizer
from mediapipe.tasks.python import vision


class VideoProcessor:
    # processes video files or webcam feed for phone detection

    def __init__(
        self,
        phone_detector: PhoneDetector,
        pose_estimator: Optional[PoseEstimator] = None
    ):
        self.phone_detector = phone_detector
        self.pose_estimator = pose_estimator

    def process(
        self,
        source,
        pose_task_path: str,
        conf_high: float,
        conf_low: float,
        hand_face_thresh: float,
        hand_phone_thresh: float,
        require_hand_proximity: bool,
        draw_pose: bool,
        save_path: Optional[str] = None
    ):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1:
            fps = 30.0

        landmarker = None
        running_mode = vision.RunningMode.VIDEO

        writer = None
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        frame_idx = 0
        t0 = time.time()

        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                H, W = frame_bgr.shape[:2]

                # try high confidence detection first
                det_high, boxes_high, _confs_high = self.phone_detector.detect(frame_bgr, conf_high)
                found_high = boxes_high.shape[0] > 0

                # defaults
                suspicious = False
                face_d = 999.0
                hand_pts_px = []
                hand_ok = False
                hand_d = 999.0
                used_conf = conf_high
                det_to_plot = det_high

                # only run pose if we actually need it
                need_pose = draw_pose or require_hand_proximity or (not found_high)
                pose_result = None

                if need_pose:
                    # create pose estimator on first use (lazy init)
                    if landmarker is None:
                        landmarker = PoseEstimator(pose_task_path, running_mode)

                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    timestamp_ms = int((frame_idx / fps) * 1000)
                    pose_result = landmarker.detect_video(frame_rgb, timestamp_ms)

                    # check if hands are near face
                    suspicious, _best_hand_norm, face_d = ProximityAnalyzer.hands_near_face(
                        pose_result, hand_face_thresh
                    )
                    hand_pts_px = ProximityAnalyzer.get_hand_points_px(pose_result, W, H)

                if found_high:
                    # high conf phone found - optionally check hand proximity
                    if require_hand_proximity and hand_pts_px:
                        hand_ok, hand_d = ProximityAnalyzer.phone_close_to_hand(
                            boxes_high, hand_pts_px, W, H, hand_phone_thresh
                        )
                    det_to_plot = det_high
                    used_conf = conf_high
                    found_low = False
                else:
                    # fallback: try low conf if pose is suspicious
                    if suspicious:
                        det_low, boxes_low, _confs_low = self.phone_detector.detect(frame_bgr, conf_low)
                        found_low = boxes_low.shape[0] > 0
                        det_to_plot = det_low
                        used_conf = conf_low

                        if require_hand_proximity and hand_pts_px and found_low:
                            hand_ok, hand_d = ProximityAnalyzer.phone_close_to_hand(
                                boxes_low, hand_pts_px, W, H, hand_phone_thresh
                            )
                    else:
                        found_low = False
                        det_to_plot = det_high
                        used_conf = conf_high

                # final decision
                active, reason = DecisionEngine.decide_phone_use(
                    found_high=found_high,
                    found_low=found_low,
                    suspicious=suspicious,
                    hand_ok=hand_ok,
                    require_hand_proximity=require_hand_proximity,
                )

                # draw annotations on frame
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

                # show fps on screen
                dt = time.time() - t0
                if dt > 0:
                    fps_now = (frame_idx + 1) / dt
                    cv2.putText(
                        annotated, f"FPS: {fps_now:.1f}", (15, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
                    )

                # set up video writer on first frame if saving
                if save_path and writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
                if writer is not None:
                    writer.write(annotated)

                cv2.imshow("Phone + MediaPipe Tasks (Adaptive + Hand Proximity)", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                frame_idx += 1

        finally:
            # always clean up everything
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
            if landmarker is not None:
                landmarker.close()