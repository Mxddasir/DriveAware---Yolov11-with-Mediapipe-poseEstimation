import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

from models.phone_detector import PhoneDetector
from models.pose_estimator import PoseEstimator
from logic.proximity_analyzer import ProximityAnalyzer
from logic.decision_engine import DecisionEngine
from utils.visualization import Visualizer
from mediapipe.tasks.python import vision


class ImageProcessor:
    # handles the full detection pipeline for a single image

    def __init__(
        self,
        phone_detector: PhoneDetector,
        pose_estimator: Optional[PoseEstimator] = None
    ):
        self.phone_detector = phone_detector
        self.pose_estimator = pose_estimator

    def process(
        self,
        image_path: str,
        pose_task_path: str,
        conf_high: float,
        conf_low: float,
        hand_face_thresh: float,
        hand_phone_thresh: float,
        require_hand_proximity: bool,
        draw_pose: bool,
        save_path: Optional[str] = None
    ) -> Tuple[np.ndarray, bool]:

        frame_bgr = cv2.imread(image_path)
        if frame_bgr is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        H, W = frame_bgr.shape[:2]

        # first try detecting phone with high confidence
        det_high, boxes_high, _ = self.phone_detector.detect(frame_bgr, conf_high)
        found_high = boxes_high.shape[0] > 0

        # set up defaults before we check pose
        suspicious = False
        face_d = 999.0
        hand_pts_px = []
        hand_ok = False
        hand_d = 999.0
        used_conf = conf_high
        det_to_plot = det_high

        # only bother with pose if we need it (drawing, hand filter, or no high-conf detection)
        need_pose = draw_pose or require_hand_proximity or (not found_high)
        pose_result = None

        if need_pose:
            # create a pose estimator if one wasnt passed in
            if self.pose_estimator is None:
                estimator = PoseEstimator(pose_task_path, vision.RunningMode.IMAGE)
            else:
                estimator = self.pose_estimator

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pose_result = estimator.detect_image(frame_rgb)

            # check if hands are near face (suspicious behaviour)
            suspicious, _best_hand_norm, face_d = ProximityAnalyzer.hands_near_face(
                pose_result, hand_face_thresh
            )
            hand_pts_px = ProximityAnalyzer.get_hand_points_px(pose_result, W, H)

            # clean up if we made our own estimator
            if self.pose_estimator is None:
                estimator.close()

        if found_high:
            # phone found at high conf - check hand proximity if needed
            if require_hand_proximity and hand_pts_px:
                hand_ok, hand_d = ProximityAnalyzer.phone_close_to_hand(
                    boxes_high, hand_pts_px, W, H, hand_phone_thresh
                )
            det_to_plot = det_high
            used_conf = conf_high
            found_low = False
        else:
            # no high conf hit - try low conf only if pose looks suspicious
            if suspicious:
                det_low, boxes_low, _ = self.phone_detector.detect(frame_bgr, conf_low)
                found_low = boxes_low.shape[0] > 0
                det_to_plot = det_low
                used_conf = conf_low

                if require_hand_proximity and hand_pts_px and found_low:
                    hand_ok, hand_d = ProximityAnalyzer.phone_close_to_hand(
                        boxes_low, hand_pts_px, W, H, hand_phone_thresh
                    )
            else:
                found_low = False

        # make the final decision
        active, reason = DecisionEngine.decide_phone_use(
            found_high=found_high,
            found_low=found_low,
            suspicious=suspicious,
            hand_ok=hand_ok,
            require_hand_proximity=require_hand_proximity,
        )

        # draw everything on the image
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

        # save if a path was given
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(save_path, annotated)

        return annotated, active