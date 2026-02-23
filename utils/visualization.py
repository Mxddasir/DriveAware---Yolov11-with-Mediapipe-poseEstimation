import cv2
import numpy as np

from .constants import POSE_CONNECTIONS


class Visualizer:
    # draws pose skeletons and status info on frames

    @staticmethod
    def draw_pose_landmarks(
        img_bgr: np.ndarray,
        pose_result,
        draw_connections: bool = True
    ) -> np.ndarray:
        # draws the pose dots and skeleton lines on the image

        if not pose_result.pose_landmarks:
            return img_bgr

        h, w = img_bgr.shape[:2]
        for landmarks in pose_result.pose_landmarks:
            # convert normalised coords to pixel positions
            pts = []
            for lm in landmarks:
                pts.append((int(lm.x * w), int(lm.y * h)))

            # draw each landmark as a green dot
            for (x, y) in pts:
                cv2.circle(img_bgr, (x, y), 3, (0, 255, 0), -1)

            # draw lines between connected joints
            if draw_connections:
                for a, b in POSE_CONNECTIONS:
                    if a < len(pts) and b < len(pts):
                        cv2.line(img_bgr, pts[a], pts[b], (0, 255, 255), 2)

        return img_bgr

    @staticmethod
    def annotate_status(
        img_bgr: np.ndarray,
        active: bool,
        suspicious: bool,
        used_conf: float,
        reason: str,
        face_dist: float,
        hand_dist: float
    ) -> np.ndarray:
        # draws a black bar at the top with detection status info

        h, w = img_bgr.shape[:2]
        status = "Phone Use: ACTIVE" if active else "Phone Use: NOT ACTIVE"
        mode = "SUSPICIOUS" if suspicious else "NORMAL"

        txt1 = f"{status} | {mode} | conf={used_conf:.2f}"
        txt2 = f"{reason} | face_d={face_dist:.3f} | hand_d={hand_dist:.3f}"

        # black rectangle behind the text so its readable
        cv2.rectangle(img_bgr, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.putText(img_bgr, txt1, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img_bgr, txt2, (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)

        return img_bgr