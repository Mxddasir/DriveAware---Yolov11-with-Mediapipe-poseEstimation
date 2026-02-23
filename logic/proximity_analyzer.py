import numpy as np
from typing import List, Tuple, Optional

from utils.constants import (
    NOSE, MOUTH_L, MOUTH_R, L_EAR, R_EAR,
    L_WRIST, R_WRIST, L_INDEX, R_INDEX, L_THUMB, R_THUMB
)


class ProximityAnalyzer:
    # checks how close hands are to face and phone

    @staticmethod
    def _norm_dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        # distance between two normalised points
        return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))

    @staticmethod
    def _pix_dist(a_xy: Tuple[int, int], b_xy: Tuple[int, int]) -> float:
        # distance between two pixel points
        return float(np.sqrt((a_xy[0] - b_xy[0]) ** 2 + (a_xy[1] - b_xy[1]) ** 2))

    @staticmethod
    def hands_near_face(
        pose_result,
        hand_face_thresh: float
    ) -> Tuple[bool, Optional[Tuple[float, float]], float]:
        # returns True if any hand point is close enough to the face

        if not pose_result.pose_landmarks:
            return False, None, 999.0

        lm = pose_result.pose_landmarks[0]
        if len(lm) < 33:
            return False, None, 999.0

        # grab face landmark positions
        face_pts = [
            (lm[NOSE].x, lm[NOSE].y),
            (lm[MOUTH_L].x, lm[MOUTH_L].y),
            (lm[MOUTH_R].x, lm[MOUTH_R].y),
            (lm[L_EAR].x, lm[L_EAR].y),
            (lm[R_EAR].x, lm[R_EAR].y),
        ]

        # grab hand landmark positions (wrists, index fingers, thumbs)
        hand_candidates = [
            (lm[L_WRIST].x, lm[L_WRIST].y),
            (lm[R_WRIST].x, lm[R_WRIST].y),
            (lm[L_INDEX].x, lm[L_INDEX].y),
            (lm[R_INDEX].x, lm[R_INDEX].y),
            (lm[L_THUMB].x, lm[L_THUMB].y),
            (lm[R_THUMB].x, lm[R_THUMB].y),
        ]

        best_hand = None
        best_d = 999.0

        # find the closest hand-face pair
        for hp in hand_candidates:
            for fp in face_pts:
                d = ProximityAnalyzer._norm_dist(hp, fp)
                if d < best_d:
                    best_d = d
                    best_hand = hp

        if best_hand is None:
            return False, None, 999.0

        return (best_d <= hand_face_thresh), best_hand, best_d

    @staticmethod
    def get_hand_points_px(
        pose_result,
        img_w: int,
        img_h: int
    ) -> List[Tuple[int, int]]:
        # gets hand landmark positions in pixel coords

        if not pose_result.pose_landmarks:
            return []

        lm = pose_result.pose_landmarks[0]
        if len(lm) < 33:
            return []

        idxs = [L_WRIST, R_WRIST, L_INDEX, R_INDEX, L_THUMB, R_THUMB]
        pts = []
        for i in idxs:
            # convert normalised coords to pixels
            x = int(lm[i].x * img_w)
            y = int(lm[i].y * img_h)
            pts.append((x, y))

        return pts

    @staticmethod
    def phone_close_to_hand(
        phone_boxes_xyxy: np.ndarray,
        hand_pts_px: List[Tuple[int, int]],
        img_w: int,
        img_h: int,
        hand_phone_thresh: float
    ) -> Tuple[bool, float]:
        # checks if any phone bounding box centre is near a hand point

        if phone_boxes_xyxy.shape[0] == 0 or not hand_pts_px:
            return False, 999.0

        # convert normalised threshold to pixel distance
        norm_base = float(max(img_w, img_h))
        thresh_px = hand_phone_thresh * norm_base

        best_px = 999999.0
        for (x1, y1, x2, y2) in phone_boxes_xyxy:
            # get centre of the phone box
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            for hp in hand_pts_px:
                dpx = ProximityAnalyzer._pix_dist((cx, cy), hp)
                if dpx < best_px:
                    best_px = dpx

        best_norm = best_px / norm_base
        return (best_px <= thresh_px), float(best_norm)