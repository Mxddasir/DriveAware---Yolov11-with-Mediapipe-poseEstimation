import numpy as np
from pathlib import Path
from typing import Tuple
from ultralytics import YOLO


class PhoneDetector:
    # runs YOLO to detect phones in a frame

    def __init__(self, model_path: str):
        # load the yolo model from the given path
        model_path = str(Path(model_path).expanduser().resolve())
        self.model = YOLO(model_path)

    def detect(
        self,
        frame_bgr: np.ndarray,
        confidence: float
    ) -> Tuple[object, np.ndarray, np.ndarray]:
        # run detection and return the result, bounding boxes, and confidence scores
        det = self.model.predict(frame_bgr, conf=confidence, verbose=False)[0]

        # if nothing was found return empty arrays
        if det.boxes is None or len(det.boxes) == 0:
            return (
                det,
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32)
            )

        boxes = det.boxes.xyxy.cpu().numpy().astype(np.float32)
        confs = det.boxes.conf.cpu().numpy().astype(np.float32)

        return det, boxes, confs