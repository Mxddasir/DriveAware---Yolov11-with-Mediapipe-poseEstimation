import numpy as np
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.core.image import Image as MpImage, ImageFormat as MpImageFormat


class PoseEstimator:
    # uses mediapipe blazepose to get body landmarks from a frame

    def __init__(self, task_path: str, running_mode: vision.RunningMode):
        # set up the pose landmarker with the given task file
        task_path = str(Path(task_path).expanduser().resolve())
        base_options = python.BaseOptions(model_asset_path=task_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=running_mode,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.running_mode = running_mode

    def detect_image(self, frame_rgb: np.ndarray):
        # run pose detection on a single image
        mp_image = MpImage(image_format=MpImageFormat.SRGB, data=frame_rgb)
        return self.landmarker.detect(mp_image)

    def detect_video(self, frame_rgb: np.ndarray, timestamp_ms: int):
        # run pose detection on a video frame (needs a timestamp)
        mp_image = MpImage(image_format=MpImageFormat.SRGB, data=frame_rgb)
        return self.landmarker.detect_for_video(mp_image, timestamp_ms)

    def close(self):
        # clean up the landmarker
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None

    # context manager so we can use "with PoseEstimator(...) as pe:"
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()