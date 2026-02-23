from dataclasses import dataclass
from pathlib import Path


@dataclass
class DetectionConfig:
    # paths to model files
    phone_model_path: str = "weights/best.pt"
    pose_task_path: str = "pose_landmarker_full.task"

    # confidence thresholds for phone detection
    conf_high: float = 0.75
    conf_low: float = 0.25

    # how close hand needs to be to face/phone (normalised)
    hand_face_thresh: float = 0.18
    hand_phone_thresh: float = 0.12

    # extra options
    require_hand_proximity: bool = False
    draw_pose: bool = False

    def __post_init__(self):
        # turn the paths into full absolute paths so nothing breaks
        self.phone_model_path = str(Path(self.phone_model_path).expanduser().resolve())
        self.pose_task_path = str(Path(self.pose_task_path).expanduser().resolve())