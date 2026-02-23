# mediapipe blazepose landmark indices
# ref: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker

# face
NOSE = 0
L_EYE_INNER = 1
L_EYE = 2
L_EYE_OUTER = 3
R_EYE_INNER = 4
R_EYE = 5
R_EYE_OUTER = 6
L_EAR = 7
R_EAR = 8
MOUTH_L = 9
MOUTH_R = 10

# upper body
L_SHOULDER = 11
R_SHOULDER = 12
L_ELBOW = 13
R_ELBOW = 14
L_WRIST = 15
R_WRIST = 16
L_PINKY = 17
R_PINKY = 18
L_INDEX = 19
R_INDEX = 20
L_THUMB = 21
R_THUMB = 22

# hips
L_HIP = 23
R_HIP = 24

# same thing but as a dict for easier lookup
POSE_LANDMARKS = {
    "NOSE": NOSE,
    "L_EYE_INNER": L_EYE_INNER,
    "L_EYE": L_EYE,
    "L_EYE_OUTER": L_EYE_OUTER,
    "R_EYE_INNER": R_EYE_INNER,
    "R_EYE": R_EYE,
    "R_EYE_OUTER": R_EYE_OUTER,
    "L_EAR": L_EAR,
    "R_EAR": R_EAR,
    "MOUTH_L": MOUTH_L,
    "MOUTH_R": MOUTH_R,
    "L_SHOULDER": L_SHOULDER,
    "R_SHOULDER": R_SHOULDER,
    "L_ELBOW": L_ELBOW,
    "R_ELBOW": R_ELBOW,
    "L_WRIST": L_WRIST,
    "R_WRIST": R_WRIST,
    "L_PINKY": L_PINKY,
    "R_PINKY": R_PINKY,
    "L_INDEX": L_INDEX,
    "R_INDEX": R_INDEX,
    "L_THUMB": L_THUMB,
    "R_THUMB": R_THUMB,
    "L_HIP": L_HIP,
    "R_HIP": R_HIP,
}

# skeleton connections for drawing the pose overlay
POSE_CONNECTIONS = [
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 12),            # shoulders
    (11, 23), (12, 24),  # torso
    (23, 24),            # hips
    (23, 25), (25, 27), (27, 29), (29, 31),  # left leg
    (24, 26), (26, 28), (28, 30), (30, 32),  # right leg
]