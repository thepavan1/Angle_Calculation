import cv2
import numpy as np
import math
import time
from ultralytics import YOLO


class OneEuroFilter:
    def __init__(self, freq=30, mincutoff=1.0, beta=0.01, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def alpha(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def filter(self, t, x):
        if self.x_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x
        dt = max(1e-6, t - self.t_prev)
        self.freq = 1.0 / dt
        self.t_prev = t
        dx = (x - self.x_prev) * self.freq
        a_d = self.alpha(self.dcutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a = self.alpha(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


filters = {}
def get_filter(name):
    if name not in filters:
        filters[name] = OneEuroFilter()
    return filters[name]


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    dot_product = np.dot(ba, bc)
    mag_ba = np.linalg.norm(ba)
    mag_bc = np.linalg.norm(bc)
    if mag_ba == 0 or mag_bc == 0:
        return None
    cosine_angle = dot_product / (mag_ba * mag_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle


def draw_angle_arc(img, a, b, c, color):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    angle_ba = math.degrees(math.atan2(ba[1], ba[0])) % 360
    angle_bc = math.degrees(math.atan2(bc[1], bc[0])) % 360
    start_angle = min(angle_ba, angle_bc)
    end_angle = max(angle_ba, angle_bc)
    sweep_angle = end_angle - start_angle
    if sweep_angle > 180:
        start_angle = max(angle_ba, angle_bc)
        sweep_angle = 360 - sweep_angle
    radius = 30
    center = (int(b[0]), int(b[1]))
    try:
        cv2.ellipse(img, center, (radius, radius), 0, start_angle, start_angle + sweep_angle, color, 2)
    except:
        pass


# Load YOLOv8 pose model - adjust 'yolov8n-pose.pt' path as needed
model = YOLO('yolov8n-pose.pt')


# Keypoint indices for angles per YOLOv8's keypoint format (COCO-style)
angles_to_calculate = [
    ("L Elbow", [6, 8, 10], (0, 255, 0)),
    ("R Elbow", [5, 7, 9], (0, 255, 0)),
    ("L Knee", [12, 14, 16], (255, 0, 0)),
    ("R Knee", [11, 13, 15], (255, 0, 0)),
    ("L Shoulder", [8, 6, 12], (0, 0, 255)),
    ("R Shoulder", [7, 5, 11], (0, 0, 255)),
    ("L Hip", [6, 12, 14], (255, 255, 0)),
    ("R Hip", [5, 11, 13], (255, 255, 0)),
]


# --- UPDATED SKELETON CONNECTIONS FOR NEAT POSE ---
# Indices: [nose, l_eye, r_eye, l_ear, r_ear, l_shoulder, r_shoulder,
#           l_elbow, r_elbow, l_wrist, r_wrist, l_hip, r_hip, l_knee, r_knee, l_ankle, r_ankle]

# Only draw lines essential for neat limb and body connections
skeleton = [
    (6, 8),   # L Shoulder - L Elbow
    (8, 10),  # L Elbow - L Wrist
    (5, 7),   # R Shoulder - R Elbow
    (7, 9),   # R Elbow - R Wrist
    (6, 12),  # L Shoulder - L Hip
    (5, 11),  # R Shoulder - R Hip
    (12, 14), # L Hip - L Knee
    (14, 16), # L Knee - L Ankle
    (11, 13), # R Hip - R Knee
    (13, 15), # R Knee - R Ankle
    (6, 5),   # L Shoulder - R Shoulder
    (12, 11), # L Hip - R Hip
    (6, 5),   # Neck-Shoulder line for upper body
]
# ---------------------------------------------------


cap = cv2.VideoCapture(0)
print("Starting YOLOv8 pose detection. Press 'q' to quit.")

cv2.namedWindow("YOLOv8n Pose Angle Estimation", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("YOLOv8n Pose Angle Estimation", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    results = model(frame)[0]

    if results.keypoints is not None and len(results.keypoints) > 0:
        keypoints = results.keypoints[0].data.cpu().numpy()  # get raw tensor from Keypoints object
        if keypoints.ndim == 3:
            keypoints = keypoints[0]
        # keypoints shape: (num_keypoints, 3) [x, y, confidence]

        # Draw keypoints
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > 0.3:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 255), -1)

        # Draw updated neat skeleton lines
        for (i1, i2) in skeleton:
            if keypoints[i1][2] > 0.3 and keypoints[i2][2] > 0.3:
                pt1 = (int(keypoints[i1][0]), int(keypoints[i1][1]))
                pt2 = (int(keypoints[i2][0]), int(keypoints[i2][1]))
                cv2.line(frame, pt1, pt2, (255, 255, 255), thickness=2)

        # Draw angles and calculate
        timestamp = time.time()
        y_offset = 30
        for name, (ia, ib, ic), color in angles_to_calculate:
            a = keypoints[ia][:2]
            b = keypoints[ib][:2]
            c = keypoints[ic][:2]
            confs = [keypoints[pt][2] for pt in (ia, ib, ic)]
            # Only calculate if all confidence > 0.3
            if all([conf > 0.3 for conf in confs]):
                angle = calculate_angle(a, b, c)
                if angle is not None:
                    filt = get_filter(name)
                    smooth_angle = filt.filter(timestamp, angle)
                    # Text sidebar
                    cv2.putText(frame, f"{name}: {smooth_angle:.1f} deg", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_offset += 25
                    # Angle text near joint
                    cv2.putText(frame, f"{smooth_angle:.0f} deg", (int(b[0]) + 10, int(b[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    # Angle arc visualization
                    draw_angle_arc(frame, a, b, c, color)

    else:
        cv2.putText(frame, "No person detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("YOLOv8n Pose Angle Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
