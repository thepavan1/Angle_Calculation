import cv2
import mediapipe as mp
import numpy as np
import time
import math

# =====================================
# One-Euro Filter
# =====================================
class OneEuroFilter:
    def __init__(self, freq=30.0, mincutoff=1.0, beta=0.01, dcutoff=1.0):
        self.freq = float(freq)
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def alpha(self, cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
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
        filters[name] = OneEuroFilter(freq=30, mincutoff=1.0, beta=0.01, dcutoff=1.0)
    return filters[name]

# =====================================
# Pose Estimation (Mediapipe)
# =====================================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    dot = np.dot(ba, bc)
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9
    cosang = np.clip(dot / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

# =====================================
# Real-Time Webcam Loop
# =====================================
cap = cv2.VideoCapture(0)

print("Starting real-time full-body angle detection. Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    h, w, _ = frame.shape
    t = time.time()

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        lm = results.pose_landmarks.landmark

        def get_xy(idx):
            return (int(lm[idx].x * w), int(lm[idx].y * h))

        angles = {
            # Elbows
            "R_Elbow": (mp_pose.PoseLandmark.RIGHT_SHOULDER,
                        mp_pose.PoseLandmark.RIGHT_ELBOW,
                        mp_pose.PoseLandmark.RIGHT_WRIST),
            "L_Elbow": (mp_pose.PoseLandmark.LEFT_SHOULDER,
                        mp_pose.PoseLandmark.LEFT_ELBOW,
                        mp_pose.PoseLandmark.LEFT_WRIST),

            # Shoulders
            "R_Shoulder": (mp_pose.PoseLandmark.RIGHT_ELBOW,
                           mp_pose.PoseLandmark.RIGHT_SHOULDER,
                           mp_pose.PoseLandmark.RIGHT_HIP),
            "L_Shoulder": (mp_pose.PoseLandmark.LEFT_ELBOW,
                           mp_pose.PoseLandmark.LEFT_SHOULDER,
                           mp_pose.PoseLandmark.LEFT_HIP),

            # Knees
            "R_Knee": (mp_pose.PoseLandmark.RIGHT_HIP,
                       mp_pose.PoseLandmark.RIGHT_KNEE,
                       mp_pose.PoseLandmark.RIGHT_ANKLE),
            "L_Knee": (mp_pose.PoseLandmark.LEFT_HIP,
                       mp_pose.PoseLandmark.LEFT_KNEE,
                       mp_pose.PoseLandmark.LEFT_ANKLE),

            # Hips
            "R_Hip": (mp_pose.PoseLandmark.RIGHT_SHOULDER,
                      mp_pose.PoseLandmark.RIGHT_HIP,
                      mp_pose.PoseLandmark.RIGHT_KNEE),
            "L_Hip": (mp_pose.PoseLandmark.LEFT_SHOULDER,
                      mp_pose.PoseLandmark.LEFT_HIP,
                      mp_pose.PoseLandmark.LEFT_KNEE)
        }

        y_offset = 30
        for name, (a_idx, b_idx, c_idx) in angles.items():
            a, b, c = get_xy(a_idx), get_xy(b_idx), get_xy(c_idx)
            if a and b and c:
                raw_angle = calculate_angle(a, b, c)
                smooth_angle = get_filter(name).filter(t, raw_angle)

                # Display on left sidebar
                cv2.putText(frame, f"{name}: {smooth_angle:.1f} deg",
                            (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0,255,0), 2)
                y_offset += 25

                # Display near the joint
                cv2.putText(frame, f"{smooth_angle:.0f} deg",
                            (b[0]+10, b[1]-10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255,255,255), 2)

    else:
        cv2.putText(frame, "No pose detected",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (0,0,255), 2)

    cv2.imshow("Real-Time Angles (All Major Joints)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
