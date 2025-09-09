import cv2
import numpy as np
import mediapipe as mp
import time
import math


# ===============================
# One-Euro Filter Implementation
# ===============================
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


        dt = t - self.t_prev if self.t_prev else 1.0 / self.freq
        self.freq = 1.0 / dt if dt > 0 else self.freq
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


# Dictionary of filters for each angle
filters = {}
def get_filter(name):
    if name not in filters:
        filters[name] = OneEuroFilter(freq=30, mincutoff=1.0, beta=0.01, dcutoff=1.0)
    return filters[name]


# ===============================
# Initialize MediaPipe Pose
# ===============================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


pose = mp_pose.Pose(static_image_mode=False,
                    model_complexity=2,
                    smooth_landmarks=True,
                    enable_segmentation=False,
                    smooth_segmentation=True,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7)


# ===============================
# Helper Functions
# ===============================
def calculate_angle(a, b, c):
    try:
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        dot_product = np.dot(ba, bc)
        magnitude_ba = np.linalg.norm(ba)
        magnitude_bc = np.linalg.norm(bc)
        if magnitude_ba == 0 or magnitude_bc == 0:
            return None
        cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle_rad = np.arccos(cosine_angle)
        return np.degrees(angle_rad)
    except:
        return None


def get_landmark_coordinates(landmarks, image_width, image_height, landmark_idx):
    if (landmarks is None or
        landmark_idx >= len(landmarks.landmark) or
        not landmarks.landmark[landmark_idx].visibility > 0.5):
        return None
    lm = landmarks.landmark[landmark_idx]
    x, y = int(lm.x * image_width), int(lm.y * image_height)
    return (x, y)


def draw_angle_arc(img, point_a, point_b, point_c, color):
    a = np.array(point_a, dtype=np.float32)
    b = np.array(point_b, dtype=np.float32)
    c = np.array(point_c, dtype=np.float32)

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

    radius = 40
    center = (int(b[0]), int(b[1]))

    try:
        cv2.ellipse(img, center, (radius, radius), 0, start_angle, start_angle + sweep_angle, color, 3)
    except:
        pass


# ===============================
# Webcam Setup
# ===============================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


print("Starting pose detection... Press 'q' to quit.")


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue


    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)


    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )


        h, w, _ = frame.shape
        timestamp = time.time()


        angles_to_calculate = [
            ("R Elbow", [mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                         mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                         mp_pose.PoseLandmark.RIGHT_WRIST.value], (0, 255, 0)),
            ("L Elbow", [mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                         mp_pose.PoseLandmark.LEFT_ELBOW.value,
                         mp_pose.PoseLandmark.LEFT_WRIST.value], (0, 255, 0)),
            ("R Knee", [mp_pose.PoseLandmark.RIGHT_HIP.value,
                         mp_pose.PoseLandmark.RIGHT_KNEE.value,
                         mp_pose.PoseLandmark.RIGHT_ANKLE.value], (255, 0, 0)),
            ("L Knee", [mp_pose.PoseLandmark.LEFT_HIP.value,
                         mp_pose.PoseLandmark.LEFT_KNEE.value,
                         mp_pose.PoseLandmark.LEFT_ANKLE.value], (255, 0, 0)),
            ("R Shoulder", [mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                            mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                            mp_pose.PoseLandmark.RIGHT_HIP.value], (0, 0, 255)),
            ("L Shoulder", [mp_pose.PoseLandmark.LEFT_ELBOW.value,
                            mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                            mp_pose.PoseLandmark.LEFT_HIP.value], (0, 0, 255)),
            ("R Hip", [mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                       mp_pose.PoseLandmark.RIGHT_HIP.value,
                       mp_pose.PoseLandmark.RIGHT_KNEE.value], (255, 255, 0)),
            ("L Hip", [mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                       mp_pose.PoseLandmark.LEFT_HIP.value,
                       mp_pose.PoseLandmark.LEFT_KNEE.value], (255, 255, 0))
        ]


        y_offset = 30
        for angle_name, landmark_indices, color in angles_to_calculate:
            points = [get_landmark_coordinates(results.pose_landmarks, w, h, idx)
                      for idx in landmark_indices]


            if all(points):
                angle = calculate_angle(points[0], points[1], points[2])
                if angle is not None:
                    filt = get_filter(angle_name)
                    smooth_angle = filt.filter(timestamp, angle)


                    angle_text = f"{angle_name}: {smooth_angle:.1f} deg"
                    cv2.putText(frame, angle_text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y_offset += 25


                    joint_x, joint_y = points[1]
                    cv2.putText(frame, f"{smooth_angle:.0f} deg",
                               (joint_x + 15, joint_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Draw arc on joint to represent angle
                    draw_angle_arc(frame, points[0], points[1], points[2], color)


        cv2.putText(frame, "Ensure full body is visible for accurate detection",
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


    else:
        cv2.putText(frame, "No pose detected - Ensure you're visible in frame",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


    cv2.imshow('Pose Angle Detection with One-Euro Filter', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
print("Application closed.")
