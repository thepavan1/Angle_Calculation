import cv2
import mediapipe as mp
import numpy as np
import time
import math
import sys


# ---------------------------
# One-Euro Filter
# ---------------------------
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
        # t in seconds, x scalar
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
        # store
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


# helper container to reuse per-angle filters
filters = {}
def get_filter(name):
    if name not in filters:
        filters[name] = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.01, dcutoff=1.0)
    return filters[name]


# ---------------------------
# AJCS helpers and Euler extraction
# ---------------------------
def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v * 0.0
    return v / n


def build_segment_frame(prox, dist, ref):
    """
    Build a right-handed local frame for a segment.
    prox, dist, ref: np.array([x,y,z]) in meters
    returns R (3x3) with columns [x_axis, y_axis, z_axis]
    """
    x = normalize(dist - prox)   # primary axis along segment (prox->dist)
    v = ref - prox
    cross = np.cross(x, v)
    if np.linalg.norm(cross) < 1e-6:
        # fallback: perturb ref slightly
        v = v + np.array([1e-3, 0, 0])
        cross = np.cross(x, v)
    z = normalize(cross)
    y = np.cross(z, x)
    R = np.column_stack((x, y, z))
    return R


def relative_rotation(R_parent, R_child):
    return R_parent.T @ R_child


def rotation_matrix_to_euler_xyz(R):
    """
    Extract Euler angles in X->Y->Z rotation order (radians).
    Return degrees [rx, ry, rz].
    """
    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6
    if not singular:
        rx = math.atan2(R[2,1], R[2,2])
        ry = math.atan2(-R[2,0], sy)
        rz = math.atan2(R[1,0], R[0,0])
    else:
        rx = math.atan2(-R[1,2], R[1,1])
        ry = math.atan2(-R[2,0], sy)
        rz = 0.0
    return np.degrees([rx, ry, rz])


# ---------------------------
# Utility: convert Mediapipe landmark to metric 3D
# ---------------------------
class Calibration:
    """
    Simple calibration: user enters their height (m) while standing visible to camera.
    We compute pixel torso length (neck to mid-hip) and derive meters-per-pixel scale.
    We apply same scale to x,y and a separate scale to mediapipe z (relative).
    """
    def __init__(self):
        self.mpp = None      # meters per pixel (x,y)
        self.z_scale = 1.0   # scale applied to mediapipe z
        self.calibrated = False


    def calibrate_from_landmarks(self, lm, W, H, user_height_m):
        try:
            neck_x = (lm[11].x + lm[12].x) / 2.0  # mid-shoulder x
            neck_y = (lm[11].y + lm[12].y) / 2.0  # mid-shoulder y
            neck_px = np.array([neck_x * W, neck_y * H])
            lhip_px = np.array([lm[23].x * W, lm[23].y * H])
            rhip_px = np.array([lm[24].x * W, lm[24].y * H])
            midhip_px = (lhip_px + rhip_px) / 2.0
            torso_px = np.linalg.norm(neck_px - midhip_px)
            if torso_px < 5:
                print("Calibration failed: torso pixel length too small.")
                return False
            torso_m = 0.285 * user_height_m  # anthropometric approx
            self.mpp = torso_m / torso_px
            self.z_scale = self.mpp * 1.0
            self.calibrated = True
            print(f"Calibration OK: meters-per-pixel = {self.mpp:.4f} m/px, z_scale = {self.z_scale:.4f}")
            return True
        except Exception as e:
            print("Calibration exception:", e)
            return False


calib = Calibration()


# ---------------------------
# Draw angle arc function
# ---------------------------
def draw_angle_arc(img, pt_a, pt_b, pt_c, color):
    """
    Draw an arc representing the angle at pt_b formed by pt_a and pt_c.
    Args:
        img: OpenCV image
        pt_a, pt_b, pt_c: (x,y) pixel coords
        color: (B,G,R)
    """
    a = np.array(pt_a, dtype=np.float32)
    b = np.array(pt_b, dtype=np.float32)
    c = np.array(pt_c, dtype=np.float32)

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
    except Exception:
        pass


# ---------------------------
# Mediapipe init
# ---------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                    enable_segmentation=False, min_detection_confidence=0.6,
                    min_tracking_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils


# ---------------------------
# Main realtime loop
# ---------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera"); sys.exit(1)


print("Realtime 3D-angle pipeline using Mediapipe. Press 'c' to calibrate (stand straight), 'q' to quit.")


last_time = time.time()
cx = cy = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # mirror
    H, W, _ = frame.shape
    cx, cy = W / 2.0, H / 2.0
    t = time.time()
    fps = 1.0 / (t - last_time) if (t - last_time) > 0 else 0.0
    last_time = t

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark  # 33 landmarks (x,y,z,visibility)
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if not calib.calibrated:
            cv2.putText(frame, "Press 'c' to calibrate (stand straight & enter height)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(frame, f"Calibrated (m/px={calib.mpp:.4f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        joints3d = []
        for ld in lm:
            x_px = ld.x * W
            y_px = ld.y * H
            z_rel = ld.z  # relative z from mediapipe (negative forward)
            if calib.calibrated and calib.mpp is not None:
                xm = (x_px - cx) * calib.mpp
                ym = (cy - y_px) * calib.mpp  # flip y so up positive
                zm = z_rel * calib.z_scale
                joints3d.append(np.array([xm, ym, zm]))
            else:
                xm = x_px - cx
                ym = cy - y_px
                zm = z_rel * 100.0
                joints3d.append(np.array([xm, ym, zm]))

        def J(idx):
            try:
                return joints3d[idx]
            except:
                return None

        # Mediapipe landmark indices for joints:
        L_SHO, R_SHO = 11, 12
        L_ELB, R_ELB = 13, 14
        L_WR, R_WR = 15, 16
        L_HIP, R_HIP = 23, 24
        L_KNE, R_KNE = 25, 26
        L_ANK, R_ANK = 27, 28

        angle_defs = {
            "R_Elbow": (R_SHO, R_ELB, R_WR),
            "L_Elbow": (L_SHO, L_ELB, L_WR),
            "R_Shoulder": (R_ELB, R_SHO, R_HIP),
            "L_Shoulder": (L_ELB, L_SHO, L_HIP),
            "R_Knee": (R_HIP, R_KNE, R_ANK),
            "L_Knee": (L_HIP, L_KNE, L_ANK),
            "R_Hip": (R_SHO, R_HIP, R_KNE),
            "L_Hip": (L_SHO, L_HIP, L_KNE)
        }

        sidebar_y = 60
        for name, (ia, ib, ic) in angle_defs.items():
            pa = J(ia)
            pb = J(ib)
            pc = J(ic)
            if pa is None or pb is None or pc is None:
                continue

            # 1) geometric angle (arccos) always positive between 0-180
            ba = pa - pb
            bc = pc - pb
            denom = (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
            cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
            raw_ang = math.degrees(math.acos(cosang))

            # 2) AJCS frames (parent & child)
            midhip = (J(L_HIP) + J(R_HIP)) / 2.0 if (J(L_HIP) is not None and J(R_HIP) is not None) else pb + np.array([0.0, -0.2, 0.0])
            try:
                R_parent = build_segment_frame(pa, pb, midhip)
                R_child = build_segment_frame(pb, pc, pa)
                R_rel = relative_rotation(R_parent, R_child)
                rx, ry, rz = rotation_matrix_to_euler_xyz(R_rel)  # degrees
            except Exception:
                rx = ry = rz = float('nan')

            # Normalize rx to positive angle bound [0,180]
            rx_abs = abs(rx)
            if rx_abs > 180:
                rx_abs = 360 - rx_abs

            # 3) Smooth each Euler component
            f_rx = get_filter(name + "_rx").filter(t, rx_abs)
            f_ry = get_filter(name + "_ry").filter(t, ry)
            f_rz = get_filter(name + "_rz").filter(t, rz)

            # 4) Display filtered flexion angle
            cv2.putText(frame, f"{name}: {f_rx:.1f} deg", (10, sidebar_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            sidebar_y += 22

            # Pixel locations for drawing angle label and arc
            pb_px = (int((pb[0] / (calib.mpp if calib.calibrated else 1.0) + cx)),
                     int((cy - pb[1] / (calib.mpp if calib.calibrated else 1.0))))
            pa_px = (int((pa[0] / (calib.mpp if calib.calibrated else 1.0) + cx)),
                     int((cy - pa[1] / (calib.mpp if calib.calibrated else 1.0))))
            pc_px = (int((pc[0] / (calib.mpp if calib.calibrated else 1.0) + cx)),
                     int((cy - pc[1] / (calib.mpp if calib.calibrated else 1.0))))

            # Display raw geometric angle near joint
            try:
                cv2.putText(frame, f"{raw_ang:.0f} deg", (pb_px[0] + 8, pb_px[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            except:
                pass

            # Draw arc to visualize angle
            draw_angle_arc(frame, pa_px, pb_px, pc_px, (0, 255, 0))

        # Show FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (W - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 50), 2)

    else:
        cv2.putText(frame, "No person detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Realtime 3D Angles (Mediapipe)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        print("\nCalibration requested. Stand straight facing camera. Enter your height in meters (e.g., 1.70):")
        try:
            height_m = float(input("Height (m): ").strip())
        except:
            print("Invalid height. Calibration cancelled.")
            continue
        if results and results.pose_landmarks:
            ok = calib.calibrate_from_landmarks(results.pose_landmarks.landmark, W, H, height_m)
            if not ok:
                print("Calibration failed — try again with clearer pose (shoulders visible and roughly vertical).")
        else:
            print("No pose to calibrate from - make sure your body is visible.")

# cleanup
cap.release()
cv2.destroyAllWindows()
