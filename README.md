

# Angle Extraction Project

## Overview
This project provides real-time human joint angle extraction using **2D keypoints**. The main script, `app.py`, delivers accurate and smooth angle estimation from live webcam video, leveraging state-of-the-art pose detection and filtering techniques.

## Component Overview
- **MediaPipe Pose:** High-accuracy body landmark detection from video frames (2D only)
- **Vector Mathematics:** Joint angle computation using the arccosine formula on 2D vectors
- **One-Euro Filter:** Adaptive temporal smoothing for stable angle output
- **OpenCV:** Real-time video processing and visualization

## Main Features
- **2D joint angle extraction:** Robust angle estimation from 2D keypoints
- **Real-time processing:** Fast video capture and visualization
- **Temporal smoothing:** One-Euro filter reduces jitter and noise for smooth angle display
- **Visual feedback:** Angles and arcs are drawn directly on the video feed for intuitive understanding

## Usage

Run the main application:
```bash
python app.py
```

The application will open your webcam, detect body landmarks, compute joint angles (elbow, knee, shoulder, hip), and display them in real time with smooth filtering and visual overlays. Press 'q' to quit.

## Project Structure
- `app.py`: **Recommended main script** for 2D angle extraction (best results)
- `lift_and_angles.py`: Alternative script; less accurate than `app.py`
- `realtime_3d_angles.py`: Alternative real-time demo
- `angle-env/`: Python virtual environment (ignored in version control)

## Requirements
- Python 3.10+
- numpy
- opencv-python
- mediapipe

Install dependencies with:
```bash
pip install numpy opencv-python mediapipe
```

## Notes
- The project is designed for 2D keypoint-based angle extraction only.
- For best results, use `app.py` with a clear view of your full body in the webcam frame.

