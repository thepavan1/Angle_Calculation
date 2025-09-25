


# Angle Calculation Project

## Overview
This project provides real-time human joint angle extraction from video using both MediaPipe and YOLOv8 pose estimation. The main focus is on accurate, smooth 2D angle estimation, with an experimental 3D mode.

## Main Features

- **2D Angle Extraction (MediaPipe):**  
	- Real-time webcam processing (`app.py`)
	- Accurate detection of elbows, knees, shoulders, and hips
	- One-Euro Filter for smooth angle output
	- Visual feedback with angle overlays and arcs

- **2D Angle Extraction (YOLOv8):**  
	- Uses YOLOv8 for pose detection (`appyolo.py`)
	- Same angle calculations and smoothing as above
	- Custom skeleton drawing

- **Experimental 3D Angle Extraction:**  
	- Uses MediaPipe 3D landmarks (`realtime_3d_angles.py`)
	- Calibration step for metric scaling
	- Euler angle extraction for joints
	- Visualizes 3D angles (less robust than 2D)

- **Camera Stream Test:**  
	- `test.py` displays frames from a phone camera stream

## Usage

### 2D Angle Extraction (MediaPipe)
```bash
python app.py
```


### 2D Angle Extraction (YOLOv8)
```bash
python appyolo.py
```

### 2D Angle Extraction from IP Webcam (YOLOv8)
- **Script:** `appyoloipcam.py`
- **Description:** Connects to an IP webcam (e.g., Android IP Webcam app) via HTTP snapshot, runs YOLOv8 pose estimation, calculates and smooths joint angles, overlays results, and saves processed frames every 3 seconds.
- **Usage:**
	```bash
	python appyoloipcam.py
	```
- **Requirements:** IP webcam accessible at the configured URL (default: `http://192.168.1.248:8080/shot.jpg`)
- **Output:** Processed frames saved in the `processed_frames` directory.

### 3D Angle Extraction (Experimental)
```bash
python realtime_3d_angles.py
```

### Camera Stream Test
```bash
python test.py
```

## Requirements

- Python 3.10+
- numpy
- opencv-python
- mediapipe
- ultralytics (for YOLOv8)
- (optional) requests (for test.py)

Install dependencies with:
```bash
pip install numpy opencv-python mediapipe ultralytics requests
```

## Notes

- For best results, use `app.py` for 2D angle extraction.
- YOLOv8 model files (`.pt`) are tracked with Git LFS and should not exceed 100 MB.
- 3D angle extraction is experimental and may be less accurate.
- Large files and model weights are ignored by `.gitignore`.

