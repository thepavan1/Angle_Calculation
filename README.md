
# Angle Extraction Project

## Overview
This project implements real-time human joint angle extraction using **2D keypoints only**. While there is an experimental attempt at 3D angle calculation using a single webcam, it is not robust and tends to be jittery. The recommended and most accurate results are obtained from the main script, `app.py`, which uses 2D pose estimation and advanced filtering for smooth angle computation.

## Component Overview
- **MediaPipe Pose:** High-accuracy body landmark detection from video frames (2D only)
- **Vector Mathematics:** Joint angle computation using the arccosine formula on 2D vectors
- **One-Euro Filter:** Adaptive temporal smoothing for stable angle output
- **OpenCV:** Real-time video processing and visualization

## Main Features
- **2D joint angle extraction:** Accurate and robust angle estimation from 2D keypoints
- **Real-time processing:** Fast video capture and visualization
- **Temporal smoothing:** One-Euro filter reduces jitter and noise
- **Command-line interface:** Flexible CLI for specifying input/output paths and processing modes

## Usage

### Example Command
```bash
python app.py \
    --json_dir path/to/openpose_jsons \
    --out_csv out_angles.csv \
    --mode mediapipe
```

### Arguments
- `--json_dir`: Path to folder containing pose estimation JSONs
- `--out_csv`: Output CSV file for angles
- `--mode`: Processing mode (`mediapipe` for 2D, `naive` for simple math, `videopose3d` for experimental 3D)

## Project Structure
- `app.py`: **Recommended main script** for 2D angle extraction (best results)
- `lift_and_angles.py`: Alternative script; less accurate than `app.py`
- `realtime_3d_angles.py`: Experimental real-time 3D demo; not robust or accurate
- `VideoPose3D/`: Contains the VideoPose3D repo (for experimental 3D lifting)
- `angle-env/`: Python virtual environment (ignored in version control)

## Requirements
- Python 3.10+
- numpy
- matplotlib
- tqdm
- opencv-python
- mediapipe

Install dependencies with:
```bash
pip install numpy matplotlib tqdm opencv-python mediapipe
```

## Notes
- The project is designed for 2D keypoint-based angle extraction. 3D calculation using a single webcam is experimental and not recommended for production use.
- For best results, use `app.py` with the default 2D mode and One-Euro filter enabled.
- The real-time script (`realtime_3d_angles.py`) is experimental and less accurate.

## License
See `VideoPose3D/LICENSE` for third-party code licensing.
