# Angle Extraction Project

## Overview
This project provides tools for extracting 3D joint angles from 2D keypoints, with a focus on human pose estimation using deep learning. The main script, `app.py`, offers the most accurate and robust results compared to other approaches in the repository.

## Main Features
- **Best-in-class 3D angle extraction:** The `app.py` script is the recommended entry point, providing reliable and accurate 3D joint angle estimation from 2D keypoints.
- **OpenPose 2D keypoint support:** Easily process OpenPose JSON outputs in COCO format.
- **VideoPose3D integration:** Lift 2D keypoints to 3D using pretrained deep learning models from the [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) repository.
- **Multiple modes:** Choose between naive lifting and advanced deep learning-based lifting (`videopose3d` mode).
- **Batch processing:** Efficiently process folders of JSON files and export results to CSV.
- **Angle calculation:** Compute joint angles from estimated 3D coordinates.
- **Command-line interface:** Flexible CLI for specifying input/output paths, model checkpoints, and processing modes.

## Usage

### Example Command
```bash
python app.py \
    --json_dir path/to/openpose_jsons \
    --out_csv out_angles.csv \
    --mode videopose3d \
    --vp3d_ckpt checkpoint/pretrained_h36m_detectron_coco.bin
```

### Arguments
- `--json_dir`: Path to folder containing OpenPose JSONs.
- `--out_csv`: Output CSV file for angles.
- `--mode`: Lifting mode (`naive` or `videopose3d`).
- `--vp3d_ckpt`: Path to VideoPose3D checkpoint (required for `videopose3d` mode).

## Project Structure
- `app.py`: **Recommended main script** for angle extraction.
- `lift_and_angles.py`: Alternative script; less accurate than `app.py`.
- `realtime_3d_angles.py`: Real-time demo; not as robust or accurate.
- `VideoPose3D/`: Contains the VideoPose3D repo for deep learning-based lifting.
- `angle-env/`: Python virtual environment (ignored in version control).

## Requirements
- Python 3.10+
- PyTorch
- numpy
- matplotlib
- tqdm
- opencv-python
- scipy
- absl-py
- pyyaml

Install dependencies with:
```bash
pip install torch numpy matplotlib tqdm opencv-python scipy absl-py pyyaml
```

## Notes
- For best results, use `app.py` with the `videopose3d` mode and a suitable pretrained checkpoint.
- The real-time script (`realtime_3d_angles.py`) is experimental and less accurate.
- The naive lifting mode is provided for comparison and simple use cases.

## License
See `VideoPose3D/LICENSE` for third-party code licensing.
