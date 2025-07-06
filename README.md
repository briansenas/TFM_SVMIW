# Single View Metrology in the Wild Master's Thesis.
## 📚 Table of Contents

- [Overview](#-camera-tools)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Extract Frames](#-1-extract-frames)
  - [Calibrate Camera](#-2-calibrate-camera)
  - [Estimate Human Height](#-3-estimate-human-height)
  - [Cut Video](#-4-cut-video)
  - [Extract Realsense Frames](#-5-extracting-frames-from-realsense-bag-files)
- [Notes](#-notes)
- [License](#-license)

## 🎥 Camera Tools

This Python project provides a modular command-line toolset for handling video-based camera workflows such as:

- Extracting frames from a video using `ffmpeg`
- Calibrating a camera using chessboard patterns
- Estimating human height using YOLO and camera parameters
- Cutting video clips between timestamps

Each feature is implemented as a standalone script with a `register_subparser()` function for CLI integration.

---

## 📁 Project Structure

```plaintext
.
├── data
│   ├── cam1-cut_frames         # Extracted frames using extract-frames
│   ├── cam1-cut.mkv            # Original footageg
│   ├── cam1.mkv                # Cut footage using cut-video.py
│   ├── extrinsics              # Explicit extrinsics from the camera
│   └── intrinsics              # Automatically generated intrinsics using calibrate-camera
├── Document/                   # Documentation regarding the project outcome and details
├── LICENSE
├── pyproject.toml
├── README.md
├── scripts/                   # Folder for auto-loaded modules
│   ├── extract_frames.py          # Extract frames from a video
│   ├── calibrate_camera.py        # Calibrate camera using chessboard frames
│   ├── estimate_height.py         # Estimate person height using YOLO + calibration
│   ├── cut_video.py               # Cut video segments using ffmpeg
├── main.py                    # Entry point if you wish to unify all commands
├── uv.lock
└── yolov8n.pt
````

---

## ⚙️ Installation

We recommend using [`uv`](https://github.com/astral-sh/uv), a fast Python package manager and virtual environment tool.

### 1. Install `uv`

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

### 2. Set up project

```bash
uv venv
uv pip install -r requirements.txt
```

> Or to install manually:
>
> ```bash
> pip install -r requirements.txt
> ```

---

## 🚀 Usage

Each script is executable from the command line and includes its own subparser for integration.

### ✅ 1. Extract Frames

Extract frames from a video at a given frame rate.

```bash
python extract_frames.py extract-frames \
  --input path/to/video.mp4 \
  --rate 1
```

This will save one frame per second in a new folder next to the video.

---

### ✅ 2. Calibrate Camera

Calibrate a camera using extracted chessboard images.

```bash
python calibrate_camera.py calibrate-camera \
  --input path/to/frames \
  --board-height 6 --board-width 6 \
  --output calibration/intrinsics.yaml
```

You’ll get an intrinsics YAML file with camera matrix and distortion coefficients.

---

### ✅ 3. Estimate Human Height

Estimate the height of a visible person in an image using YOLO and calibration data.

```bash
python estimate_height.py estimate-height \
  --image frames/img001.png \
  --intrinsics calibration/intrinsics.yaml \
  --extrinsics calibration/extrinsics_config.yaml \
  --model yolov8n.pt
```

Make sure the person is fully visible and matching the extrinsics details (distance from the camera).

**Extrinsics config:**

```yaml
camera_height: 1.6              # meters
camera_pitch: 30.0              # meters
distance_to_object: 3.0         # meters
```

---

### ✅ 4. Cut Video

Trim a video between start and end timestamps.

```bash
python cut_video.py cut-video \
  --input video.mp4 \
  --start 00:01:30 \
  --end 00:03:00
```

Outputs a new file named `video-cut.mp4`.

---

### ✅ 5. Extracting Frames from RealSense `.bag` Files

This script allows you to extract color frames from a **RealSense `.bag` file** recorded with Intel RealSense cameras. You can control:

```bash
python extract_realsense_frames.py extract-realsense-frames \
  --bag path/to/your_file.bag \
  --output path/to/output_dir \
  --rate 1.0 \
  --start 00:00:10 \
  --end 00:00:30
```
This command extracts one frame every 0.5 seconds between 5 and 20 seconds of the video.

## 🧠 Notes

* Make sure `ffmpeg` is installed and accessible in your system path. For linux: `sudo apt install ffmpeg`.
* The YOLO model defaults to `yolov8n.pt`, but you can use other pretrained models or custom ones.

---

## 📬 License

GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007

---
