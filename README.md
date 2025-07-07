# Single View Metrology in the Wild Master's Thesis.
## 📚 Table of Contents

- [Overview](#-camera-tools)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
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
├── scripts/                     # Folder for auto-loaded modules
│   ├── extract_frames.py            # Extract frames from a video
│   ├── calibrate_camera.py          # Calibrate camera using chessboard frames
│   ├── estimate_height.py           # Estimate person height using YOLO + calibration
│   ├── cut_video.py                 # Cut video segments using ffmpeg
|   └── extract_realsense_frames.py  # Extract frames from a video from a .bag file
├── main.py                      # Entry point if you wish to unify all commands
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
For more information you can run the following code or read more at [scripts/README.md](scripts/README.md).

```bash
python main.py -h
```

## 🧠 Notes

* Make sure `ffmpeg` is installed and accessible in your system path. For linux: `sudo apt install ffmpeg`.
* The YOLO model defaults to `yolov8n.pt`, but you can use other pretrained models or custom ones.

---

## 📬 License

GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007

---
