# Scripts
![High Overview of the scripts architecture](../imgs/architecture_diagram.mx.drawio.png?raw=true)
![Command Design Pattern Diagram](../imgs/design_pattern_diagram.mx.drawio.png?raw=true)
## Table of Contents
  - [Extract Frames](#-1-extract-frames)
  - [Calibrate Camera](#-2-calibrate-camera)
  - [Estimate Human Height](#-3-estimate-human-height)
  - [Cut Video](#-4-cut-video)
  - [Extract Realsense Frames](#-5-extracting-frames-from-realsense-bag-files)

### ✅ 1. Extract Frames

Extract frames from a video at a given frame rate.

```bash
python main.py extract-frames \
  --input path/to/video.mp4 \
  --rate 1
```

This will save one frame per second in a new folder next to the video.

---

### ✅ 2. Calibrate Camera

Calibrate a camera using extracted chessboard images.

```bash
python main.py calibrate-camera \
  --input path/to/frames \
  --board-height 6 --board-width 6 \
  --output calibration/intrinsics.yaml
```

You’ll get an intrinsics YAML file with camera matrix and distortion coefficients.

---

### ✅ 3. Estimate Human Height

Estimate the height of a visible person in an image using YOLO and calibration data.

```bash
python main.py estimate-height \
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
python main.py cut-video \
  --input video.mp4 \
  --start 00:01:30 \
  --end 00:03:00
```

Outputs a new file named `video-cut.mp4`.

---

### ✅ 5. Extracting Frames from RealSense `.bag` Files

This script allows you to extract color frames from a **RealSense `.bag` file** recorded with Intel RealSense cameras. You can control:

```bash
python main.py extract-realsense-frames \
  --bag path/to/your_file.bag \
  --output path/to/output_dir \
  --rate 1.0 \
  --start 00:00:10 \
  --end 00:00:30
```
This command extracts one frame every 0.5 seconds between 5 and 20 seconds of the video.
