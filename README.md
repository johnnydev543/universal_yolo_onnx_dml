# Universal YOLO ONNX Runtime (DirectML / Hailo / CPU)

A single Python script for real-time object detection **and pose estimation**  
that supports **any video source** and multiple backends:

- ✅ YouTube, RTSP, HTTP, local video, or webcam  
- ✅ Backends: **AMD iGPU (DirectML)**, **Hailo-8 NPU**, or **CPU fallback**  
- ✅ Models: **YOLO11n** (object detection) and **YOLO11s-pose** (pose / skeleton)
- ✅ Output: Realtime OpenCV window with labels, colored skeletons, and FPS-adaptive throttling  
- ✅ Written in pure Python — no GUI frameworks, just `onnxruntime`, `ultralytics`, and `opencv-python`

---

## ✨ Features

| Feature | Description |
|----------|--------------|
| 🎥 Source | YouTube link, RTSP stream, HTTP video, local file, or webcam index |
| ⚙️ Backend | ONNX Runtime: DirectML (AMD GPU), HailoExecutionProvider, or CPU |
| 🧍‍♂️ Pose mode | YOLO pose model with multi-color skeleton visualization |
| 🧠 Model export | Automatically converts `.pt` → `.onnx` if not found |
| 💬 Labels | Displays human-readable class names (COCO80 or custom) |
| 🧩 Configurable | Thresholds, NMS, processing interval, class filter, keypoint threshold |
| ⚡ Efficiency | Frame-grabber thread + frame skipping + adaptive FPS pacing |

---

## 🧰 Requirements

Python ≥ 3.10

```bash
pip install onnxruntime-directml ultralytics opencv-python yt-dlp
```

> 💡 For Hailo-8 users  
> Install Hailo SDK and ONNX Runtime Execution Provider for Hailo  
> (should make `'HailoExecutionProvider'` appear in `onnxruntime.get_available_providers()`).

---

## 🚀 Usage

### 1️⃣ Webcam
```bash
python universal_yolo_onnx_dml.py --source 0
```

### 2️⃣ Local video / RTSP / HTTP stream
```bash
python universal_yolo_onnx_dml.py --source ./video.mp4
python universal_yolo_onnx_dml.py --source rtsp://user:pass@ip:554/stream
```

### 3️⃣ YouTube
```bash
python universal_yolo_onnx_dml.py --source "https://youtu.be/abcd1234"
```

### 4️⃣ Pose Estimation (skeleton mode)
```bash
python universal_yolo_onnx_dml.py --source 0 --pose
```

### 5️⃣ Select backend
```bash
# AMD iGPU (DirectML)
python universal_yolo_onnx_dml.py --source 0 --provider dml

# CPU only
python universal_yolo_onnx_dml.py --source 0 --provider cpu

# Hailo-8 (requires SDK + EP)
python universal_yolo_onnx_dml.py --source 0 --provider hailo
```

### 6️⃣ Extra options
| Option | Description |
|--------|-------------|
| `--conf 0.25` | Confidence threshold |
| `--iou 0.45` | IoU threshold (detect) |
| `--classes 0,2` | Only track specific classes (e.g. person, car) |
| `--process-every 2` | Run inference every 2 frames (energy saving) |
| `--names my_labels.txt` | Custom class names file |
| `--prefer-height 480` | YouTube stream quality preference |
| `--force-cpu` | Force CPU inference even if GPU is present |

---

## 🧩 Example Outputs

| Mode | Description |
|------|--------------|
| **Detect** | YOLO11n bounding boxes + class names + confidence |
| **Pose** | YOLO11s-pose with colored skeleton and per-person confidence |

---

## 🧠 Architecture Overview

```
YouTube / RTSP / File / Webcam
   ↓
FrameGrabber Thread  ←─ throttled by source FPS
   ↓
preprocess(frame)
   ↓
ONNX Runtime Inference (DirectML / Hailo / CPU)
   ↓
postprocess_detect / postprocess_pose
   ↓
draw_detect / draw_pose
   ↓
OpenCV display
```

---

## 📦 Files

```
universal_yolo_onnx_dml.py     # main script
yolo11n.pt                     # detection model (optional, auto-downloaded by Ultralytics)
yolo11s-pose.pt                # pose model (optional)
```


## 🖼️ Preview (Pose Mode)

![pose_demo](docs/pose_demo.jpg)

---
