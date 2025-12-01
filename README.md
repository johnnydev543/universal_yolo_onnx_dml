# Universal YOLO ONNX Runtime (DirectML / Hailo / CPU)

一個模組化的 Python 專案，用於實時物件偵測、姿態估計和**物件計數**。
專案已重構為服務層和應用層，以提高重用性和可維護性。

* ✅ **模組化服務:** 核心推論邏輯封裝在 `yolo_service.py` 的 `YOLOService` 類別中。
* ✅ **專用應用:** 透過 `count_app.py` 專注於無視覺輸出的計數任務。
* ✅ **模型目錄:** 所有模型自動存儲和讀取自 `models/` 目錄。
* ✅ **多種後端:** 支援 **AMD iGPU (DirectML)**, **Hailo-8 NPU**, 或 **CPU fallback**。
* ✅ **豐富來源:** 支援 YouTube 連結, RTSP 串流, HTTP 視訊, 本地檔案或網路攝像頭。

---

## 🧰 Requirements

Python ≥ 3.10

```bash
pip install onnxruntime-directml ultralytics opencv-python yt-dlp
````

*注意：如果遇到 onnxruntime 相關的 `ImportError`，請參考 Troubleshooting 部分。*

-----

## 🚀 專案結構 (Files)

| 檔案 | 職責 |
|---|---|
| `yolo_service.py` | **核心服務**。封裝所有推論、追蹤、硬體和視訊I/O邏輯，並作為主要的視覺化 CLI 入口。 |
| `count_app.py` | **專用應用入口**。匯入 `yolo_service`，運行純計數邏輯，無視覺輸出。 |
| `hardware_manager.py` | 處理模型轉換 (`.pt` → `.onnx`) 和 ONNX Runtime Session 初始化。 |
| `inference_engine.py` | 處理幀的預處理、推論執行和後處理。 |
| `tracker_module.py` | 實現物件追蹤 (IoU Tracker)。 |
| `visualizer.py` | 處理所有繪圖、標籤和統計視覺化。 |
| `video_source.py` | 處理所有視訊源 I/O (Webcam, File, YouTube, FrameGrabber Thread)。 |
| `models/` | **所有模型** (`.pt` 和 `.onnx`) 的預設存放目錄。 |

-----

## ✨ 執行方式 (Usage)

### 1\. 視覺化和追蹤 (使用 `yolo_service.py`)

這是舊版 `main.py` 的功能，現在由服務模組接管。

```bash
# 運行 Webcam 0，啟用追蹤
python yolo_service.py --source 0 --tracker

# 運行 YouTube 連結，強制使用 CPU
python yolo_service.py --source "[https://www.youtube.com/watch?v=wm27ElpSxbM](https://www.youtube.com/watch?v=wm27ElpSxbM)" --provider cpu

# 運行並使用姿態估計模型
python yolo_service.py --source video.mp4 --pose
```

**提示:** 在 OpenCV 視窗中，按下 `Q` 或 `ESC` 鍵退出程式。

### 2\. 純物件計數 (使用 `count_app.py`)

此應用程式專門用於計數，預設禁用視覺化輸出 (`--no-display`)，並建議啟用追蹤 (`--tracker`) 以穩定計數結果。

```bash
# 運行視訊檔案，在終端機中輸出計數結果 (每 5 幀處理一次)
python count_app.py --source my_people.mp4 --process-every 5
```

**提示:** 在計數應用中，請使用 `Ctrl+C` 退出。

-----

## ⚙️ 參數選項 (Arguments)

| 參數 | 預設值 | 說明 |
|---|---|---|
| `--source` | `0` | 視訊源 (Webcam 索引、檔案路徑或 URL) |
| `--detect-model` | `yolo11s.pt` | 偵測模型檔名 (自動在 `models/` 中查找/下載) |
| `--pose-model` | `yolo11s-pose.pt` | 姿態模型檔名 (自動在 `models/` 中查找/下載) |
| `--pose` | `False` | 啟用姿態估計模式 |
| `--tracker` | `False` | 啟用物件追蹤 (僅限偵測模式) |
| `--provider` | `auto` | ONNX Execution Provider (`auto`, `dml`, `hailo`, `cpu`) |
| `--force-cpu` | `False` | 即使 GPU 存在，仍強制使用 CPU 推論 |
| `--conf` | `0.25` | 信心分數門檻 |
| `--iou` | `0.45` | NMS IoU 門檻 (Detect 模式) |
| `--kpt` | `0.5` | 關鍵點信心門檻 (Pose 模式) |
| `--classes` | `None` | 過濾類別 (逗號分隔的索引) |
| `--no-display` | `False` | 禁用視覺輸出 (僅在終端機中列印統計資訊) |
| `--names` | `None` | 自定義類別名稱檔案路徑 |
| `--process-every` | `1` | 每 N 幀進行一次推論處理 (節省性能) |
| `--prefer-height` | `480` | YouTube 串流解析度偏好 |

-----

## ❗ Troubleshooting

### 1\. onnxruntime 不兼容 (ImportError)

```
ImportError: cannot import name 'OrtDeviceMemoryType' from 'onnxruntime.capi._pybind_state'
```

這表示 `onnxruntime` 版本與你的 Python 環境不兼容。請在你的虛擬環境中執行以下命令強制更新：

```bash
pip uninstall -y onnxruntime onnxruntime-gpu onnxruntime-directml
pip install onnxruntime-directml
```
