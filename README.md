\# Universal YOLO ONNX Runtime (DirectML / Hailo / CPU)

一個模組化的 Python 專案，用於實時物件偵測、姿態估計和**\*\*物件計數\*\***。  
專案已重構為服務層和應用層，以提高重用性和可維護性。

\- ✅ **\*\*模組化服務:\*\*** 核心推論邏輯封裝在 \`yolo\_service.py\` 中。  
\- ✅ **\*\*專用應用:\*\*** 透過 \`count\_app.py\` 專注於無視覺輸出的計數任務。  
\- ✅ **\*\*模型目錄:\*\*** 所有模型自動存儲和讀取自 \`models/\` 目錄。  
\- ✅ **\*\*多種後端:\*\*** 支援 **\*\*AMD iGPU (DirectML)\*\***, **\*\*Hailo-8 NPU\*\***, 或 **\*\*CPU fallback\*\***。  
\- ✅ **\*\*豐富來源:\*\*** 支援 YouTube 連結, RTSP 串流, HTTP 視訊, 本地檔案或網路攝像頭。

\---

\#\# 🧰 Requirements

Python ≥ 3.10

\`\`\`bash  
pip install onnxruntime-directml ultralytics opencv-python yt-dlp

*注意：如果遇到 onnxruntime 相關的 ImportError，請確保您的 onnxruntime 版本與 Python 3.12 兼容。*

---

## **🚀 專案結構 (Files)**

專案已重構為以下模組：

| 檔案 | 職責 |
| :---- | :---- |
| yolo\_service.py | **核心服務**。封裝所有推論、追蹤、硬體和視訊I/O邏輯，並作為主要的視覺化 CLI 入口。 |
| count\_app.py | **專用應用入口**。匯入 yolo\_service，運行純計數邏輯，無視覺輸出。 |
| hardware\_manager.py | 處理模型轉換 (.pt → .onnx) 和 ONNX Runtime Session 初始化。 |
| inference\_engine.py | 處理幀的預處理、推論執行和後處理。 |
| tracker\_module.py | 實現物件追蹤 (IoU Tracker)。 |
| visualizer.py | 處理所有繪圖、標籤和統計視覺化。 |
| video\_source.py | 處理所有視訊源 I/O (Webcam, File, YouTube, FrameGrabber Thread)。 |
| models/ | **所有模型** (.pt 和 .onnx) 的存放目錄。 |

---

## **✨ 執行方式 (Usage)**

### **1\. 視覺化和追蹤 (使用 yolo\_service.py)**

這是舊版 main.py 的功能，現在由服務模組接管。

Bash

\# 運行 Webcam 0，啟用追蹤  
python yolo\_service.py \--source 0 \--tracker

\# 運行 YouTube 連結，強制使用 CPU  
python yolo\_service.py \--source "\[https://www.youtube.com/watch?v=\](https://www.youtube.com/watch?v=)..." \--provider cpu

\# 運行並使用姿態估計模型  
python yolo\_service.py \--source video.mp4 \--pose

### **2\. 純物件計數 (使用 count\_app.py)**

此應用程式專門用於計數，預設禁用視覺化輸出 (--no-display)，並建議啟用追蹤 (--tracker) 以穩定計數結果。

Bash

\# 運行視訊檔案，在終端機中輸出計數結果 (每 5 幀處理一次)  
python count\_app.py \--source my\_people.mp4 \--process-every 5

---

## **⚙️ 參數選項 (Arguments)**

| 參數 | 預設值 | 說明 |
| :---- | :---- | :---- |
| \--source | 0 | 視訊源 (Webcam 索引、檔案路徑或 URL) |
| \--detect-model | yolo11s.pt | 偵測模型檔名 (存放在 models/ 目錄下) |
| \--pose-model | yolo11s-pose.pt | 姿態模型檔名 (存放在 models/ 目錄下) |
| \--pose | False | 啟用姿態估計模式 |
| \--tracker | False | 啟用物件追蹤 (僅限偵測模式) |
| \--provider | auto | ONNX Execution Provider (auto, dml, hailo, cpu) |
| \--force-cpu | False | 即使 GPU 存在，仍強制使用 CPU 推論 |
| \--conf | 0.25 | 信心分數門檻 |
| \--no-display | False | 禁用視覺輸出 (僅在終端機中列印統計資訊) |
| \--process-every | 1 | 每 N 幀進行一次推論處理 (節省性能) |
| \--names | None | 自定義類別名稱檔案路徑 |
| \--prefer-height | 480 | YouTube 串流解析度偏好 |

---

## **🧠 架構概覽 (Architecture Overview)**

專案現在劃分為清晰的層次：

1. **I/O 層:** video\_source.py (FrameGrabber Thread) 負責穩定地讀取視訊幀。  
2. **服務層 (YOLOService):**  
   * hardware\_manager 準備模型。  
   * process\_frame 調用 inference\_engine 和 tracker\_module 處理單幀。  
3. **應用層:**  
   * yolo\_service.py 運行帶有 visualizer 的完整視覺化應用。  
   * count\_app.py 運行純數據處理和計數應用。

YouTube / RTSP / File / Webcam  
   ↓  
FrameGrabber Thread (video\_source.py)  
   ↓  
YOLOService.process\_frame(frame)  
   ↓  
(inference\_engine, tracker\_module) 處理數據  
   ↓  
count\_app.py (計數)  /  yolo\_service.py (視覺化輸出)

---

## **❗ Troubleshooting**

### **1\. onnxruntime 不兼容 (ImportError)**

ImportError: cannot import name 'OrtDeviceMemoryType' from 'onnxruntime.capi.\_pybind\_state'

這表示您的 onnxruntime 版本與您的 Python 3.12 環境不兼容。請執行以下命令強制更新：

Bash

pip uninstall \-y onnxruntime onnxruntime-gpu onnxruntime-directml  
pip install onnxruntime-directml

### **2\. 視訊源初始化失敗 (NameError: name 'grabber' is not defined)**

NameError: name 'grabber' is not defined

這通常發生在視訊源 (Webcam 或串流) 打開失敗時。yolo\_service.py 中的 grabber 變數在初始化失敗時未被賦值。請確保您的視訊源路徑正確，或嘗試使用 Webcam 索引 0。

### **3\. Ctrl+C 無法立即退出**

如果您的應用程式在按下 Ctrl+C 後卡住：  
請確認在 video\_source.py 中，FrameGrabber 執行緒已被設置為 self.daemon \= True。這將確保主程式退出時強制終止讀取執行緒。

\# video\_source.py  
class FrameGrabber(threading.Thread):  
    def \_\_init\_\_(self, ...):  
        super().\_\_init\_\_()  
        self.daemon \= True  \# 確保這行存在  
        \# ...  
