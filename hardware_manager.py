# hardware_manager.py

import os, shutil
import onnxruntime as ort
from typing import Dict, Any, Tuple, Optional

# --- ONNX Runtime (DirectML / Hailo / CPU) ---

def create_ort_session(onnx_path: str, force_cpu: bool = False, provider: str = "auto") -> Tuple[ort.InferenceSession, str, str, int]:
    """
    建立 ONNX Runtime 推論會話，並根據 provider 參數選擇 Execution Provider。
    
    傳回：(session, input_name, output_name, input_size)
    """
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    avail = ort.get_available_providers()
    
    def pick_providers():
        if provider == "hailo":
            if "HailoExecutionProvider" not in avail:
                raise RuntimeError("指定 --provider hailo，但未找到 HailoExecutionProvider。請安裝 Hailo 的 ORT EP。")
            return ["HailoExecutionProvider"]
        if provider == "dml":
            if "DmlExecutionProvider" not in avail:
                # 原始腳本中的錯誤處理
                raise RuntimeError("指定 --provider dml，但此環境沒有 DmlExecutionProvider。請安裝 onnxruntime-directml。")
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        if provider == "cpu" or force_cpu:
            return ["CPUExecutionProvider"]
        
        # auto：優先 DML、再 Hailo、最後 CPU
        if "DmlExecutionProvider" in avail:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        if "HailoExecutionProvider" in avail:
            return ["HailoExecutionProvider"]
        return ["CPUExecutionProvider"]

    providers = pick_providers()
    print(f"[Providers available] {avail}")
    print(f"[Providers using] {providers}")

    # 建立會話
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)

    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    # 自動偵測模型輸入尺寸
    in_shape = sess.get_inputs()[0].shape
    def _toi(x):
        try: return int(x)
        except: return None
    
    h = _toi(in_shape[2]) if len(in_shape) > 2 else None
    w = _toi(in_shape[3]) if len(in_shape) > 3 else None
    
    img = 640
    if h and w and h == w:
        img = h
        print(f"[Model Input] 固定 {img}x{img} (shape={in_shape})")
    else:
        print(f"[Model Input] 動態/非正方 {in_shape}，採用 {img}x{img}")

    return sess, in_name, out_name, img


def ensure_onnx_model(pt_path: str, onnx_path: str, imgsz: int = 640):
    """
    確保 ONNX 檔案存在。若缺失，則從 YOLO .pt 檔案匯出。
    這個函數與原始腳本中的邏輯相同，但獨立出來。
    """
    if os.path.exists(onnx_path):
        print(f"[Model] 使用現有 {onnx_path}")
        return

    print(f"[Model] 找不到 {onnx_path}，準備從 {pt_path} 匯出 ONNX (imgsz={imgsz})")

    try:
        from importlib import import_module
        # 動態匯入 ultralytics
        ultralytics_mod = import_module('ultralytics')
        YOLO = getattr(ultralytics_mod, 'YOLO', None)
        if YOLO is None:
            sub = import_module('ultralytics.yolo')
            YOLO = getattr(sub, 'YOLO')
    except Exception:
        raise ImportError("請先安裝 ultralytics： pip install ultralytics")

    load_arg = pt_path
    if not os.path.exists(pt_path):
        # 讓 Ultralytics 嘗試從網路下載官方權重
        load_arg = os.path.basename(pt_path)
        print(f"[Model] {pt_path} 不存在，改用 {load_arg} 嘗試由 Ultralytics 下載")

    model = YOLO(load_arg)
    
    # 確保目標目錄存在
    odir = os.path.dirname(os.path.abspath(onnx_path)) or "."
    os.makedirs(odir, exist_ok=True)

    # 執行匯出
    # 注意：原始腳本的匯出邏輯有點複雜，這裡簡化並依賴 model.export 的返回值
    try:
        res = model.export(format="onnx", opset=20, dynamic=False, imgsz=imgsz, simplify=True)
    except Exception as e:
        raise RuntimeError(f"Ultralytics ONNX 匯出失敗：{e}")

    # 嘗試解析導出的路徑並移動/複製到 onnx_path
    candidate_paths = []
    if isinstance(res, (str, os.PathLike)):
        candidate_paths.append(str(res))
    
    # 檢查預期的檔案名稱 (通常是 PT 檔名改為 ONNX 擴展名)
    base_onnx = os.path.splitext(os.path.basename(pt_path))[0] + ".onnx"
    candidate_paths += [base_onnx, "model.onnx", "weights.onnx"]
    
    copied = False
    for cand in set(candidate_paths): # 使用 set 去重
        if cand and os.path.exists(cand):
            try:
                # 複製到我們預定的路徑
                shutil.copyfile(cand, onnx_path)
                print(f"[Model] 已匯出並複製到 {onnx_path}")
                copied = True
                break
            except Exception as e:
                print(f"[Model] 複製 {cand} → {onnx_path} 失敗：{e}")

    if not os.path.exists(onnx_path):
        if copied:
            return
        raise FileNotFoundError(f"ONNX 匯出完成但未在預期位置找到：{onnx_path}。請檢查 Ultralytics 的輸出目錄並手動移動。")
        
    print(f"[Model] 已匯出 {onnx_path}")

# end of hardware_manager.py