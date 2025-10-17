import os, sys, time, queue, threading, argparse
import cv2, numpy as np

# -------------------------
# 來源解析：webcam / 檔案 / URL / YouTube
# -------------------------
YTDL_AVAILABLE = False
try:
    import yt_dlp
    YTDL_AVAILABLE = True
except Exception:
    pass

def is_youtube(url: str) -> bool:
    if not isinstance(url, str): return False
    u = url.lower()
    return ("youtube.com" in u) or ("youtu.be" in u)

def get_youtube_stream(url: str, prefer_height=480):
    if not YTDL_AVAILABLE:
        raise RuntimeError("請先 pip install yt-dlp 以支援 YouTube 來源")
    fmt = f"best[ext=mp4][height<={prefer_height}]/best[height<={prefer_height}]/best"
    ydl_opts = {"quiet": True, "skip_download": True, "format": fmt}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        # 盡量拿到 fps；若無就回 None，由下游估算
        fps = info.get("fps")
        if fps is None and "requested_formats" in info and info["requested_formats"]:
            fps = info["requested_formats"][0].get("fps")
        return info["url"], (float(fps) if fps else None)

def resolve_source(src, prefer_height=480):
    """
    傳回 (opencv 可開啟的 source, fps_hint)
    - src 為 int/數字字串 → webcam
    - src 為 YouTube → 轉直連 URL
    - 其他字串（檔案/rtsp/http）→ 原樣
    """
    # webcam index
    if isinstance(src, int) or (isinstance(src, str) and src.isdigit()):
        return int(src), None

    # YouTube
    if isinstance(src, str) and is_youtube(src):
        url, fps = get_youtube_stream(src, prefer_height=prefer_height)
        return url, fps

    # file / rtsp / http(s)
    return src, None

# -------------------------
# ONNX Runtime（DirectML → CPU）
# -------------------------
def create_ort_session(onnx_path: str, force_cpu=False, provider: str = "auto"):
    import onnxruntime as ort
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
    print("[Providers available]", avail)
    print("[Providers using]", providers)

    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)

    in_name  = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    # 自動偵測模型輸入尺寸
    in_shape = sess.get_inputs()[0].shape  # 例如 [1,3,640,640] 或動態
    def _toi(x):
        try: return int(x)
        except: return None
    h = _toi(in_shape[2]) if len(in_shape) > 2 else None
    w = _toi(in_shape[3]) if len(in_shape) > 3 else None
    if h and w and h == w:
        img = h
        print(f"[Model Input] 固定 {img}x{img} (shape={in_shape})")
    else:
        img = 640
        print(f"[Model Input] 動態/非正方 {in_shape}，採用 {img}x{img}")

    return sess, in_name, out_name, img

def ensure_onnx_model(pt_path: str, onnx_path: str, imgsz=640):
    """
    若指定的 .onnx 檔不存在，從 .pt 模型匯出 ONNX。
    若存在，則直接使用。
    """
    import os
    if os.path.exists(onnx_path):
        print(f"[Model] 使用現有 {onnx_path}")
        return

    print(f"[Model] 找不到 {onnx_path}，從 {pt_path} 匯出 ONNX（imgsz={imgsz}）")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("請先安裝 ultralytics： pip install ultralytics")

    model = YOLO(pt_path)
    model.export(format="onnx", opset=20, dynamic=False, imgsz=imgsz, simplify=True)

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX 匯出失敗：未生成 {onnx_path}")
    else:
        print(f"[Model] 已匯出 {onnx_path}")


# -------------------------
# 前後處理 & NMS
# -------------------------
def letterbox(im, new_shape):
    h, w = im.shape[:2]
    if isinstance(new_shape, int): new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / h, new_shape[1] / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_shape[0] - nh) // 2
    left = (new_shape[1] - nw) // 2
    im_padded = cv2.copyMakeBorder(
        im_resized, top, new_shape[0]-nh-top, left, new_shape[1]-nw-left,
        cv2.BORDER_CONSTANT, value=(114,114,114)
    )
    return im_padded, r, left, top

def preprocess(frame, img_size):
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im, r, dw, dh = letterbox(im, new_shape=(img_size, img_size))
    im = (im.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    return im, r, dw, dh

def iou(box, boxes):
    inter_x1 = np.maximum(box[0], boxes[:,0]); inter_y1 = np.maximum(box[1], boxes[:,1])
    inter_x2 = np.minimum(box[2], boxes[:,2]); inter_y2 = np.minimum(box[3], boxes[:,3])
    inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    area1 = (box[2]-box[0]) * (box[3]-box[1])
    area2 = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    return inter / np.maximum(area1 + area2 - inter, 1e-6)

def nms(boxes, scores, iou_thres=0.45):
    idxs = scores.argsort()[::-1]; keep = []
    while idxs.size:
        i = idxs[0]; keep.append(i)
        if idxs.size == 1: break
        idxs = idxs[1:][iou(boxes[i], boxes[idxs[1:]]) < iou_thres]
    return keep

# -------------------------
# 後處理：Detect
# -------------------------
def postprocess_detect(pred, orig_shape, r, dw, dh, conf_thres=0.25, iou_thres=0.45, classes=None):
    if isinstance(pred, list): pred = pred[0]
    out = np.array(pred)
    if out.ndim == 3: out = out[0]       # (1,*,*) → (*,*)
    if out.shape[0] in (6,84,85) and out.shape[0] < out.shape[1]:
        out = out.T                      # 統一成 (N,C)
    if out.shape[1] < 6: return []

    H, W = orig_shape[:2]

    if out.shape[1] == 6:
        # 已含 NMS
        xyxy = out[:, :4].astype(np.float32).copy()
        conf = out[:, 4].astype(np.float32)
        cls_ids = out[:, 5].astype(int)
        m = conf >= conf_thres
        xyxy, conf, cls_ids = xyxy[m], conf[m], cls_ids[m]
        if classes is not None and xyxy.size:
            keep = np.isin(cls_ids, classes)
            xyxy, conf, cls_ids = xyxy[keep], conf[keep], cls_ids[keep]
        if xyxy.size == 0: return []
        xyxy[:, [0,2]] = np.clip(xyxy[:, [0,2]], 0, W-1)
        xyxy[:, [1,3]] = np.clip(xyxy[:, [1,3]], 0, H-1)
        keep_idx = nms(xyxy, conf, iou_thres=iou_thres)  # 可視需要關閉
        return [(xyxy[i], conf[i], int(cls_ids[i])) for i in keep_idx]

    # 不含 NMS
    box_xywh = out[:, :4].astype(np.float32)
    if out.shape[1] == 84:
        cls_scores = out[:, 4:].astype(np.float32)
        cls_ids = np.argmax(cls_scores, axis=1)
        conf = cls_scores[np.arange(len(cls_scores)), cls_ids]
    else:
        obj = out[:, 4].astype(np.float32)
        cls_scores = out[:, 5:].astype(np.float32)
        cls_ids = np.argmax(cls_scores, axis=1)
        cls_conf = cls_scores[np.arange(len(cls_scores)), cls_ids]
        conf = obj * cls_conf

    m = conf >= conf_thres
    box_xywh, conf, cls_ids = box_xywh[m], conf[m], cls_ids[m]
    if box_xywh.size == 0: return []
    if classes is not None:
        keep = np.isin(cls_ids, classes)
        box_xywh, conf, cls_ids = box_xywh[keep], conf[keep], cls_ids[keep]
        if box_xywh.size == 0: return []

    xyxy = np.empty_like(box_xywh)
    xyxy[:, 0] = (box_xywh[:, 0] - box_xywh[:, 2]/2 - dw) / r
    xyxy[:, 1] = (box_xywh[:, 1] - box_xywh[:, 3]/2 - dh) / r
    xyxy[:, 2] = (box_xywh[:, 0] + box_xywh[:, 2]/2 - dw) / r
    xyxy[:, 3] = (box_xywh[:, 1] + box_xywh[:, 3]/2 - dh) / r
    xyxy[:, [0,2]] = np.clip(xyxy[:, [0,2]], 0, W-1)
    xyxy[:, [1,3]] = np.clip(xyxy[:, [1,3]], 0, H-1)
    keep_idx = nms(xyxy, conf, iou_thres=iou_thres)
    return [(xyxy[i], conf[i], int(cls_ids[i])) for i in keep_idx]

# -------------------------
# 後處理：Pose
# -------------------------
def postprocess_pose(pred, orig_shape, r, dw, dh, conf_thres=0.25, kpt_thres=0.20):
    if isinstance(pred, list): pred = pred[0]
    out = np.array(pred)
    if out.ndim == 3: out = out[0]
    if out.shape[0] == 56 and out.shape[0] < out.shape[1]:
        out = out.T
    if out.shape[1] < 55: 
        return []

    box_xywh = out[:, :4].astype(np.float32)
    det_conf = out[:, 4].astype(np.float32)
    kflat    = out[:, 5:].astype(np.float32)
    if kflat.shape[1] % 3 != 0:
        return []

    num_kpts = kflat.shape[1] // 3
    kpts = kflat.reshape(-1, num_kpts, 3)

    # 先用置信度過濾
    m = det_conf >= conf_thres
    box_xywh, det_conf, kpts = box_xywh[m], det_conf[m], kpts[m]
    if box_xywh.size == 0:
        return []

    H, W = orig_shape[:2]

    # 轉回原圖座標（bbox）
    xyxy = np.empty_like(box_xywh)
    xyxy[:, 0] = (box_xywh[:, 0] - box_xywh[:, 2] / 2 - dw) / r
    xyxy[:, 1] = (box_xywh[:, 1] - box_xywh[:, 3] / 2 - dh) / r
    xyxy[:, 2] = (box_xywh[:, 0] + box_xywh[:, 2] / 2 - dw) / r
    xyxy[:, 3] = (box_xywh[:, 1] + box_xywh[:, 3] / 2 - dh) / r
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, W - 1)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, H - 1)

    # 還原 keypoints 到原圖
    kpts_out = kpts.copy()
    kpts_out[..., 0] = (kpts[..., 0] - dw) / r
    kpts_out[..., 1] = (kpts[..., 1] - dh) / r
    kpts_out[..., 0] = np.clip(kpts_out[..., 0], 0, W - 1)
    kpts_out[..., 1] = np.clip(kpts_out[..., 1], 0, H - 1)

    # === ① IoU NMS：先用人框去掉重疊 ===
    def _iou(a, b):
        xx1 = max(a[0], b[0]); yy1 = max(a[1], b[1])
        xx2 = min(a[2], b[2]); yy2 = min(a[3], b[3])
        iw = max(0.0, xx2 - xx1); ih = max(0.0, yy2 - yy1)
        inter = iw * ih
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        return inter / max(ua, 1e-6)

    order = np.argsort(-det_conf)
    keep  = []
    suppressed = np.zeros(len(order), dtype=bool)
    iou_thres = 0.5  # 可調

    for i, idx in enumerate(order):
        if suppressed[i]: 
            continue
        keep.append(idx)
        for j in range(i+1, len(order)):
            if suppressed[j]: 
                continue
            if _iou(xyxy[idx], xyxy[order[j]]) >= iou_thres:
                suppressed[j] = True

    xyxy = xyxy[keep]
    det_conf = det_conf[keep]
    kpts_out = kpts_out[keep]

    # === ② 近鄰去重：框中心太近也視為同一人（解決殘餘疊框）===
    if len(xyxy) > 1:
        centers = np.column_stack(((xyxy[:,0]+xyxy[:,2])/2, (xyxy[:,1]+xyxy[:,3])/2))
        # 距離閾值：對角線的 5%
        diag = np.sqrt((W**2 + H**2))
        thr  = 0.05 * diag
        keep2 = []
        used = np.zeros(len(xyxy), dtype=bool)
        order2 = np.argsort(-det_conf)
        for i in order2:
            if used[i]: 
                continue
            keep2.append(i)
            used[i] = True
            di = np.linalg.norm(centers - centers[i], axis=1)
            dup = np.where((di < thr) & (~used))[0]
            used[dup] = True
        xyxy = xyxy[keep2]
        det_conf = det_conf[keep2]
        kpts_out = kpts_out[keep2]

    # 回傳
    return [(xyxy[i], det_conf[i], kpts_out[i]) for i in range(len(xyxy))]


# 內建 COCO80 名稱（若沒給 --names 就用這個）
COCO80 = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light',
    'fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow',
    'elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase',
    'frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard',
    'surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl',
    'banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake',
    'chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote',
    'keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock',
    'vase','scissors','teddy bear','hair drier','toothbrush'
]

def load_names(path: str | None):
    if not path:
        return COCO80
    try:
        with open(path, "r", encoding="utf-8") as f:
            names = [ln.strip() for ln in f if ln.strip()]
        return names
    except Exception as e:
        print(f"[Names] 無法讀取 {path}：{e}，改用 COCO80")
        return COCO80


# -------------------------
# 繪製：Detect
# -------------------------
def draw_detect(frame, dets, names=None):
    names = names or COCO80
    n = len(names)
    for (x1,y1,x2,y2), c, cls in dets:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, (0,255,0), 2)
        label = f"{names[cls] if 0 <= cls < n else cls} {c:.2f}"
        cv2.putText(frame, label, (p1[0], p1[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return frame


# -------------------------
# 繪製：Pose（彩色骨架、細字體）
# -------------------------
COCO_PAIRS_COLORED = [
    ((5, 7), (0,128,255)), ((7, 9), (0,128,255)),    # 左臂
    ((6, 8), (255,128,0)), ((8,10), (255,128,0)),    # 右臂
    ((5, 6), (0,255,255)),                           # 肩-肩
    ((5,11), (0,255,255)), ((6,12), (0,255,255)),    # 肩-髖
    ((11,12),(0,255,255)),                           # 髖-髖
    ((11,13),(0,255,0)),  ((13,15),(0,255,0)),       # 左腿
    ((12,14),(0,128,0)),  ((14,16),(0,128,0)),       # 右腿
    ((0,5), (255,0,255)), ((0,6), (255,0,255)),      # 鼻-肩
    ((0,1), (255,0,255)), ((1,3), (255,0,255)),      # 鼻-左眼-左耳
    ((0,2), (255,0,255)), ((2,4), (255,0,255)),      # 鼻-右眼-右耳
]

def draw_pose(frame, dets, kpt_thres=0.20):
    for (x1, y1, x2, y2), c, kpts in dets:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, (200,200,200), 1)
        cv2.putText(frame, f"{c:.2f}", (p1[0], p1[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

        pts = []
        for i in range(kpts.shape[0]):
            x, y, s = kpts[i]
            if s >= kpt_thres:
                cv2.circle(frame, (int(x), int(y)), 2, (255,255,255), -1, cv2.LINE_AA)
                pts.append((i, int(x), int(y), True))
            else:
                pts.append((i, int(x), int(y), False))

        for (a, b), color in COCO_PAIRS_COLORED:
            ia = pts[a]; ib = pts[b]
            if ia[3] and ib[3]:
                cv2.line(frame, (ia[1], ia[2]), (ib[1], ib[2]), color, 2, cv2.LINE_AA)
    return frame

# -------------------------
# 讀取執行緒（任何來源），節流＋小緩衝
# -------------------------
class FrameGrabber(threading.Thread):
    def __init__(self, source, fps_hint=None):
        super().__init__(daemon=True)
        self.source = source
        self.fps = max(5.0, min(60.0, float(fps_hint) if fps_hint else 30.0))
        self.period = 1.0 / self.fps
        self.q = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.cap = None
        self.fail = 0

    def open(self):
        if self.cap: self.cap.release()
        # 嘗試用 FFMPEG 後端；若不支援會退回預設
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            # 再試一次不用 FFMPEG
            self.cap = cv2.VideoCapture(self.source)
        # 嘗試從來源讀 FPS
        src_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if src_fps and src_fps > 1 and not np.isnan(src_fps):
            self.fps = max(5.0, min(60.0, float(src_fps)))
            self.period = 1.0 / self.fps
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    def run(self):
        self.open()
        next_t = time.perf_counter()
        while not self.stop_event.is_set():
            if not self.cap.isOpened():
                self.fail += 1
                time.sleep(0.5)
                self.open(); continue
            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.fail += 1
                if self.fail > 50:
                    self.open(); self.fail = 0
                time.sleep(0.01); continue
            self.fail = 0

            if self.q.full():
                try: _ = self.q.get_nowait()
                except queue.Empty: pass
            try: self.q.put_nowait(frame)
            except queue.Full: pass

            # 節流：依來源 FPS
            next_t += self.period
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()

    def read(self, timeout=1.0):
        try:
            return True, self.q.get(timeout=timeout)
        except queue.Empty:
            return False, None

    def release(self):
        self.stop_event.set()
        try: self.join(timeout=1.0)
        except RuntimeError: pass
        if self.cap: self.cap.release()

# -------------------------
# 主流程
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", "-s", default="0",
                    help="來源：webcam index（如 0）、檔案路徑、rtsp/http url、YouTube 連結")
    ap.add_argument("--pose", action="store_true", help="使用 yolo11s-pose（骨架）")
    ap.add_argument("--pt", default=None, help="自訂 .pt 路徑（預設 detect: yolo11n.pt / pose: yolo11s-pose.pt）")
    ap.add_argument("--onnx", default=None, help="自訂 .onnx 路徑（預設 detect: yolo11n.onnx / pose: yolo11s-pose.onnx）")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--kpt", type=float, default=0.20, help="pose keypoint 顯示門檻")
    ap.add_argument("--classes", type=str, default=None, help="只保留指定類別（逗號分隔，如 0,2），僅 detect 模式適用")
    ap.add_argument("--process-every", type=int, default=1, help="每 N 幀推一次（節能）")
    ap.add_argument("--force-cpu", action="store_true", help="強制用 CPU，不用 DirectML")
    ap.add_argument("--prefer-height", type=int, default=480, help="YouTube 解析度偏好")
    ap.add_argument("--backend", default="ort", choices=["ort"],  # 目前走 ONNX Runtime 路線
                    help="推論後端。先用 ORT；Hailo 用 ORT 的 HailoExecutionProvider")
    ap.add_argument("--provider", default="auto", choices=["auto", "dml", "cpu", "hailo"],
                    help="ONNX Runtime Provider：auto/dml/cpu/hailo")
    ap.add_argument("--names", default=None,
                    help="類別名稱檔 (.txt，每行一個類別)。不指定則用 COCO80 內建")
    args = ap.parse_args()

    NAMES = load_names(args.names)

    USE_POSE = args.pose
    PT_MODEL = args.pt if args.pt else ("yolo11s-pose.pt" if USE_POSE else "yolo11n.pt")
    ONNX_MODEL = args.onnx if args.onnx else ("yolo11s-pose.onnx" if USE_POSE else "yolo11n.onnx")
    CLASSES = None
    if (not USE_POSE) and args.classes:
        CLASSES = [int(x) for x in args.classes.split(",") if x.strip().isdigit()]

    # 來源解析
    source, fps_hint = resolve_source(args.source, prefer_height=args.prefer_height)
    print(f"[Source] {source}  (fps_hint={fps_hint})")

    # 準備 ONNX
    ensure_onnx_model(PT_MODEL, ONNX_MODEL, imgsz=640)

    sess, in_name, out_name, IMG = create_ort_session(
        ONNX_MODEL,
        force_cpu=args.force_cpu,
        provider=args.provider
    )

    # 啟動讀取執行緒
    grabber = FrameGrabber(source, fps_hint=fps_hint)
    grabber.start()

    # 顯示節流（再一次以來源 FPS pace）
    shown = 0
    dt = 1.0 / max(5.0, min(60.0, (fps_hint if fps_hint else grabber.fps)))
    last_show = time.perf_counter()
    frame_id = 0
    last_dets = []
    MAX_CONSEC_FAIL = 200
    consec_fail = 0

    try:
        while True:
            ok, frame = grabber.read(timeout=2.0)
            if not ok or frame is None:
                consec_fail += 1
                if consec_fail > MAX_CONSEC_FAIL:
                    print("⚠️ 來源中斷過久，結束播放")
                    break
                continue
            consec_fail = 0
            frame_id += 1

            if frame_id % args.process_every == 0:
                inp, r, dw, dh = preprocess(frame, img_size=IMG)
                pred = sess.run([out_name], {in_name: inp})[0]

                if USE_POSE:
                    last_dets = postprocess_pose(
                        pred, frame.shape, r, dw, dh,
                        conf_thres=args.conf, kpt_thres=args.kpt
                    )
                    annotated = draw_pose(frame, last_dets, kpt_thres=args.kpt)

                else:
                    last_dets = postprocess_detect(
                        pred, frame.shape, r, dw, dh,
                        conf_thres=args.conf, iou_thres=args.iou, classes=CLASSES
                    )
                    annotated = draw_detect(frame, last_dets, names=NAMES)


            annotated = draw_pose(frame, last_dets, kpt_thres=args.kpt) if USE_POSE else draw_detect(frame, last_dets)

            # 顯示節流
            now = time.perf_counter()
            sleep = (last_show + dt) - now
            if sleep > 0:
                time.sleep(sleep)
            last_show = time.perf_counter()

            cv2.imshow("YOLO (Universal, ONNX + DirectML)", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
    finally:
        grabber.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
