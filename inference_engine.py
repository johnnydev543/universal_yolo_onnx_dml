# inference_engine.py

import cv2
import numpy as np
from typing import List, Tuple, Any, Optional

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

# --- 前後處理 & NMS ---

def letterbox(im: np.ndarray, new_shape: Tuple[int, int]) -> Tuple[np.ndarray, float, int, int]:
    """
    Resize image to a fixed shape with padding (letterbox).
    Returns padded image, scale ratio (r), and padding offsets (dw, dh).
    (Adapted from original script's letterbox)
    """
    h, w = im.shape[:2]
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

def preprocess(frame: np.ndarray, img_size: int) -> Tuple[np.ndarray, float, int, int]:
    """
    Preprocess the frame for YOLO ONNX inference.
    Returns: (input_tensor, ratio, pad_width, pad_height)
    (Adapted from original script's preprocess)
    """
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    im, r, dw, dh = letterbox(im, new_shape=(img_size, img_size))
    # HWC to NCHW, normalize
    im = (im.astype(np.float32) / 255.0).transpose(2, 0, 1)[None, ...]
    return im, r, dw, dh

def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Compute IoU of a single box with multiple boxes (vectorized).
    (Adapted from original script's iou)
    """
    inter_x1 = np.maximum(box[0], boxes[:,0]); inter_y1 = np.maximum(box[1], boxes[:,1])
    inter_x2 = np.minimum(box[2], boxes[:,2]); inter_y2 = np.minimum(box[3], boxes[:,3])
    inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    area1 = (box[2]-box[0]) * (box[3]-box[1])
    area2 = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    return inter / np.maximum(area1 + area2 - inter, 1e-6)

def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float = 0.45) -> List[int]:
    """
    Non-Maximum Suppression (NMS).
    (Adapted from original script's nms)
    """
    idxs = scores.argsort()[::-1]; keep = []
    while idxs.size:
        i = idxs[0]; keep.append(i)
        if idxs.size == 1: break
        idxs = idxs[1:][iou(boxes[i], boxes[idxs[1:]]) < iou_thres]
    return keep

# --- 後處理：Detect ---

def postprocess_detect(pred: Any, orig_shape: Tuple[int, int, int], r: float, dw: int, dh: int, conf_thres: float = 0.25, iou_thres: float = 0.45, classes: Optional[List[int]] = None) -> List[Tuple[np.ndarray, float, int]]:
    """
    Post-process raw YOLO detection output to bounding boxes in original image coordinates.
    Returns: List[(xyxy, conf, cls_id)]
    (Adapted from original script's postprocess_detect)
    """
    if isinstance(pred, list): pred = pred[0]
    out = np.array(pred)
    if out.ndim == 3: out = out[0]       # (1,*,*) → (*,*)
    if out.shape[0] in (6, 84, 85) and out.shape[0] < out.shape[1]:
        out = out.T                      # 統一成 (N,C)
    if out.shape[1] < 6: return []

    H, W = orig_shape[:2]

    # --- 輸出已含 NMS (e.g., ONNX exports with Postprocessing operator) ---
    if out.shape[1] == 6:
        xyxy = out[:, :4].astype(np.float32).copy()
        conf = out[:, 4].astype(np.float32)
        cls_ids = out[:, 5].astype(int)
        
        m = conf >= conf_thres
        xyxy, conf, cls_ids = xyxy[m], conf[m], cls_ids[m]
        
        if classes is not None and xyxy.size:
            keep = np.isin(cls_ids, classes)
            xyxy, conf, cls_ids = xyxy[keep], conf[keep], cls_ids[keep]
            
        if xyxy.size == 0: return []
        
        # 座標已經是原圖比例，只需裁剪
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, W-1)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, H-1)
        
        # 即使模型已內建 NMS，腳本仍可選擇再跑一次 (可視需要關閉)
        keep_idx = nms(xyxy, conf, iou_thres=iou_thres)
        return [(xyxy[i], conf[i], int(cls_ids[i])) for i in keep_idx]

    # --- 輸出未含 NMS (e.g., 84/85 寬度的原始輸出) ---
    box_xywh = out[:, :4].astype(np.float32)
    if out.shape[1] == 84:
        # YOLOv8 Seg/Det-without-NMS 輸出
        cls_scores = out[:, 4:].astype(np.float32)
        cls_ids = np.argmax(cls_scores, axis=1)
        conf = cls_scores[np.arange(len(cls_scores)), cls_ids]
    else: # 85 寬度 (YOLOv5/v7/v8 Det)
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

    # 轉換 box_xywh (模型空間) → xyxy (原圖空間)
    xyxy = np.empty_like(box_xywh)
    # 逆向 letterbox 轉換
    xyxy[:, 0] = (box_xywh[:, 0] - box_xywh[:, 2]/2 - dw) / r
    xyxy[:, 1] = (box_xywh[:, 1] - box_xywh[:, 3]/2 - dh) / r
    xyxy[:, 2] = (box_xywh[:, 0] + box_xywh[:, 2]/2 - dw) / r
    xyxy[:, 3] = (box_xywh[:, 1] + box_xywh[:, 3]/2 - dh) / r
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, W-1)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, H-1)
    
    keep_idx = nms(xyxy, conf, iou_thres=iou_thres)
    return [(xyxy[i], conf[i], int(cls_ids[i])) for i in keep_idx]

# --- 後處理：Pose ---

def postprocess_pose(pred: Any, orig_shape: Tuple[int, int, int], r: float, dw: int, dh: int, conf_thres: float = 0.25) -> List[Tuple[np.ndarray, float, np.ndarray]]:
    """
    Post-process raw YOLO pose output to bounding boxes and keypoints in original image coordinates.
    Returns: List[(xyxy, conf, kpts)]
    (Adapted from original script's postprocess_pose, simplified NMS/dedup to standard NMS for modularity)
    """
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

    # 篩選
    m = det_conf >= conf_thres
    box_xywh, det_conf, kflat = box_xywh[m], det_conf[m], kflat[m]
    if box_xywh.size == 0:
        return []

    H, W = orig_shape[:2]
    num_kpts = kflat.shape[1] // 3
    kpts = kflat.reshape(-1, num_kpts, 3)

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

    # NMS
    keep_idx = nms(xyxy, det_conf, iou_thres=0.5) # Pose NMS threshold typically around 0.5
    
    return [(xyxy[i], det_conf[i], kpts_out[i]) for i in keep_idx]

# --- 統一介面 (可選) ---

def run_inference(
    sess: Any, 
    in_name: str, 
    out_name: str, 
    frame: np.ndarray, 
    img_size: int, 
    is_pose: bool, 
    conf_thres: float, 
    iou_thres: float, 
    classes: Optional[List[int]] = None
) -> Tuple[List, float, int, int]:
    """
    執行單次推論並後處理，傳回結果和 letterbox 參數。
    """
    inp, r, dw, dh = preprocess(frame, img_size=img_size)
    pred = sess.run([out_name], {in_name: inp})[0]
    
    if is_pose:
        results = postprocess_pose(
            pred, frame.shape, r, dw, dh, conf_thres=conf_thres
        )
    else:
        results = postprocess_detect(
            pred, frame.shape, r, dw, dh, conf_thres=conf_thres, iou_thres=iou_thres, classes=classes
        )
        
    return results, r, dw, dh

# end of inference_engine.py