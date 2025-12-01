# visualizer.py

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any

# 從 inference_engine 取得內建名稱
try:
    from inference_engine import COCO80
except ImportError:
    COCO80 = [f'class_{i}' for i in range(80)] # 避免循環依賴或錯誤

# 顏色生成器：為每個 ID/類別/追蹤分配一致的顏色
def make_colors(seed: int = 42, num_colors: int = 100) -> Dict[int, Tuple[int, int, int]]:
    """生成一組固定的 BGR 顏色，用於類別或追蹤 ID"""
    np.random.seed(seed)
    colors = {}
    for i in range(num_colors):
        color = np.random.randint(0, 256, size=3, dtype=np.uint8).tolist()
        colors[i] = tuple(color)
    return colors

# COCO 關鍵點連線 (僅用於 Pose 模式)
POSE_KPT_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [13, 11], [12, 11], [6, 12], [6, 11], [6, 7], [7, 8],
    [8, 9], [9, 10], [3, 2], [1, 2], [1, 0], [0, 4], [0, 5], [4, 6], [5, 7]
]
# 左右關鍵點的索引
LEFT_KPT_INDICES = [1, 3, 5, 7, 9, 11, 13, 15]
RIGHT_KPT_INDICES = [2, 4, 6, 8, 10, 12, 14, 16]

class Visualizer:
    def __init__(self, names_path: Optional[str] = None, tracker_max_id: int = 1000):
        self.names = self.load_names(names_path)
        self.cls_colors = make_colors(seed=2024, num_colors=len(self.names))
        # 追蹤 ID 顏色
        self.track_colors = make_colors(seed=1337, num_colors=tracker_max_id)
        
    def load_names(self, names_path: Optional[str]) -> List[str]:
        """
        載入類別名稱，若無則使用內建 COCO80。
        (Adapted from original script's load_names)
        """
        if names_path and os.path.exists(names_path):
            with open(names_path, 'r', encoding='utf-8') as f:
                return [s.strip() for s in f.readlines()]
        return COCO80

    def draw_detect(self, im: np.ndarray, results: List[Tuple[np.ndarray, float, int, Optional[int]]]) -> np.ndarray:
        """
        繪製偵測或追蹤結果 (BBox + Label)。
        
        Args:
            im: 原始圖像。
            results: List of (xyxy, conf, cls_id, [track_id])
        """
        lw = max(round(sum(im.shape) / 2 / 2000), 2) # 線寬
        tf = max(lw - 1, 1)                           # 文字厚度
        
        for det in results:
            xyxy, conf, cls_id = det[:3]
            track_id = det[3] if len(det) > 3 else None
            
            # 選擇顏色：追蹤優先，否則使用類別顏色
            if track_id is not None:
                color = self.track_colors.get(track_id % len(self.track_colors), (0, 255, 0))
            else:
                color = self.cls_colors.get(cls_id, (255, 255, 255))
            
            # 建立標籤
            c = self.names[cls_id] if cls_id < len(self.names) else str(cls_id)
            label = f'{track_id}: {c} {conf:.2f}' if track_id is not None else f'{c} {conf:.2f}'
            
            p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
            
            # 繪製邊框
            cv2.rectangle(im, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)

            # 繪製標籤
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]
            outside = p1[1] - h - 3 >= 0
            p2_txt = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            
            cv2.rectangle(im, p1, (p2_txt[0], p1[1] if outside else p2_txt[1]), color, -1, cv2.LINE_AA)
            cv2.putText(im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2 * tf + 2), 
                        0, lw / 3, (255, 255, 255), thickness=tf, lineType=cv2.LINE_AA)
        
        return im

    def draw_pose(self, im: np.ndarray, results: List[Tuple[np.ndarray, float, np.ndarray, Optional[int]]], kpt_thres: float = 0.5, show_bbox: bool = True) -> np.ndarray:
        """
        繪製姿態估計結果 (Keypoints + Skeleton)。
        
        Args:
            im: 原始圖像。
            results: List of (xyxy, conf, kpts, [track_id])
        """
        lw = max(round(sum(im.shape) / 2 / 2000), 2)
        
        for det in results:
            xyxy, conf, kpts = det[:3]
            track_id = det[3] if len(det) > 3 else None
            
            # 選擇顏色
            color = self.track_colors.get(track_id % len(self.track_colors), (255, 0, 0)) if track_id is not None else (255, 0, 0)
            
            # 1. 繪製邊框
            if show_bbox:
                p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                cv2.rectangle(im, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
                
                # 繪製 ID/Conf 標籤
                label = f'ID:{track_id} {conf:.2f}' if track_id is not None else f'{conf:.2f}'
                w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=lw)[0]
                cv2.rectangle(im, p1, (p1[0] + w, p1[1] - h - 3), color, -1, cv2.LINE_AA)
                cv2.putText(im, label, (p1[0], p1[1] - 2), 0, lw / 3, (255, 255, 255), thickness=lw, lineType=cv2.LINE_AA)


            # 2. 繪製骨架連線
            # kpts 結構為 (N, 3)，N 是關鍵點數量，3 是 (x, y, conf)
            for i, (p1_idx, p2_idx) in enumerate(POSE_KPT_SKELETON):
                # 關鍵點索引需要 +1 (從 0 開始的索引轉為 1-based)
                p1_kpt = kpts[p1_idx]
                p2_kpt = kpts[p2_idx]
                
                if p1_kpt[2] > kpt_thres and p2_kpt[2] > kpt_thres:
                    # 根據左右側給予不同顏色
                    line_color = (0, 255, 0) # 預設顏色
                    if p1_idx in LEFT_KPT_INDICES or p2_idx in LEFT_KPT_INDICES:
                        line_color = (255, 0, 0) # 左側 (藍色)
                    elif p1_idx in RIGHT_KPT_INDICES or p2_idx in RIGHT_KPT_INDICES:
                        line_color = (0, 0, 255) # 右側 (紅色)
                        
                    cv2.line(im, (int(p1_kpt[0]), int(p1_kpt[1])), (int(p2_kpt[0]), int(p2_kpt[1])), line_color, thickness=lw//2, lineType=cv2.LINE_AA)

            # 3. 繪製關鍵點
            for x, y, conf in kpts:
                if conf > kpt_thres:
                    cv2.circle(im, (int(x), int(y)), radius=lw, color=(0, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

        return im

# end of visualizer.py