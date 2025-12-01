# tracker_module.py

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from inference_engine import iou # 導入前一步驟的 IoU 函數

# 追蹤器的狀態結構
# (bbox_xyxy, conf, cls_id, track_id)

class ObjectTracker:
    """
    一個簡單的、基於 IoU 匹配的追蹤器。
    用於將連續幀的偵測結果連結起來，並分配唯一的追蹤 ID。
    """
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}

    def _iou_match(self, det_bbox: np.ndarray, tracked_bboxes: np.ndarray) -> np.ndarray:
        """計算單個偵測框與多個追蹤框之間的 IoU"""
        return iou(det_bbox, tracked_bboxes)

    def update(self, detections: List[Tuple[np.ndarray, float, int]]) -> List[Tuple[np.ndarray, float, int, int]]:
        """
        更新追蹤器狀態，並傳回帶有追蹤 ID 的結果。
        
        Args:
            detections: List of (xyxy, conf, cls_id)
            
        Returns:
            List of (xyxy, conf, cls_id, track_id)
        """
        if not detections:
            # 如果沒有新的偵測，更新所有現有軌跡的 age
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > self.max_age:
                    del self.tracks[track_id]
            return []

        det_bboxes = np.array([d[0] for d in detections])
        
        current_ids = list(self.tracks.keys())
        current_bboxes = np.array([self.tracks[tid]['bbox'] for tid in current_ids])
        
        matched_dets = [False] * len(detections)
        tracked_to_det_match: Dict[int, int] = {} # {track_id: det_index}

        # 1. 匹配：尋找最佳 IoU 匹配
        if current_bboxes.size > 0:
            for t_idx, track_id in enumerate(current_ids):
                # 與所有未匹配的偵測進行 IoU 比較
                ious = self._iou_match(current_bboxes[t_idx], det_bboxes)
                
                # 找到最佳匹配 (最大 IoU)
                best_match_idx = np.argmax(ious)
                max_iou = ious[best_match_idx]

                if max_iou >= self.iou_threshold and not matched_dets[best_match_idx]:
                    # 匹配成功
                    tracked_to_det_match[track_id] = best_match_idx
                    matched_dets[best_match_idx] = True
        
        # 2. 更新現有軌跡
        for track_id in list(self.tracks.keys()):
            if track_id in tracked_to_det_match:
                det_idx = tracked_to_det_match[track_id]
                xyxy, conf, cls_id = detections[det_idx]
                
                # 更新軌跡狀態
                self.tracks[track_id].update({
                    'bbox': xyxy,
                    'conf': conf,
                    'cls_id': cls_id,
                    'age': 0,
                    'hits': self.tracks[track_id]['hits'] + 1,
                })
            else:
                # 未匹配到，增加 age
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > self.max_age:
                    del self.tracks[track_id]
        
        # 3. 創建新軌跡 (未匹配的偵測)
        for d_idx, det in enumerate(detections):
            if not matched_dets[d_idx]:
                xyxy, conf, cls_id = det
                self.tracks[self.next_id] = {
                    'bbox': xyxy,
                    'conf': conf,
                    'cls_id': cls_id,
                    'age': 0,
                    'hits': 1,
                    'track_id': self.next_id,
                }
                self.next_id += 1
        
        # 4. 格式化輸出
        results: List[Tuple[np.ndarray, float, int, int]] = []
        for track_id, track in self.tracks.items():
            if track['age'] == 0: # 只輸出當前幀被更新的軌跡
                results.append((track['bbox'], track['conf'], track['cls_id'], track_id))

        return results

# end of tracker_module.py