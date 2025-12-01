# yolo_service.py

import os, time, argparse
from typing import Optional, Any, List, Tuple

# 匯入所有底層模組
from hardware_manager import create_ort_session, ensure_onnx_model
from inference_engine import run_inference, COCO80 
from tracker_module import ObjectTracker
from visualizer import Visualizer
from video_source import resolve_source, FrameGrabber

import cv2
import numpy as np

# --- 新增模型目錄常數 ---
MODEL_DIR = "models"
# ------------------------

class YOLOService:
    """
    將 YOLO 模型的推論、追蹤和視訊源管理封裝為一個可重用的服務。
    """
    def __init__(self, args: argparse.Namespace):
        """初始化服務：設定模型、硬體、追蹤器。"""
        self.args = args
        self.sess: Any = None
        self.in_name: str = ""
        self.out_name: str = ""
        self.img_size: int = 640
        self.tracker: Optional[ObjectTracker] = None
        self.visualizer: Optional[Visualizer] = None
        self.is_pose: bool = args.pose
        self.classes: Optional[List[int]] = self._parse_classes(args.classes)
        
        self._initialize_model()
        self._initialize_tracker()
        
        # 視覺化器只在需要繪圖時才初始化
        if not args.no_display:
            self.visualizer = Visualizer(names_path=args.names)
            
        print("[YOLOService] 服務初始化完成。")

    def _parse_classes(self, classes_str: Optional[str]) -> Optional[List[int]]:
        """解析類別過濾參數。"""
        if classes_str:
            try:
                return [int(c.strip()) for c in classes_str.split(',')]
            except ValueError:
                print("[Warning] 無效的 --classes 參數，將忽略類別過濾。")
        return None

    def _initialize_model(self):
        """
        載入 ONNX 模型並設定 ONNX Runtime 會話。
        模型將被解析並預設存放在 models/ 目錄下。
        """
        pt_model_name = self.args.pose_model if self.args.pose else self.args.detect_model
        
        # 1. 解析 PT 路徑：如果只給了檔名，則將其放在 models/ 目錄下
        if not os.path.dirname(pt_model_name):
            PT_PATH = os.path.join(MODEL_DIR, pt_model_name)
        else:
            PT_PATH = pt_model_name
            
        # 2. 解析 ONNX 路徑
        ONNX_PATH = os.path.splitext(PT_PATH)[0] + ".onnx"
        
        # 確保模型目標目錄存在 (例如 models/)
        model_target_dir = os.path.dirname(ONNX_PATH)
        if model_target_dir:
            os.makedirs(model_target_dir, exist_ok=True) 

        print(f"[Model Init] PT Path: {PT_PATH}, ONNX Path: {ONNX_PATH}")

        ensure_onnx_model(PT_PATH, ONNX_PATH, imgsz=self.img_size)
        self.sess, self.in_name, self.out_name, self.img_size = \
            create_ort_session(ONNX_PATH, self.args.force_cpu, self.args.provider)

    def _initialize_tracker(self):
        """初始化追蹤器。"""
        if self.args.tracker and not self.args.pose: 
            self.tracker = ObjectTracker(iou_threshold=0.3, max_age=5)
            print("[Tracker] 物件追蹤已啟用。")
        elif self.args.tracker and self.args.pose:
            print("[Warning] Pose 模式下暫不支持追蹤，已自動禁用 --tracker。")
            
    def process_frame(self, frame: np.ndarray) -> Any:
        """處理單幀，執行推論和追蹤。"""
        results, _, _, _ = run_inference(
            self.sess, self.in_name, self.out_name, frame, self.img_size, self.is_pose, 
            self.args.conf, self.args.iou, self.classes
        )
        
        if self.tracker is not None and not self.is_pose: 
            return self.tracker.update(results) 
        
        return results

    def run_stream(self):
        """
        運行視訊串流的主迴圈，同時處理可選的視覺輸出。
        """
        
        # === FIX: 確保 grabber 在作用域內 ===
        grabber = None 
        # ====================================

        # 1. 視訊源準備 (video_source)
        try:
            cap, source_fps = resolve_source(self.args.source, self.args.prefer_height)
        except Exception as e:
            print(f"\n[FATAL ERROR] 無法初始化視訊源: {e}")
            return

        # 2. 啟動 FrameGrabber
        grabber = FrameGrabber(cap, source_fps or 30.0, self.args.process_every)
        grabber.start()

        print("\n[Ready] 開始推論迴圈...")
        if not self.args.no_display:
            print(">>> 按下 [Q] 鍵 或 [ESC] 鍵 以關閉視窗並退出程式。")
            
        frame_id = 0
        start_time = time.perf_counter()
        last_dets: Any = [] 
        last_processed_frame_time = start_time

        while not grabber.stopped:
            frame = grabber.grab_frame()
            if frame is None:
                if time.perf_counter() - start_time > 10 and not grabber.is_alive():
                    print("[Error] 超時未收到幀，且抓取執行緒已停止。")
                    break
                time.sleep(0.001) 
                continue
            
            annotated = frame.copy()
            
            # 3. 推論和追蹤
            if frame_id % self.args.process_every == 0:
                last_processed_frame_time = time.perf_counter()
                
                # 執行處理
                last_dets = self.process_frame(frame)
                
                # 計算人數 (假設 COCO 0 = person)
                person_count = 0
                if not self.is_pose:
                    PERSON_CLASS_ID = 0
                    for det in last_dets:
                        cls_id = det[2] 
                        if cls_id == PERSON_CLASS_ID:
                            person_count += 1
                
            # 4. 輸出控制 (視覺輸出/靜默輸出)
            current_time = time.perf_counter()
            processing_time = current_time - last_processed_frame_time
            fps = 1.0 / processing_time if processing_time > 0 else 0
            
            count_message = f"People: {person_count}" if not self.is_pose else "Pose Mode"
            
            if self.args.no_display:
                print(f"Frame: {frame_id} | FPS: {fps:.2f} | Detections: {len(last_dets)} | {count_message}")
            else:
                # 繪製推論結果
                if self.visualizer:
                    if self.is_pose:
                        annotated = self.visualizer.draw_pose(annotated, last_dets, kpt_thres=self.args.kpt, show_bbox=True)
                    else:
                        annotated = self.visualizer.draw_detect(annotated, last_dets)
                        
                # 繪製統計
                cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(annotated, count_message, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
                cv2.imshow("YOLO Service Output", annotated)
            
            # 5. 退出檢查
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')): 
                grabber.stop()
                break
            
            frame_id += 1

        # 6. 清理
        # === FIX: 確保只有當 grabber 成功初始化時才調用 stop() ===
        if grabber: 
            grabber.stop()
        # ==========================================================
        cv2.destroyAllWindows()


def main_cli():
    """作為命令行工具使用時的主入口點"""
    parser = argparse.ArgumentParser(description="Universal YOLO ONNX Service CLI.")
    
    # 複製所有必要的參數定義
    parser.add_argument("--source", type=str, default="0", help="視訊源 (Webcam 索引, 檔案路徑, 或 URL)")
    parser.add_argument("--detect-model", type=str, default="yolo11s.pt", help=f"偵測模型檔名 (.pt 或 .onnx)，預設存放於 {MODEL_DIR}/ 目錄下。")
    parser.add_argument("--pose-model", type=str, default="yolo11s-pose.pt", help=f"姿態估計模型檔名 (.pt 或 .onnx)，預設存放於 {MODEL_DIR}/ 目錄下。")
    parser.add_argument("--pose", action="store_true", help="啟用姿態估計模式 (使用 --pose-model)")
    parser.add_argument("--provider", type=str, default="auto", choices=["auto", "dml", "hailo", "cpu"], help="ONNX Execution Provider (加速器)")
    parser.add_argument("--force-cpu", action="store_true", help="強制使用 CPU 推論")
    parser.add_argument("--conf", type=float, default=0.25, help="信心分數門檻")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU 門檻 (Detect 模式)")
    parser.add_argument("--kpt", type=float, default=0.5, help="關鍵點信心門檻 (Pose 模式)")
    parser.add_argument("--classes", type=str, default=None, help="過濾類別 (逗號分隔的索引)")
    parser.add_argument("--tracker", action="store_true", help="啟用物件追蹤")
    parser.add_argument("--no-display", action="store_true", help="禁用視覺輸出")
    parser.add_argument("--names", type=str, default=None, help="自定義類別名稱檔案路徑")
    parser.add_argument("--process-every", type=int, default=1, help="每 N 幀進行一次推論處理")
    parser.add_argument("--prefer-height", type=int, default=480, help="YouTube 串流解析度偏好")
    
    args = parser.parse_args()
    service = YOLOService(args)
    service.run_stream()

if __name__ == '__main__':
    main_cli()