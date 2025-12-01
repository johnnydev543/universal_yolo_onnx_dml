# video_source.py

import os, sys, time, queue, threading, argparse
from typing import Optional, Any, Dict, cast, Tuple
import cv2

YTDL_AVAILABLE = False
try:
    # 這是原始腳本中用於處理 YouTube 連結的依賴
    import yt_dlp
    YTDL_AVAILABLE = True
except Exception:
    pass

# -------------------------
# 來源解析工具函數
# -------------------------

def is_youtube(url: str) -> bool:
    """檢查給定的 URL 是否為 YouTube 連結。"""
    if not isinstance(url, str): return False
    u = url.lower()
    return ("youtube.com" in u) or ("youtu.be" in u)

def get_youtube_stream(url: str, prefer_height: int = 480) -> Tuple[str, Optional[float]]:
    """
    使用 yt-dlp 取得 YouTube 串流的 URL。
    
    傳回：(stream_url, fps)
    """
    if not YTDL_AVAILABLE:
        raise RuntimeError("請先 pip install yt-dlp 以支援 YouTube 來源")
        
    # 選擇最佳的 mp4 串流，高度不超過 prefer_height
    fmt = f"best[ext=mp4][height<={prefer_height}]/best[height<={prefer_height}]/best"
    
    # yt-dlp 選項：靜默模式, 不下載, 格式過濾
    ydl_opts: Dict[str, Any] = {"quiet": True, "skip_download": True, "format": fmt}
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            info = cast(Dict[str, Any], info)
            
            # 取得最佳格式的 URL
            stream_url = info.get('url')
            if not stream_url and info.get('formats'):
                # 如果頂層沒有 'url'，從格式列表中找最好的
                best_format = info['formats'][0] 
                stream_url = best_format.get('url')

            if not stream_url:
                raise RuntimeError(f"yt-dlp 無法為 {url} 找到可用的串流 URL。")

            # 取得 FPS (如果可用)
            fps = info.get('fps')
            if fps is None and info.get('formats'):
                 for fmt_data in info['formats']:
                     if fmt_data.get('fps'):
                         fps = fmt_data['fps']
                         break

            print(f"[Source] 成功解析 YouTube 串流: {stream_url}")
            return stream_url, fps
            
        except yt_dlp.DownloadError as e:
            raise RuntimeError(f"yt-dlp 解析 YouTube 連結失敗: {e}")
        except Exception as e:
            raise RuntimeError(f"yt-dlp 發生未知錯誤: {e}")


def resolve_source(source: str, prefer_height: int = 480) -> Tuple[cv2.VideoCapture, Optional[float]]:
    """
    根據輸入字串解析並開啟視訊源 (Webcam, File, URL, YouTube)。
    
    傳回：(cv2.VideoCapture instance, fps)
    """
    source_url = source
    source_fps: Optional[float] = None
    
    try:
        # 1. 嘗試解析為整數 (Webcam 索引)
        idx = int(source)
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # 嘗試設定一個高解析度
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        print(f"[Source] 開啟 Webcam {idx}, FPS: {source_fps:.2f}")
        
    except ValueError:
        # 2. 檔案或 URL
        if is_youtube(source):
            # 處理 YouTube 連結，取得串流 URL 和 FPS
            source_url, source_fps = get_youtube_stream(source, prefer_height=prefer_height)
        
        # 3. 開啟視訊檔案/串流
        cap = cv2.VideoCapture(source_url)
        source_fps = source_fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        print(f"[Source] 開啟檔案/串流 {source_url}, FPS: {source_fps:.2f}")
    
    if not cap.isOpened():
         raise FileNotFoundError(f"[Source] 無法開啟視訊源：{source}")
         
    return cap, source_fps

# -------------------------
# FrameGrabber 執行緒
# -------------------------

class FrameGrabber(threading.Thread):
    """
    一個獨立的執行緒，用於從視訊源中讀取幀，並進行 FPS 節流。
    將讀取與主推論迴圈分離，避免 I/O 阻塞。
    (Adapted from original script's FrameGrabber)
    """
    def __init__(self, cap: cv2.VideoCapture, fps: float, process_every: int = 1, queue_size: int = 1):
        super().__init__()
        self.daemon = True
        self.cap = cap
        self.fps = fps
        self.dt = 1.0 / self.fps # 每個幀的目標時間間隔
        self.process_every = max(1, process_every) # 處理間隔
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.start_time = time.perf_counter()
        
        print(f"[Grabber] 來源 FPS: {self.fps:.2f}, 讀取間隔: {self.dt * self.process_every:.4f}s")
        
    def run(self):
        """執行緒主迴圈：持續讀取幀並放入佇列"""
        last_grab = time.perf_counter()
        frame_id = 0
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                print("[Grabber] 視訊源已結束或讀取失敗。")
                break

            # 節流邏輯：只有在需要處理的幀才讀取並進行節流等待
            if frame_id % self.process_every == 0:
                
                # 放入佇列：如果佇列滿了，丟棄最舊的幀
                if self.frame_queue.full():
                    try: self.frame_queue.get_nowait()
                    except queue.Empty: pass
                self.frame_queue.put(frame)

                # 節流等待
                now = time.perf_counter()
                sleep = (last_grab + self.dt * self.process_every) - now
                if sleep > 0:
                    time.sleep(sleep)
                last_grab = time.perf_counter()
            
            frame_id += 1

    def grab_frame(self) -> Optional[cv2.Mat]:
        """從佇列中取出一個幀供主程式使用"""
        try:
            # 使用非阻塞獲取
            return self.frame_queue.get(timeout=0.005) 
        except queue.Empty:
            return None

    def stop(self):
        """停止執行緒並釋放視訊源"""
        self.stopped = True
        self.join()
        if self.cap:
            self.cap.release()

# end of video_source.py