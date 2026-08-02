"""
===============================================================================
AI THERMAL TOMATO ANALYZER - DEEP LEARNING PREDICTION MODULE
===============================================================================
Mô tả: 
Chạy suy luận (Inference) bằng mô hình YOLO Segmentation. 
Tạo ảnh overlay, mask nhị phân và trích xuất tham số nông nghiệp.
(Đã lược bỏ OpenCV vẽ thủ công, tối ưu bằng core của Ultralytics & Numpy)
===============================================================================
"""

import cv2
import numpy as np
import torch
import logging
import time
from ultralytics import YOLO

# --- THÊM DÒNG NÀY ĐỂ BỎ QUA LỖI TORCH SERIALIZATION ---
from ultralytics.nn.tasks import SegmentationModel
torch.serialization.add_safe_globals([SegmentationModel])
# -----------------------------------------------------

# Cấu hình logging
logger = logging.getLogger('ThermalAnalyzer.Predict')

class ThermalImageAnalyzer:
    def __init__(self, model_path: str):
        self.model_path = model_path
        logger.info(f"Đang tải mô hình học sâu từ: {model_path} ...")
        self.model = YOLO(model_path)
        device = 'CUDA' if torch.cuda.is_available() else 'CPU'
        logger.info(f"Đã tải thành công. Đang chạy trên thiết bị: {device}")

    def process(self, image_path: str):
        start_time = time.time()
        CONF_THRESHOLD = 0.6
        
        # 1. Đọc ảnh
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            raise ValueError(f"Không thể đọc file ảnh: {image_path}")
            
        h, w = orig_img.shape[:2]
        total_pixels = h * w
        
        # 2. Chạy model YOLO
        results = self.model.predict(
            source=orig_img,
            conf=CONF_THRESHOLD,
            imgsz=640,
            save=False,
            verbose=False
        )
        result = results[0]
        
        # Biến mặc định
        binary_mask = np.zeros((h, w), dtype=np.uint8)
        seg_img = orig_img.copy()
        detected_regions = 0
        mean_conf = 0.0
        mask_pixels = 0
        
        # 3. Xử lý kết quả (KHÔNG DÙNG OpenCV truyền thống)
        if result.masks is not None and len(result.masks.data) > 0:
            detected_regions = len(result.boxes)
            mean_conf = float(result.boxes.conf.mean().cpu().numpy())
            
            # --- TẠO ẢNH OVERLAY BẰNG ENGINE CỦA YOLO ---
            # Dùng luôn tính năng plot() của YOLO, tự động bo viền, làm mờ, dán nhãn cực đẹp
            seg_img = result.plot(labels=False, boxes=False, conf=False)
            
            # --- TẠO BINARY MASK BẰNG TENSOR/NUMPY (Nhanh hơn cv2.fillPoly) ---
            # Lấy tensor mask từ YOLO, gộp tất cả các vùng lá lại thành 1 mask duy nhất
            mask_tensor = torch.any(result.masks.data, dim=0).byte().cpu().numpy()
            
            # Đảm bảo kích thước mask khớp với kích thước ảnh gốc
            if mask_tensor.shape != (h, w):
                binary_mask = cv2.resize(mask_tensor, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                binary_mask = mask_tensor
                
            binary_mask = binary_mask * 255 # Đổi [0,1] thành [0, 255] cho ảnh xám
            mask_pixels = int(np.count_nonzero(binary_mask))

        # 4. Tính toán Metrics
        coverage = round((mask_pixels / total_pixels) * 100, 2) if total_pixels > 0 else 0
        inference_time = round((time.time() - start_time) * 1000, 2)
        
        metrics = {
            "leaf_area": mask_pixels,
            "coverage": coverage,
            "detected_regions": detected_regions,
            "confidence": round(mean_conf, 3),
            "inference_time": inference_time
        }
        
        return seg_img, binary_mask, metrics
