"""
===============================================================================
AI THERMAL TOMATO ANALYZER - DEEP LEARNING PREDICTION MODULE
===============================================================================
Mô tả: 
Chạy suy luận (Inference) bằng mô hình YOLO Segmentation. 
Tạo ảnh overlay, mask nhị phân và trích xuất tham số nông nghiệp.
===============================================================================
"""

import cv2
import numpy as np
import torch
import logging
import time
from ultralytics import YOLO

# Cấu hình logging
logger = logging.getLogger('ThermalAnalyzer.Predict')

class ThermalImageAnalyzer:
    def __init__(self, model_path: str):
        """
        Khởi tạo và load mô hình YOLO vào RAM.
        Giúp tối ưu hóa thời gian xử lý API, không phải load lại model mỗi khi upload ảnh.
        """
        self.model_path = model_path
        logger.info(f"Đang tải mô hình học sâu từ: {model_path} ...")
        
        # Khởi tạo mô hình YOLO Segmentation
        self.model = YOLO(model_path)
        
        # Kiểm tra thiết bị phần cứng (GPU/CPU)
        device = 'CUDA' if torch.cuda.is_available() else 'CPU'
        logger.info(f"Đã tải thành công mô hình. Đang chạy trên thiết bị: {device}")

    def process(self, image_path: str):
        """
        Đọc ảnh nhiệt, chạy qua mô hình YOLO-Seg và trả về:
        - Ảnh BGR đã vẽ Overlay (Segmentation)
        - Ảnh Grayscale (Binary Mask)
        - Bộ chỉ số đo đạc (Metrics Dictionary)
        """
        start_time = time.time()
        
        # ==========================================
        # 1. CẤU HÌNH THAM SỐ
        # ==========================================
        CONF_THRESHOLD = 0.6        # Ngưỡng tin cậy (Confidence score) >= 60%
        
        # 2. ĐỌC ẢNH ĐẦU VÀO
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            raise ValueError(f"Không thể đọc file ảnh tại đường dẫn: {image_path}")
        
        h, w = orig_img.shape[:2]
        total_pixels = h * w
        
        # 3. GỌI MÔ HÌNH VÀ DỰ ĐOÁN (INFERENCE)
        # Tắt tính năng vẽ ảnh mặc định của Ultralytics để ta tự vẽ custom UI
        results = self.model.predict(
            source=orig_img,
            conf=CONF_THRESHOLD,
            imgsz=640,
            save=False,
            verbose=False
        )
        
        result = results[0] # Lấy kết quả của bức ảnh đầu tiên
        
        # 4. TẠO CÁC BIẾN KẾT QUẢ KHỞI TẠO TRỐNG
        binary_mask = np.zeros((h, w), dtype=np.uint8)
        seg_img = orig_img.copy()
        
        detected_regions = 0
        mean_conf = 0.0
        
        # 5. XỬ LÝ KẾT QUẢ PHÂN ĐOẠN (MASK & BOUNDING BOX)
        if result.masks is not None and result.boxes is not None:
            detected_regions = len(result.boxes)
            
            # Tính toán độ tin cậy trung bình (Thang đo 0.0 -> 1.0)
            confs = result.boxes.conf.cpu().numpy()
            if len(confs) > 0:
                mean_conf = float(np.mean(confs))
            
            # YOLO trả về tọa độ các đường viền đa giác (Polygons) của tán lá
            for xy in result.masks.xy:
                # Chuyển đổi tọa độ sang int32 theo chuẩn của OpenCV
                pts = np.array(xy, dtype=np.int32)
                
                # Tô màu trắng (255) cho vùng bên trong đa giác trên Binary Mask
                cv2.fillPoly(binary_mask, [pts], 255)
                
            # ==============================================
            # 6. VẼ HIỆU ỨNG OVERLAY DẠ QUANG CHO ẢNH KẾT QUẢ
            # ==============================================
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            overlay = orig_img.copy()
            # Tô màu xanh lá vào vùng tán cây trên ảnh overlay
            cv2.drawContours(overlay, contours, -1, (36, 179, 0), -1)
            
            # Hòa trộn màu (Alpha Blending) tạo độ trong suốt 40% cho vùng phủ
            cv2.addWeighted(overlay, 0.4, seg_img, 0.6, 0, seg_img)
            
            # Vẽ nét đứt/liền bo viền sắc nét màu xanh lá mạ
            cv2.drawContours(seg_img, contours, -1, (0, 255, 64), 2)

        # ==============================================
        # 7. TÍNH TOÁN CÁC CHỈ SỐ NÔNG NGHIỆP TỪ BINARY MASK
        # ==============================================
        mask_pixels = int(np.count_nonzero(binary_mask))
        
        # Tỷ lệ che phủ = (Diện tích lá / Tổng diện tích khung hình) * 100
        coverage = round((mask_pixels / total_pixels) * 100, 2)
        
        # TÍNH DIỆN TÍCH LÁ THEO ĐƠN VỊ PIXEL (Số đếm thô)
        leaf_area = mask_pixels 
        
        inference_time = round((time.time() - start_time) * 1000, 2)
        
        metrics = {
            "leaf_area": leaf_area, 
            "coverage": coverage,
            "detected_regions": detected_regions,
            "confidence": round(mean_conf, 3),
            "inference_time": inference_time
        }
        
        # 8. TRẢ VỀ THEO ĐÚNG TIÊU CHUẨN CỦA APP.PY
        return seg_img, binary_mask, metrics
