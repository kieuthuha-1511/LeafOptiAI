# 4. TẠO CÁC BIẾN KẾT QUẢ KHỞI TẠO TRỐNG
    binary_mask = np.zeros((h, w), dtype=np.uint8)
    seg_img = orig_img.copy()
    
    detected_regions = 0
    mean_conf = 0.0
    bounding_boxes = [] # Khởi tạo danh sách chứa tọa độ bounding box
    
    # 5. XỬ LÝ KẾT QUẢ VÙNG PHÁT HIỆN (BOUNDING BOXES)
    if result.boxes is not None and len(result.boxes) > 0:
        detected_regions = len(result.boxes)
        
        # Tính toán độ tin cậy trung bình (Thang đo 0.0 -> 1.0)
        confs = result.boxes.conf.cpu().numpy()
        if len(confs) > 0:
            mean_conf = float(np.mean(confs))
        
        overlay = orig_img.copy()
        
        # Duyệt qua các khung hình chữ nhật phát hiện được
        for box in result.boxes:
            # Lấy tọa độ [x1, y1, x2, y2]
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            # Lưu tọa độ Bounding Box
            bounding_boxes.append({
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1
            })
            
            # Tô màu trắng (255) cho vùng Bounding Box trên Binary Mask
            cv2.rectangle(binary_mask, (x1, y1), (x2, y2), 255, -1)
            
            # Tô màu xanh lá vào vùng Bounding Box trên ảnh overlay
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (36, 179, 0), -1)
        
        # ==============================================
        # 6. VẼ HIỆU ỨNG OVERLAY DẠ QUANG CHO ẢNH KẾT QUẢ
        # ==============================================
        # Hòa trộn màu (Alpha Blending) tạo độ trong suốt 40% cho vùng Bounding Box
        cv2.addWeighted(overlay, 0.4, seg_img, 0.6, 0, seg_img)
        
        # Vẽ khung viền sắc nét màu xanh lá mạ
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cv2.rectangle(seg_img, (x1, y1), (x2, y2), (0, 255, 64), 2)

    # 7. TÍNH TOÁN CÁC CHỈ SỐ NÔNG NGHIỆP
    mask_pixels = int(np.count_nonzero(binary_mask))
    coverage = round((mask_pixels / total_pixels) * 100, 2)
    leaf_area = mask_pixels # Sử dụng trực tiếp pixels Bounding Box
    
    # Đóng gói dữ liệu (inference_time sẽ được app.py tự động đo và ghi đè thêm)
    metrics = {
        "leaf_area": leaf_area,
        "coverage": coverage,
        "detected_regions": detected_regions,
        "confidence": round(mean_conf, 3),
        "bounding_boxes": bounding_boxes # Trả về mảng bounding box
    }
    
    # 8. TRẢ VỀ THEO ĐÚNG TIÊU CHUẨN CỦA APP.PY
    return seg_img, binary_mask, metrics
"""
===============================================================================
AI THERMAL TOMATO ANALYZER - DEEP LEARNING PREDICTION MODULE
===============================================================================
Mô tả: 
Chạy suy luận (Inference) bằng mô hình YOLO Object Detection. 
Tạo ảnh overlay bounding box, mask nhị phân dạng hình chữ nhật và trích xuất tham số nông nghiệp.
===============================================================================
"""

import cv2
import numpy as np
import torch
import logging
from ultralytics import YOLO

# Cấu hình logging
logger = logging.getLogger('ThermalAnalyzer.Predict')

# ==========================================
# CƠ CHẾ CACHE MÔ HÌNH (Tránh load lại nhiều lần)
# ==========================================
_model_instance = None
_current_model_path = None

def get_yolo_model(model_path: str):
    """
    Load mô hình YOLO và lưu vào cache (Singleton pattern).
    Giúp tối ưu hóa thời gian xử lý API, không phải load lại model mỗi khi upload ảnh.
    """
    global _model_instance, _current_model_path
    
    if _model_instance is None or _current_model_path != model_path:
        logger.info(f"Đang tải mô hình học sâu từ: {model_path} ...")
        # Khởi tạo mô hình YOLO Object Detection
        _model_instance = YOLO(model_path)
        _current_model_path = model_path
        
        # Kiểm tra thiết bị phần cứng (GPU/CPU)
        device = 'CUDA' if torch.cuda.is_available() else 'CPU'
        logger.info(f"Đã tải thành công mô hình. Đang chạy trên thiết bị: {device}")
        
    return _model_instance

# ==========================================
# HÀM XỬ LÝ CHÍNH ĐƯỢC GỌI TỪ APP.PY
# ==========================================
def predict_thermal_image(image_path: str, model_path: str):
    """
    Đọc ảnh nhiệt, chạy qua mô hình YOLO-Detect và trả về:
    - Ảnh BGR đã vẽ Overlay Bounding Box (Object Detection)
    - Ảnh Grayscale (Binary Mask dạng Bounding Box)
    - Bộ chỉ số đo đạc (Metrics Dictionary)
    """
    
    # 1. CẤU HÌNH THAM SỐ
    CONF_THRESHOLD = 0.6        # Ngưỡng tin cậy (Confidence score) >= 60%
    
    # 2. ĐỌC ẢNH ĐẦU VÀO
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise ValueError(f"Không thể đọc file ảnh tại đường dẫn: {image_path}")
    
    h, w = orig_img.shape[:2]
    total_pixels = h * w
    
    # 3. GỌI MÔ HÌNH VÀ DỰ ĐOÁN (INFERENCE)
    model = get_yolo_model(model_path)
    
    # Tắt tính năng vẽ ảnh mặc định của Ultralytics để tự vẽ custom UI
    results = model.predict(
        source=orig_img,
        conf=CONF_THRESHOLD,
        imgsz=640,
        save=False,
        verbose=False
    )
    
    result = results[0] # Lấy kết quả của bức ảnh đầu tiên
