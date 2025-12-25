"""
TEMPLATE: Tính năng tính khoảng cách và phát âm thanh cho Object Detection
Các thành phần chính từ dự án hiện tại để tích hợp vào dự án khác
"""

import pyttsx3
import cv2
import numpy as np
import time
from threading import Thread
from queue import Queue

# ================================
# 1. PHẦN PHÁT ÂM THANH (Text-to-Speech)
# ================================

def setup_audio_system():
    """Khởi tạo hệ thống phát âm thanh"""
    global queue, audio_thread
    
    def speak(q):
        engine = pyttsx3.init()
        engine.setProperty('rate', 235)  # Tốc độ nói
        engine.setProperty('volume', 1.0)  # Âm lượng

        while True:
            if not q.empty():
                label, distance, position = q.get()
                rounded_distance = round(distance * 2) / 2  # Làm tròn 0.5
                
                # Chuyển đổi số thành chuỗi (bỏ .0 nếu là số nguyên)
                rounded_distance_str = str(int(rounded_distance)) if rounded_distance.is_integer() else str(rounded_distance)
                
                # Phát âm thanh thông báo
                if label in class_avg_sizes:
                    engine.say(f"{label} IS {rounded_distance_str} METERS ON {position}")
                    engine.runAndWait()
                
                # Xóa queue sau khi phát
                with q.mutex:
                    q.queue.clear()
            else:
                time.sleep(0.1)  # Tránh busy waiting
    
    # Tạo queue và thread cho audio
    queue = Queue()
    audio_thread = Thread(target=speak, args=(queue,))
    audio_thread.daemon = True  # Thread sẽ dừng khi main program dừng
    audio_thread.start()
    
    return queue

# ================================
# 2. PHẦN TÍNH TOÁN KHOẢNG CÁCH
# ================================

def calculate_distance(box, frame_width, class_avg_sizes, result):
    """
    Tính khoảng cách dựa trên độ rộng vật thể trong frame
    
    Args:
        box: YOLO detection box
        frame_width: Chiều rộng của frame
        class_avg_sizes: Dictionary chứa width_ratio cho các class
        result: YOLO result để lấy tên class
    
    Returns:
        float: Khoảng cách tính bằng mét
    """
    # Tính độ rộng vật thể trong pixel
    object_width = box.xyxy[0, 2].item() - box.xyxy[0, 0].item()
    
    # Lấy tên class
    label = result.names[box.cls[0].item()]
    
    # Áp dụng hệ số hiệu chỉnh cho từng loại vật thể
    if label in class_avg_sizes:
        object_width *= class_avg_sizes[label]["width_ratio"]
    
    # Công thức tính khoảng cách dựa trên FOV 70 độ
    distance = (frame_width * 0.5) / np.tan(np.radians(70 / 2)) / (object_width + 1e-6)
    return round(distance, 2)

def get_position(frame_width, box):
    """
    Xác định vị trí của vật thể (LEFT/FORWARD/RIGHT)
    
    Args:
        frame_width: Chiều rộng frame
        box: Tọa độ bounding box [x1, y1, x2, y2]
    
    Returns:
        str: "LEFT", "FORWARD", hoặc "RIGHT"
    """
    x_center = box[0]  # Tọa độ x của góc trái trên
    
    if x_center < frame_width // 3:
        return "LEFT"
    elif x_center < 2 * (frame_width // 3):
        return "FORWARD"
    else:
        return "RIGHT"

def blur_person(image, box):
    """
    Làm mờ khuôn mặt người để bảo vệ privacy
    
    Args:
        image: Frame ảnh
        box: YOLO detection box
    
    Returns:
        image: Frame đã được làm mờ
    """
    x, y, w, h = box.xyxy[0].cpu().numpy().astype(int)
    
    # Làm mờ phần đầu (8% chiều cao từ trên xuống)
    top_region = image[y:y+int(0.08 * h), x:x+w]
    blurred_top_region = cv2.GaussianBlur(top_region, (15, 15), 0)
    image[y:y+int(0.08 * h), x:x+w] = blurred_top_region
    
    return image

# ================================
# 3. CẤU HÌNH CÁC LOẠI VẬT THỂ
# ================================

class_avg_sizes = {
    "person": {"width_ratio": 2.5},
    "car": {"width_ratio": 0.37},
    "bicycle": {"width_ratio": 2.3},
    "motorcycle": {"width_ratio": 2.4},
    "bus": {"width_ratio": 0.3},
    "traffic light": {"width_ratio": 2.95},
    "stop sign": {"width_ratio": 2.55},
    "bench": {"width_ratio": 1.6},
    "cat": {"width_ratio": 1.9},
    "dog": {"width_ratio": 1.5},
}

# ================================
# 4. HÀM XỬ LÝ DETECTION VÀ HIỂN THỊ
# ================================

def process_detections(results, frame, queue, class_avg_sizes):
    """
    Xử lý các detection và hiển thị trên frame
    
    Args:
        results: YOLO results
        frame: Frame hiện tại
        queue: Audio queue
        class_avg_sizes: Dictionary size configs
    
    Returns:
        frame: Frame đã được xử lý
    """
    result = results[0]
    nearest_object = None
    min_distance = float('inf')
    
    # Định nghĩa màu sắc
    colorGreen = (0, 255, 0)    # Người - Xanh lá
    colorYellow = (0, 255, 255) # Xe hơi - Vàng
    colorBlue = (255, 0, 0)     # Vật thể khác - Xanh dương
    colorRed = (0, 0, 255)      # Vật gần nhất - Đỏ
    thickness = 2

    # Xử lý từng detection
    for box in result.boxes:
        label = result.names[box.cls[0].item()]
        cords = [round(x) for x in box.xyxy[0].tolist()]
        
        # Tính khoảng cách
        distance = calculate_distance(box, frame.shape[1], class_avg_sizes, result)
        
        # Tìm vật thể gần nhất
        if distance < min_distance:
            min_distance = distance
            nearest_object = (label, round(distance, 1), cords)
        
        # Vẽ bounding box và text theo loại vật thể
        if label == "person":
            frame = blur_person(frame, box)  # Làm mờ khuôn mặt
            cv2.rectangle(frame, (cords[0], cords[1]), (cords[2], cords[3]), colorGreen, thickness)
            cv2.putText(frame, f"{label} - {distance:.1f}m", (cords[0], cords[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colorGreen, thickness)
                       
        elif label == "car":
            cv2.rectangle(frame, (cords[0], cords[1]), (cords[2], cords[3]), colorYellow, thickness)
            cv2.putText(frame, f"{label} - {distance:.1f}m", (cords[0], cords[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colorYellow, thickness)
                       
        elif label in class_avg_sizes:
            cv2.rectangle(frame, (cords[0], cords[1]), (cords[2], cords[3]), colorBlue, thickness)
            cv2.putText(frame, f"{label} - {distance:.1f}m", (cords[0], cords[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colorBlue, thickness)

    # Xử lý vật thể gần nhất
    if nearest_object:
        if nearest_object[0] in class_avg_sizes:
            # Vẽ bounding box đỏ cho vật gần nhất
            cv2.rectangle(frame, (nearest_object[2][0], nearest_object[2][1]),
                         (nearest_object[2][2], nearest_object[2][3]), colorRed, thickness)
            text = f"{nearest_object[0]} - {round(nearest_object[1], 1)}m"
            cv2.putText(frame, text, (nearest_object[2][0], nearest_object[2][1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, colorRed, thickness)
            
            # Phát âm thanh nếu vật thể gần (≤ 12.5m)
            if nearest_object[1] <= 12.5:
                position = get_position(frame.shape[1], nearest_object[2])
                queue.put((nearest_object[0], nearest_object[1], position))
    
    return frame

# ================================
# 5. TEMPLATE SỬ DỤNG TRONG DỰ ÁN KHÁC
# ================================

def integrate_distance_audio(your_model, your_video_source):
    """
    Template tích hợp vào dự án khác
    
    Args:
        your_model: YOLO model của bạn
        your_video_source: Nguồn video của bạn
    """
    
    # 1. Khởi tạo audio system
    audio_queue = setup_audio_system()
    
    # 2. Main loop
    while True:
        # Đọc frame từ video source của bạn
        ret, frame = your_video_source.read()
        if not ret:
            break
            
        # Chạy YOLO detection
        results = your_model.predict(frame)
        
        # Xử lý detection với tính năng khoảng cách và audio
        frame = process_detections(results, frame, audio_queue, class_avg_sizes)
        
        # Hiển thị frame
        cv2.imshow('Object Detection with Distance & Audio', frame)
        
        # Xử lý phím nhấn
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()

# ================================
# 6. HƯỚNG DẪN SỬ DỤNG
# ================================

"""
CÁCH TÍCH HỢP VÀO DỰ ÁN CỦA BẠN:

1. Import các hàm cần thiết:
   from distance_audio_template import setup_audio_system, process_detections, class_avg_sizes

2. Trong main loop của bạn:
   audio_queue = setup_audio_system()  # Chỉ gọi 1 lần
   
   while True:
       ret, frame = cap.read()
       results = model.predict(frame)
       
       # Thêm dòng này để có tính năng khoảng cách + audio
       frame = process_detections(results, frame, audio_queue, class_avg_sizes)
       
       cv2.imshow('window', frame)
       # ... phần còn lại

3. Yêu cầu thư viện:
   pip install pyttsx3 ultralytics opencv-python numpy

4. Tùy chỉnh:
   - Sửa class_avg_sizes để phù hợp với model của bạn
   - Thay đổi ngưỡng khoảng cách (hiện tại 12.5m)
   - Điều chỉnh tốc độ/âm lượng trong setup_audio_system()
"""
