import cv2
import numpy as np
import onnxruntime as ort
import time
import os
import pyttsx3
from threading import Thread
from queue import Queue

# --- 1. CẤU HÌNH ---
USE_CAM = True  # Chuyển sang True để sử dụng camera
VIDEO_PATH = "./sample_video.mp4"  # Đường dẫn video mẫu (nếu có)

# DLL Paths - Không cần thiết trên macOS
# Chỉ sử dụng trên Windows với CUDA
# cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
# cudnn_bin = r"C:\Program Files\NVIDIA\CUDNN\v9.17\bin"
# if os.path.exists(cuda_bin): os.add_dll_directory(cuda_bin)
# if os.path.exists(cudnn_bin): os.add_dll_directory(cudnn_bin)

# --- 2. KHỞI TẠO BẢNG LUT (GAMMA CORRECTION) ---
# Gamma < 1.0 làm sáng vùng tối mà không cháy vùng sáng. 0.8 là mức tối ưu.
gamma = 0.8
invGamma = 1.0 / gamma
lut_table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

# --- 3. KHỞI TẠO MODEL ---
# Ưu tiên CPU trên macOS, GPU nếu có
providers = ['CPUExecutionProvider']
# Nếu có GPU AMD/Metal trên Mac M1/M2, có thể thêm 'CoreMLExecutionProvider'
session_coco = None
session_custom = None 

try:
    opt = ort.SessionOptions()
    session_coco = ort.InferenceSession("yolov10s.onnx", providers=providers, sess_options=opt)
    # session_custom = ort.InferenceSession("best.onnx", providers=providers, sess_options=opt)
    print("✅ Hệ thống đã sẵn sàng với CPU")
except Exception as e:
    print(f"❌ Lỗi: {e}"); exit()

class_names_coco = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
class_names_custom = ['door', 'trash_can']

# --- 4. HỆ THỐNG ÂM THANH VÀ TÍNH KHOẢNG CÁCH ---

# Cấu hình kích thước trung bình cho các loại vật thể (để tính khoảng cách)
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

# Biến global cho hệ thống âm thanh
audio_queue = None
audio_thread = None

def setup_audio_system():
    """Khởi tạo hệ thống phát âm thanh"""
    global audio_queue, audio_thread
    
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
    audio_queue = Queue()
    audio_thread = Thread(target=speak, args=(audio_queue,))
    audio_thread.daemon = True  # Thread sẽ dừng khi main program dừng
    audio_thread.start()
    
    return audio_queue

def calculate_distance(object_width, frame_width, label):
    """
    Tính khoảng cách dựa trên độ rộng vật thể trong frame
    """
    # Áp dụng hệ số hiệu chỉnh cho từng loại vật thể
    if label in class_avg_sizes:
        object_width *= class_avg_sizes[label]["width_ratio"]
    
    # Công thức tính khoảng cách dựa trên FOV 70 độ
    distance = (frame_width * 0.5) / np.tan(np.radians(70 / 2)) / (object_width + 1e-6)
    return round(distance, 2)

def get_position(frame_width, x_center):
    """Xác định vị trí của vật thể (LEFT/FORWARD/RIGHT)"""
    if x_center < frame_width // 3:
        return "LEFT"
    elif x_center < 2 * (frame_width // 3):
        return "FORWARD"
    else:
        return "RIGHT"

def blur_person_face(frame, x1, y1, x2, y2):
    """Làm mờ khuôn mặt người để bảo vệ privacy"""
    h = y2 - y1
    # Làm mờ phần đầu (8% chiều cao từ trên xuống)
    face_y2 = y1 + int(0.08 * h)
    if face_y2 > y1:
        face_region = frame[y1:face_y2, x1:x2]
        if face_region.size > 0:
            blurred_face = cv2.GaussianBlur(face_region, (15, 15), 0)
            frame[y1:face_y2, x1:x2] = blurred_face
    return frame

# --- BIẾN LƯU TRỮ ---
last_custom_results = [] 
frame_count = 0

# --- KHỞI TẠO HỆ THỐNG ÂM THANH ---
print("🔊 Đang khởi tạo hệ thống âm thanh...")
audio_queue = setup_audio_system()
print("✅ Hệ thống âm thanh đã sẵn sàng")

cap = cv2.VideoCapture(0 if USE_CAM else VIDEO_PATH)
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Ngưỡng Y để xét Gần/Xa (Cạnh dưới khung hình > 75% chiều cao ảnh là Rất Gần)
danger_line = int(orig_h * 0.75)

# Cấu hình kích thước trung bình cho các loại vật thể (để tính khoảng cách)
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

# Biến global cho hệ thống âm thanh
audio_queue = None
audio_thread = None

def setup_audio_system():
    """Khởi tạo hệ thống phát âm thanh"""
    global audio_queue, audio_thread
    
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
    audio_queue = Queue()
    audio_thread = Thread(target=speak, args=(audio_queue,))
    audio_thread.daemon = True  # Thread sẽ dừng khi main program dừng
    audio_thread.start()
    
    return audio_queue

def calculate_distance(object_width, frame_width, label):
    """
    Tính khoảng cách dựa trên độ rộng vật thể trong frame
    """
    # Áp dụng hệ số hiệu chỉnh cho từng loại vật thể
    if label in class_avg_sizes:
        object_width *= class_avg_sizes[label]["width_ratio"]
    
    # Công thức tính khoảng cách dựa trên FOV 70 độ
    distance = (frame_width * 0.5) / np.tan(np.radians(70 / 2)) / (object_width + 1e-6)
    return round(distance, 2)

def get_position(frame_width, x_center):
    """Xác định vị trí của vật thể (LEFT/FORWARD/RIGHT)"""
    if x_center < frame_width // 3:
        return "LEFT"
    elif x_center < 2 * (frame_width // 3):
        return "FORWARD"
    else:
        return "RIGHT"

def blur_person_face(frame, x1, y1, x2, y2):
    """Làm mờ khuôn mặt người để bảo vệ privacy"""
    h = y2 - y1
    # Làm mờ phần đầu (8% chiều cao từ trên xuống)
    face_y2 = y1 + int(0.08 * h)
    if face_y2 > y1:
        face_region = frame[y1:face_y2, x1:x2]
        if face_region.size > 0:
            blurred_face = cv2.GaussianBlur(face_region, (15, 15), 0)
            frame[y1:face_y2, x1:x2] = blurred_face
    return frame

# Biến lưu trữ vật thể gần nhất để phát âm thanh
nearest_object_info = {"label": None, "distance": float('inf'), "last_announcement": 0}

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_count += 1
    start_time = time.time()

    # --- BƯỚC 3: TIỀN XỬ LÝ (TỐI ƯU REAL-TIME) ---
    # 1. Resize trước để giảm tải cho các bước sau
    img_640 = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_CUBIC)
    
    # 2. Áp dụng LUT thay cho convertScaleAbs (Xử lý adaptive sáng/tối)
    img_640 = cv2.LUT(img_640, lut_table)
    
    # 3. Chuyển đổi format cho AI
    blob = img_640.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0)

    # --- BƯỚC 4: CHẠY INFERENCE ---
    # Model COCO: Chạy 100% frame
    results_coco = session_coco.run(None, {session_coco.get_inputs()[0].name: blob})[0][0]
    
    # Model Custom: Skip frame (Chạy frame chẵn, hiển thị kết quả cũ ở frame lẻ)
    if session_custom is not None:
        if frame_count % 2 == 0:
            last_custom_results = session_custom.run(None, {session_custom.get_inputs()[0].name: blob})[0][0]

    # --- BƯỚC 5: VẼ KẾT QUẢ VÀ TÍNH KHOẢNG CÁCH ---
    nearest_object = None
    min_distance = float('inf')
    current_time = time.time()
    
    # Vẽ COCO với tính khoảng cách
    for pred in results_coco:
        x1, y1, x2, y2, score, cls_id = pred
        if score > 0.35:
            ix1, iy1 = int(x1 * orig_w / 640), int(y1 * orig_h / 640)
            ix2, iy2 = int(x2 * orig_w / 640), int(y2 * orig_h / 640)
            
            label = class_names_coco[int(cls_id)]
            
            # Tính khoảng cách nếu vật thể trong danh sách hỗ trợ
            if label in class_avg_sizes:
                object_width = ix2 - ix1
                distance = calculate_distance(object_width, orig_w, label)
                
                # Tìm vật thể gần nhất
                if distance < min_distance:
                    min_distance = distance
                    x_center = (ix1 + ix2) // 2
                    nearest_object = (label, distance, x_center)
                
                # Làm mờ khuôn mặt nếu là người
                if label == "person":
                    frame = blur_person_face(frame, ix1, iy1, ix2, iy2)
                
                # Chọn màu dựa trên khoảng cách thay vì vị trí Y
                if distance <= 3.0:  # Rất gần - Đỏ
                    color = (0, 0, 255)
                elif distance <= 8.0:  # Gần - Cam
                    color = (0, 165, 255)
                else:  # Xa - Xanh lá
                    color = (0, 255, 0)
                
                display_label = f"{label} - {distance:.1f}m"
            else:
                # Sử dụng logic cũ cho vật thể không hỗ trợ tính khoảng cách
                color = (0, 0, 255) if iy2 > danger_line else (0, 255, 0)
                display_label = f"{label} {'(NEAR)' if iy2 > danger_line else ''}"
            
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, 2)
            cv2.putText(frame, display_label, (ix1, iy1-10), 0, 0.5, color, 2)

    # Vẽ Custom (Màu Vàng - Chống chớp tắt)
    for pred in last_custom_results:
        x1, y1, x2, y2, score, cls_id = pred
        if score > 0.25:
            ix1, iy1 = int(x1 * orig_w / 640), int(y1 * orig_h / 640)
            ix2, iy2 = int(x2 * orig_w / 640), int(y2 * orig_h / 640)
            
            # Cảnh báo gần cho vật thể Custom
            color = (0, 0, 255) if iy2 > danger_line else (0, 255, 255)
            
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, 2)
            cv2.putText(frame, f"TARGET: {class_names_custom[int(cls_id)]}", (ix1, iy1-10), 0, 0.5, color, 2)

    # Xử lý âm thanh cho vật thể gần nhất
    if nearest_object and nearest_object[1] <= 12.5:  # Chỉ thông báo vật thể ≤ 12.5m
        # Tránh spam âm thanh - chỉ phát mỗi 3 giây
        if (nearest_object_info["label"] != nearest_object[0] or 
            abs(nearest_object_info["distance"] - nearest_object[1]) > 1.0 or
            current_time - nearest_object_info["last_announcement"] > 3.0):
            
            position = get_position(orig_w, nearest_object[2])
            audio_queue.put((nearest_object[0], nearest_object[1], position))
            
            nearest_object_info["label"] = nearest_object[0]
            nearest_object_info["distance"] = nearest_object[1]
            nearest_object_info["last_announcement"] = current_time

    # Hiển thị FPS và thông tin
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.1f} | CPU Mode", (20, 40), 0, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("DIPR Project - Blind Support", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()