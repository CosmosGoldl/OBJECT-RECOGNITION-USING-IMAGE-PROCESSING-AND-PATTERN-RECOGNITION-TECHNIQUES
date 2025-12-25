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

# --- 2. KHỞI TẠO BẢNG LUT (GAMMA CORRECTION) ---
gamma = 0.8
invGamma = 1.0 / gamma
lut_table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

# --- 3. KHỞI TẠO DUAL MODEL ---
providers = ['CPUExecutionProvider']
session_coco = None
session_custom = None

try:
    opt = ort.SessionOptions()
    # Load COCO model (80 classes)
    session_coco = ort.InferenceSession("yolov10s.onnx", providers=providers, sess_options=opt)
    print("✅ COCO Model (yolov10s.onnx) loaded successfully")
    
    # Load Custom model (door, trash_can)
    session_custom = ort.InferenceSession("best.onnx", providers=providers, sess_options=opt)
    print("✅ Custom Model (best.onnx) loaded successfully")
    
    print("🚀 DUAL MODEL SYSTEM READY!")
except Exception as e:
    print(f"❌ Lỗi loading models: {e}"); exit()

# --- 4. DANH SÁCH CLASSES ---
class_names_coco = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Custom model classes (từ best.onnx)
class_names_custom = ['door', 'trash_can']

# --- 5. HỆ THỐNG ÂM THANH VÀ TÍNH KHOẢNG CÁCH ---

# Cấu hình kích thước cho tính khoảng cách - TẤT CẢ OBJECT COCO + CUSTOM
class_avg_sizes = {
    # CUSTOM OBJECTS - được thêm vào
    "door": {"width_ratio": 0.8},      # Cửa
    "trash_can": {"width_ratio": 2.5}, # Thùng rác
    
    # Con người và động vật
    "person": {"width_ratio": 2.5},
    "bird": {"width_ratio": 8.0},
    "cat": {"width_ratio": 1.9},
    "dog": {"width_ratio": 1.5},
    "horse": {"width_ratio": 0.8},
    "sheep": {"width_ratio": 1.2},
    "cow": {"width_ratio": 0.6},
    "elephant": {"width_ratio": 0.3},
    "bear": {"width_ratio": 0.9},
    "zebra": {"width_ratio": 0.8},
    "giraffe": {"width_ratio": 0.4},
    
    # Phương tiện
    "bicycle": {"width_ratio": 2.3},
    "car": {"width_ratio": 0.37},
    "motorcycle": {"width_ratio": 2.4},
    "airplane": {"width_ratio": 0.1},
    "bus": {"width_ratio": 0.3},
    "train": {"width_ratio": 0.2},
    "truck": {"width_ratio": 0.25},
    "boat": {"width_ratio": 0.5},
    
    # Giao thông
    "traffic light": {"width_ratio": 2.95},
    "fire hydrant": {"width_ratio": 3.0},
    "stop sign": {"width_ratio": 2.55},
    "parking meter": {"width_ratio": 4.0},
    
    # Đồ vật lớn
    "bench": {"width_ratio": 1.6},
    "chair": {"width_ratio": 2.2},
    "couch": {"width_ratio": 1.0},
    "bed": {"width_ratio": 0.8},
    "dining table": {"width_ratio": 1.2},
    "toilet": {"width_ratio": 2.8},
    "tv": {"width_ratio": 1.8},
    
    # Đồ vật vừa
    "backpack": {"width_ratio": 3.5},
    "umbrella": {"width_ratio": 2.8},
    "handbag": {"width_ratio": 4.0},
    "tie": {"width_ratio": 8.0},
    "suitcase": {"width_ratio": 2.0},
    "laptop": {"width_ratio": 3.2},
    "book": {"width_ratio": 6.0},
    "clock": {"width_ratio": 3.0},
    "vase": {"width_ratio": 3.5},
    
    # Đồ thể thao
    "frisbee": {"width_ratio": 4.5},
    "skis": {"width_ratio": 6.0},
    "snowboard": {"width_ratio": 3.5},
    "sports ball": {"width_ratio": 5.5},
    "kite": {"width_ratio": 2.5},
    "baseball bat": {"width_ratio": 4.8},
    "baseball glove": {"width_ratio": 4.2},
    "skateboard": {"width_ratio": 3.8},
    "surfboard": {"width_ratio": 2.8},
    "tennis racket": {"width_ratio": 3.2},
    
    # Đồ ăn uống (đồ vật nhỏ)
    "bottle": {"width_ratio": 8.0},
    "wine glass": {"width_ratio": 8.5},
    "cup": {"width_ratio": 9.0},
    "fork": {"width_ratio": 12.0},
    "knife": {"width_ratio": 10.0},
    "spoon": {"width_ratio": 11.0},
    "bowl": {"width_ratio": 6.0},
    "banana": {"width_ratio": 10.0},
    "apple": {"width_ratio": 8.5},
    "sandwich": {"width_ratio": 5.0},
    "orange": {"width_ratio": 9.0},
    "broccoli": {"width_ratio": 7.0},
    "carrot": {"width_ratio": 12.0},
    "hot dog": {"width_ratio": 8.0},
    "pizza": {"width_ratio": 4.0},
    "donut": {"width_ratio": 9.0},
    "cake": {"width_ratio": 4.5},
    
    # Đồ gia dụng
    "potted plant": {"width_ratio": 3.0},
    "mouse": {"width_ratio": 10.0},
    "remote": {"width_ratio": 7.0},
    "keyboard": {"width_ratio": 3.8},
    "cell phone": {"width_ratio": 8.0},
    "microwave": {"width_ratio": 1.8},
    "oven": {"width_ratio": 1.5},
    "toaster": {"width_ratio": 3.5},
    "sink": {"width_ratio": 1.4},
    "refrigerator": {"width_ratio": 0.9},
    "scissors": {"width_ratio": 8.0},
    "teddy bear": {"width_ratio": 3.2},
    "hair drier": {"width_ratio": 5.0},
    "toothbrush": {"width_ratio": 12.0},
}

def setup_audio_system():
    """Khởi tạo hệ thống phát âm thanh với kiểm soát gián đoạn"""
    def speak(q):
        global audio_is_speaking
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 100)  # Chậm hơn để nghe rõ
            engine.setProperty('volume', 1.0)

            while True:
                if not q.empty() and not audio_is_speaking:
                    audio_is_speaking = True
                    
                    # Lấy tất cả objects trong queue để tạo thông báo tổng hợp
                    objects_by_position = {"LEFT": [], "FORWARD": [], "RIGHT": []}
                    
                    # Thu thập tất cả objects
                    while not q.empty():
                        try:
                            label, distance, position = q.get_nowait()
                            objects_by_position[position].append(label)
                        except:
                            break
                    
                    # Tạo thông báo theo vị trí
                    message_parts = []
                    
                    for position in ["LEFT", "FORWARD", "RIGHT"]:
                        if objects_by_position[position]:
                            objects_list = ", ".join(objects_by_position[position])
                            message_parts.append(f"{objects_list} on {position}")
                    
                    # Phát thông báo tổng hợp
                    if message_parts:
                        full_message = ". ".join(message_parts)
                        print(f"🔊 Speaking: {full_message}")  # Debug
                        engine.say(full_message)
                        engine.runAndWait()
                        print(f"✅ Audio completed")  # Debug
                        time.sleep(1.0)  # Pause sau khi phát xong
                    
                    audio_is_speaking = False
                
                else:
                    time.sleep(0.1)  # Kiểm tra thường xuyên hơn
                    
        except Exception as e:
            print(f"❌ Audio error: {e}")
            audio_is_speaking = False
    
    audio_queue = Queue()
    audio_thread = Thread(target=speak, args=(audio_queue,))
    audio_thread.daemon = True
    audio_thread.start()
    
    return audio_queue

def calculate_distance(object_width, frame_width, label):
    """Tính khoảng cách dựa trên độ rộng vật thể trong frame - Hỗ trợ TẤT CẢ object"""
    # Áp dụng hệ số hiệu chỉnh nếu có trong danh sách
    if label in class_avg_sizes:
        object_width *= class_avg_sizes[label]["width_ratio"]
    else:
        # Hệ số mặc định cho object không có trong danh sách
        object_width *= 2.0  # Hệ số trung bình
    
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
    face_y2 = y1 + int(0.08 * h)
    if face_y2 > y1:
        face_region = frame[y1:face_y2, x1:x2]
        if face_region.size > 0:
            blurred_face = cv2.GaussianBlur(face_region, (15, 15), 0)
            frame[y1:face_y2, x1:x2] = blurred_face
    return frame

def validate_trash_can(ix1, iy1, ix2, iy2, orig_h, orig_w):
    """Xác thực trash_can để giảm false positive"""
    box_height = iy2 - iy1
    box_width = ix2 - ix1
    box_center_y = (iy1 + iy2) // 2
    box_area = box_width * box_height
    frame_area = orig_w * orig_h
    
    # 1. Thùng rác thường ở phần dưới của frame (không ở trên cao)
    if box_center_y < orig_h * 0.3:  # Loại bỏ vật thể ở 30% trên cùng
        print(f"❌ TRASH_CAN rejected: Too high (center_y={box_center_y} < {orig_h * 0.3:.0f})")
        return False
    
    # 2. Thùng rác có tỷ lệ height/width hợp lý 
    aspect_ratio = box_height / (box_width + 1e-6)
    if aspect_ratio < 0.5 or aspect_ratio > 3.0:
        print(f"❌ TRASH_CAN rejected: Bad aspect ratio ({aspect_ratio:.2f})")
        return False
    
    # 3. Kích thước không được quá lớn (loại bỏ detection toàn màn hình)
    area_ratio = box_area / frame_area
    if area_ratio > 0.7:  # Không được chiếm quá 70% màn hình
        print(f"❌ TRASH_CAN rejected: Too large ({area_ratio:.2f} of frame)")
        return False
    
    # 4. Kích thước tối thiểu
    min_area = frame_area * 0.001  # Ít nhất 0.1% diện tích frame
    if box_area < min_area:
        print(f"❌ TRASH_CAN rejected: Too small ({box_area} < {min_area:.0f})")
        return False
    
    print(f"✅ TRASH_CAN validated: area_ratio={area_ratio:.3f}, aspect_ratio={aspect_ratio:.2f}")
    return True

# Biến global để kiểm soát audio
audio_is_speaking = False

# --- 6. KHỞI TẠO ---
print("🔊 Đang khởi tạo hệ thống âm thanh...")
audio_queue = setup_audio_system()
print("✅ Hệ thống âm thanh đã sẵn sàng")

# Khởi tạo camera với ưu tiên DroidCam
def init_camera():
    """Khởi tạo camera với ưu tiên DroidCam, fallback về webcam"""
    if USE_CAM:
        # Danh sách IP thường dùng cho DroidCam (có thể điều chỉnh theo mạng của bạn)
        droidcam_urls = [
            "http://192.168.1.100:4747/video",  # IP mặc định DroidCam
            "http://192.168.43.1:4747/video",   # Hotspot Android
            "http://10.0.0.100:4747/video",     # Mạng khác
        ]
        
        print("📱 Đang tìm kiếm DroidCam...")
        
        # Thử kết nối với từng URL DroidCam
        for url in droidcam_urls:
            print(f"🔍 Thử kết nối: {url}")
            try:
                cap = cv2.VideoCapture(url)
                # Thử đọc một frame để kiểm tra kết nối
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"✅ DroidCam kết nối thành công: {url}")
                    return cap
                else:
                    cap.release()
            except Exception as e:
                print(f"❌ Không thể kết nối {url}: {e}")
        
        # Nếu không kết nối được DroidCam, fallback về webcam
        print("⚠️  Không tìm thấy DroidCam, chuyển sang webcam mặc định...")
        try:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            if ret and frame is not None:
                print("✅ Webcam mặc định kết nối thành công")
                return cap
            else:
                cap.release()
                raise Exception("Webcam không hoạt động")
        except Exception as e:
            print(f"❌ Lỗi kết nối webcam: {e}")
            raise e
    else:
        # Sử dụng video file
        print(f"📹 Đang mở video file: {VIDEO_PATH}")
        return cv2.VideoCapture(VIDEO_PATH)

cap = init_camera()
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Ngưỡng Y để xét Gần/Xa (cho vật thể không hỗ trợ tính khoảng cách)
danger_line = int(orig_h * 0.75)

# Biến lưu trữ để tránh spam âm thanh
nearest_object_info = {"label": None, "distance": float('inf'), "last_announcement": 0}
frame_count = 0

# Lưu kết quả custom model để tối ưu performance
last_custom_results = []

# --- 7. MAIN LOOP ---
while True:
    ret, frame = cap.read()
    if not ret: 
        break
    
    frame_count += 1
    start_time = time.time()

    # Tiền xử lý với Letterbox (giống model-tranied.py)
    r = min(640 / orig_w, 640 / orig_h)
    new_unpad = (int(round(orig_w * r)), int(round(orig_h * r)))
    img_resized = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    img_640 = np.full((640, 640, 3), 114, dtype=np.uint8)
    dw = (640 - new_unpad[0]) // 2
    dh = (640 - new_unpad[1]) // 2
    img_640[dh:dh + new_unpad[1], dw:dw + new_unpad[0]] = img_resized
    
    img_640 = cv2.LUT(img_640, lut_table)
    blob = img_640.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0)

    # DUAL MODEL INFERENCE
    # COCO Model - chạy mỗi frame
    results_coco = session_coco.run(None, {session_coco.get_inputs()[0].name: blob})[0][0]
    
    # Custom Model - chỉ chạy mỗi 2 frame để tối ưu performance
    if frame_count % 2 == 0:
        last_custom_results = session_custom.run(None, {session_custom.get_inputs()[0].name: blob})[0][0]

    # Hàm mapping tọa độ (từ letterbox về original)
    def scale_coords(x1, y1, x2, y2):
        rx1 = int((x1 - dw) / r)
        ry1 = int((y1 - dh) / r)
        rx2 = int((x2 - dw) / r)
        ry2 = int((y2 - dh) / r)
        return (max(0, rx1), max(0, ry1), min(orig_w, rx2), min(orig_h, ry2))

    # Xử lý kết quả DUAL MODEL với tính khoảng cách và âm thanh
    objects_under_2m = []  # Lưu tất cả vật thể dưới 2m
    current_time = time.time()
    
    # === XỬ LÝ COCO MODEL RESULTS ===
    for pred in results_coco:
        x1, y1, x2, y2, score, cls_id = pred
        if score > 0.35:
            ix1, iy1, ix2, iy2 = scale_coords(x1, y1, x2, y2)
            
            label = class_names_coco[int(cls_id)]
            
            # TẤT CẢ OBJECT ĐỀU ĐƯỢC TÍNH KHOẢNG CÁCH
            object_width = ix2 - ix1
            distance = calculate_distance(object_width, orig_w, label)
            
            # Thu thập vật thể dưới 2m cho audio
            if distance < 2.0:
                x_center = (ix1 + ix2) // 2
                position = get_position(orig_w, x_center)
                objects_under_2m.append((label, distance, position))
            
            # Làm mờ khuôn mặt nếu là người
            if label == "person":
                frame = blur_person_face(frame, ix1, iy1, ix2, iy2)
            
            # NGƯỠNG MÀU: >= 2m = XANH, < 2m = ĐỎ
            color = (0, 255, 0) if distance >= 2.0 else (0, 0, 255)
            
            # Hiển thị label với khoảng cách
            display_label = f"{label} - {distance:.1f}m"
            
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, 2)
            cv2.putText(frame, display_label, (ix1, iy1-10), 0, 0.5, color, 2)

    # === XỬ LÝ CUSTOM MODEL RESULTS ===
    for pred in last_custom_results:
        x1, y1, x2, y2, score, cls_id = pred
        label = class_names_custom[int(cls_id)]
        
        if score > (0.4 if label == 'door' else 0.45):  # Tăng từ 0.25 lên 0.45 cho trash_can
            ix1, iy1, ix2, iy2 = scale_coords(x1, y1, x2, y2)
            
            # Validation đặc biệt cho trash_can
            if label == 'trash_can' and not validate_trash_can(ix1, iy1, ix2, iy2, orig_h, orig_w):
                continue  # Bỏ qua detection này nếu không hợp lệ
            
            # Xác thực trash_can để giảm false positive
            if label == "trash_can" and not validate_trash_can(ix1, iy1, ix2, iy2, orig_h, orig_w):
                continue
            
            # Debug info cho custom detections
            print(f"📍 {label.upper()}: score={score:.3f}, pos=({ix1},{iy1},{ix2},{iy2}), center_y={(iy1+iy2)//2}")
            
            # Tính khoảng cách cho custom objects
            object_width = ix2 - ix1
            distance = calculate_distance(object_width, orig_w, label)
            
            # Thu thập vật thể dưới 2m cho audio
            if distance < 2.0:
                x_center = (ix1 + ix2) // 2
                position = get_position(orig_w, x_center)
                objects_under_2m.append((f"TARGET: {label.upper()}", distance, position))
            
            # Màu đặc biệt cho custom objects: >= 2m = VÀNG, < 2m = ĐỎ
            color = (0, 255, 255) if distance >= 2.0 else (0, 0, 255)
            
            display_label = f"TARGET: {label.upper()} - {distance:.1f}m ({score:.2f})"
            
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, 2)
            cv2.putText(frame, display_label, (ix1, iy1-10), 0, 0.5, color, 2)

    # Xử lý âm thanh cho TẤT CẢ vật thể dưới 2m với kiểm soát gián đoạn
    if objects_under_2m and not audio_is_speaking and (current_time - nearest_object_info["last_announcement"] > 5.0):
        # Gửi tất cả objects dưới 2m vào queue
        for label, distance, position in objects_under_2m:
            audio_queue.put((label, distance, position))
        
        nearest_object_info["last_announcement"] = current_time
        print(f"🔊 Queuing {len(objects_under_2m)} objects under 2m to audio system")  # Debug

    # Hiển thị FPS và thông tin hệ thống
    fps = 1.0 / (time.time() - start_time)
    info_text = f"FPS: {fps:.1f} | DUAL MODEL: COCO + CUSTOM | Distance + Audio"
    cv2.putText(frame, info_text, (20, 40), 0, 0.7, (255, 255, 255), 2)
    
    # Hiển thị thống kê
    stats_text = f"Objects < 2m: {len(objects_under_2m)} | Audio: {'SPEAKING' if audio_is_speaking else 'READY'}"
    cv2.putText(frame, stats_text, (20, 70), 0, 0.5, (255, 255, 255), 2)
    
    cv2.imshow("DIPR Project - DUAL MODEL: Object Detection + Distance + Audio", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
