import cv2
import numpy as np
import onnxruntime as ort
import time
import os
import subprocess
from threading import Thread
from queue import Queue

##########################################################################################
# 1. CONFIG + INIT
USE_CAM = False
VIDEO_PATH = "D:\\DIPR project\\1000015019.mp4"

cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
cudnn_bin = r"C:\Program Files\NVIDIA\CUDNN\v9.17\bin"
if os.path.exists(cuda_bin): os.add_dll_directory(cuda_bin)
if os.path.exists(cudnn_bin): os.add_dll_directory(cudnn_bin)

##########################################################################################
# 2. IMAGE PROCESSING
gamma = 0.8
invGamma = 1.0 / gamma
LUT = np.array([((i / 255.0) ** invGamma) * 255
                for i in range(256)]).astype("uint8")

def preprocess_frame(frame, orig_w, orig_h):
    """Resize + Letterbox + Gamma correction"""
    r = min(640 / orig_w, 640 / orig_h)
    new_size = (int(orig_w * r), int(orig_h * r))

    resized = cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)

    dw = (640 - new_size[0]) // 2
    dh = (640 - new_size[1]) // 2
    canvas[dh:dh + new_size[1], dw:dw + new_size[0]] = resized

    canvas = cv2.LUT(canvas, LUT)

    blob = canvas.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0)

    return blob, r, dw, dh

def blur_person_face(frame, x1, y1, x2, y2):
    """Gaussian Blur cho vùng mặt"""
    h = y2 - y1
    face_h = int(0.08 * h)
    face = frame[y1:y1 + face_h, x1:x2]
    if face.size > 0:
        frame[y1:y1 + face_h, x1:x2] = cv2.GaussianBlur(face, (15, 15), 0)
    return frame

##########################################################################################
# 3. MODEL INFERENCE
providers = [('CUDAExecutionProvider', {}), 'CPUExecutionProvider']

print("\n=== LOADING MODELS ===")
try:
    session_coco = ort.InferenceSession("yolov10s.onnx", providers=providers)
    print("COCO Model (yolov10s.onnx) loaded successfully")
except Exception as e:
    print(f"Error loading COCO model: {e}"); exit()

try:
    session_custom = ort.InferenceSession("best.onnx", providers=providers)
    print("Custom Model (best.onnx) loaded successfully")
except Exception as e:
    print(f"Error loading Custom model: {e}"); exit()

print("🚀 DUAL MODEL SYSTEM READY!\n")

class_names_coco = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
class_names_custom = ['door', 'trash_can']

##########################################################################################
# 4. LOGIC – DISTANCE / POSITION / VALIDATION
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


def scale_coords(x1, y1, x2, y2, r, dw, dh, w, h):
    return (
        max(0, int((x1 - dw) / r)),
        max(0, int((y1 - dh) / r)),
        min(w, int((x2 - dw) / r)),
        min(h, int((y2 - dh) / r))
    )

def calculate_distance(obj_w, frame_w, label):
    """Tính khoảng cách dựa trên độ rộng vật thể - Hỗ trợ TẤT CẢ objects"""
    if label in class_avg_sizes:
        ratio = class_avg_sizes[label].get("width_ratio", 2.0)
    else:
        ratio = 2.0
    obj_w *= ratio
    distance = (frame_w * 0.5) / np.tan(np.radians(70 / 2)) / (obj_w + 1e-6)
    return round(distance, 2)

def get_position(frame_w, cx):
    if cx < frame_w / 3: return "LEFT"
    if cx < frame_w * 2 / 3: return "FORWARD"
    return "RIGHT"

def validate_trash_can(ix1, iy1, ix2, iy2, orig_h, orig_w):
    """Xác thực trash_can để giảm false positive"""
    box_height = iy2 - iy1
    box_width = ix2 - ix1
    box_center_y = (iy1 + iy2) // 2
    box_area = box_width * box_height
    frame_area = orig_w * orig_h
    
    if box_center_y < orig_h * 0.3:
        return False
    aspect_ratio = box_height / (box_width + 1e-6)
    if aspect_ratio < 0.5 or aspect_ratio > 3.0:
        return False
    area_ratio = box_area / frame_area
    if area_ratio > 0.7:
        return False
    min_area = frame_area * 0.001
    if box_area < min_area:
        return False
    return True

##########################################################################################
# 5. AUDIO WARNING SYSTEM
audio_is_speaking = False

def setup_audio_system():
    """Khởi tạo hệ thống phát âm thanh với kiểm soát gián đoạn"""
    def speak(q):
        global audio_is_speaking
        try:
            last_announcement_time = 0
            announcement_cooldown = 5.0  # Tăng từ 2.5s → 5s để phát từ từ hơn

            def say_in_background(message):
                """Phát audio qua PowerShell TTS - Windows built-in"""
                try:
                    # Escape quotes trong message
                    message = message.replace('"', '\"')
                    # Dùng PowerShell Add-Type for TTS
                    ps_cmd = f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak("{message}")'
                    subprocess.Popen(['powershell', '-Command', ps_cmd], 
                                   stdout=subprocess.DEVNULL, 
                                   stderr=subprocess.DEVNULL)
                    
                    # Tính thời gian phát dự kiến
                    estimated_time = len(message) * 0.05  # 50ms per character
                    time.sleep(max(1.5, estimated_time))
                except Exception as e:
                    print(f"TTS error: {e}")

            while True:
                current_time = time.time()
                
                # Kiểm tra có object trong queue và đủ thời gian từ thông báo cuối cùng
                if not q.empty() and (current_time - last_announcement_time) > announcement_cooldown:
                    audio_is_speaking = True
                    
                    # Lấy tất cả objects trong queue để tạo thông báo tổng hợp
                    objects_by_position = {"LEFT": [], "FORWARD": [], "RIGHT": []}
                    seen_objects = set()
                    
                    # Thu thập tất cả objects (loại bỏ duplicates)
                    while not q.empty():
                        try:
                            label, distance, position = q.get_nowait()
                            obj_key = f"{label}_{position}"
                            if obj_key not in seen_objects:
                                objects_by_position[position].append(label)
                                seen_objects.add(obj_key)
                        except:
                            break
                    
                    # Tạo thông báo theo vị trí - giới hạn max 2 objects per position
                    message_parts = []
                    
                    for position in ["LEFT", "FORWARD", "RIGHT"]:
                        if objects_by_position[position]:
                            # Chỉ lấy 1 object gần nhất per position (giảm số lượng phát)
                            top_objects = objects_by_position[position][:1]
                            objects_list = ", ".join(top_objects)
                            message_parts.append(f"{objects_list} on {position}")
                    
                    # Phát thông báo tổng hợp trong background thread
                    if message_parts:
                        full_message = ". ".join(message_parts)
                        print(f"Speaking: {full_message}")
                        
                        # Phát audio trong thread riêng để không block
                        audio_thread = Thread(target=say_in_background, args=(full_message,))
                        audio_thread.daemon = True
                        audio_thread.start()
                        
                        # Clear queue sau khi phát để không phát lại objects cũ
                        try:
                            audio_queue.queue.clear()
                        except:
                            pass
                        
                        print(f"Audio started")
                        last_announcement_time = time.time()
                    
                    audio_is_speaking = False
                
                else:
                    time.sleep(0.05)
                    
        except Exception as e:
            print(f"Audio error: {e}")
            audio_is_speaking = False
    
    audio_queue = Queue()
    audio_thread = Thread(target=speak, args=(audio_queue,))
    audio_thread.daemon = True
    audio_thread.start()
    
    return audio_queue

##########################################################################################
# 6. KHỞI TẠO CAMERA
def init_camera():
    """Khởi tạo camera - hỗ trợ webcam"""
    if USE_CAM:
        print("Trying to initialize camera...")
        try:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            if ret and frame is not None:
                print("Camera initialized successfully")
                return cap
            else:
                cap.release()
                raise Exception("Camera not responding")
        except Exception as e:
            print(f"Camera error: {e}")
            raise e
    else:
        print(f"Loading video: {VIDEO_PATH}")
        return cv2.VideoCapture(VIDEO_PATH)

cap = init_camera()
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video size: {orig_w}x{orig_h}\n")

##########################################################################################
# 7. MAIN LOOP
nearest_object_info = {"label": None, "distance": float('inf'), "last_announcement": 0}
frame_count = 0
last_custom = []
last_custom_results = []

audio_queue = setup_audio_system()

danger_line = int(orig_h * 0.75)

while True:
    ret, frame = cap.read()
    if not ret: break
    start = time.time()
    frame_count += 1

    blob, r, dw, dh = preprocess_frame(frame, orig_w, orig_h)

    coco_out = session_coco.run(None, {session_coco.get_inputs()[0].name: blob})[0][0]
    if frame_count % 2 == 0:
        last_custom = session_custom.run(None, {session_custom.get_inputs()[0].name: blob})[0][0]

    danger_objects = []

    for x1, y1, x2, y2, conf, cls in coco_out:
        if conf < 0.35: continue
        label = class_names_coco[int(cls)]
        ix1, iy1, ix2, iy2 = scale_coords(x1, y1, x2, y2, r, dw, dh, orig_w, orig_h)

        dist = calculate_distance(ix2 - ix1, orig_w, label)
        cx = (ix1 + ix2) // 2
        pos = get_position(orig_w, cx)

        if dist < 2:
            danger_objects.append((label, dist, pos))

        if label == "person":
            frame = blur_person_face(frame, ix1, iy1, ix2, iy2)

        color = (0, 0, 255) if dist < 2 else (0, 255, 0)
        cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f} {dist:.1f}m", (ix1, iy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for x1, y1, x2, y2, conf, cls in last_custom:
        if conf < 0.35: continue
        label = class_names_custom[int(cls)]
        ix1, iy1, ix2, iy2 = scale_coords(x1, y1, x2, y2, r, dw, dh, orig_w, orig_h)

        dist = calculate_distance(ix2 - ix1, orig_w, label)
        cx = (ix1 + ix2) // 2
        pos = get_position(orig_w, cx)

        if dist < 2:
            danger_objects.append((label, dist, pos))

        if label == "trash_can":
            if not validate_trash_can(ix1, iy1, ix2, iy2, orig_h, orig_w):
                continue
            if conf < 0.4:
                continue

        color = (0, 0, 255) if dist < 2 else (0, 255, 0)
        cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f} {dist:.1f}m", (ix1, iy1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Chỉ lấy top 5 objects gần nhất để gửi cho audio (loại bỏ quá nhiều)
    if danger_objects:
        # Sort by distance (gần nhất trước)
        sorted_objects = sorted(danger_objects, key=lambda x: x[1])[:5]
        for label, dist, pos in sorted_objects:
            audio_queue.put((label, dist, pos))
    else:
        # Nếu không có object < 2m, xóa queue để tránh lặp lại
        try:
            audio_queue.queue.clear()
        except:
            pass

    fps = 1 / (time.time() - start)
    info_text = f"FPS: {fps:.1f} | DUAL MODEL: COCO + CUSTOM"
    cv2.putText(frame, info_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    stats_text = f"Objects < 2m: {len(danger_objects)} | Audio: {'SPEAKING' if audio_is_speaking else 'READY'}"
    cv2.putText(frame, stats_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("DIPR – Dual Model: Detection + Distance + Audio", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
