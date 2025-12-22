import cv2
import numpy as np
import onnxruntime as ort
import time
import os

# --- 1. CẤU HÌNH ---
USE_CAM = False 
VIDEO_PATH = "D:\\DIPR project\\1000015019.mp4"

# DLL Paths (Giữ nguyên cấu hình hệ thống của bạn)
cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
cudnn_bin = r"C:\Program Files\NVIDIA\CUDNN\v9.17\bin"
if os.path.exists(cuda_bin): os.add_dll_directory(cuda_bin)
if os.path.exists(cudnn_bin): os.add_dll_directory(cudnn_bin)

# --- 2. KHỞI TẠO BẢNG LUT (GAMMA CORRECTION) ---
# Gamma < 1.0 làm sáng vùng tối mà không cháy vùng sáng. 0.8 là mức tối ưu.
gamma = 0.8
invGamma = 1.0 / gamma
lut_table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

# --- 3. KHỞI TẠO MODEL ---
providers = [('CUDAExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
session_coco = None
session_custom = None 

try:
    opt = ort.SessionOptions()
    session_coco = ort.InferenceSession("yolov10s.onnx", providers=providers, sess_options=opt)
    # session_custom = ort.InferenceSession("best.onnx", providers=providers, sess_options=opt)
    print("✅ Hệ thống đã sẵn sàng với GPU")
except Exception as e:
    print(f"❌ Lỗi: {e}"); exit()

class_names_coco = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
class_names_custom = ['door', 'trash_can']

# --- BIẾN LƯU TRỮ ---
last_custom_results = [] 
frame_count = 0

cap = cv2.VideoCapture(0 if USE_CAM else VIDEO_PATH)
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Ngưỡng Y để xét Gần/Xa (Cạnh dưới khung hình > 75% chiều cao ảnh là Rất Gần)
danger_line = int(orig_h * 0.75)

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

    # --- BƯỚC 5: VẼ KẾT QUẢ ---
    # Vẽ COCO (Màu Xanh lá)
    for pred in results_coco:
        x1, y1, x2, y2, score, cls_id = pred
        if score > 0.35:
            ix1, iy1 = int(x1 * orig_w / 640), int(y1 * orig_h / 640)
            ix2, iy2 = int(x2 * orig_w / 640), int(y2 * orig_h / 640)
            
            # Xét khoảng cách dựa trên tọa độ chân (iy2)
            color = (0, 0, 255) if iy2 > danger_line else (0, 255, 0)
            label = f"{class_names_coco[int(cls_id)]} {'(NEAR)' if iy2 > danger_line else ''}"
            
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, 2)
            cv2.putText(frame, label, (ix1, iy1-10), 0, 0.5, color, 2)

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

    # Hiển thị FPS và thông tin
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.1f} | GPU Mode", (20, 40), 0, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("DIPR Project - Blind Support", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()