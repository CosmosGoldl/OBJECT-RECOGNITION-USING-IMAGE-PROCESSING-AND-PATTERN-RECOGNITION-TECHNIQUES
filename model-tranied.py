import cv2
import numpy as np
import onnxruntime as ort
import time
import os

# --- 1. CẤU HÌNH ---
USE_CAM = True  # Changed to True for macOS camera usage
VIDEO_PATH = "sample_video.mp4"  # Placeholder for video file

# Note: DLL paths are for Windows only, not needed on macOS

# --- 2. KHỞI TẠO BẢNG LUT ---
gamma = 0.8
invGamma = 1.0 / gamma
lut_table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

# --- 3. KHỞI TẠO MODEL ---
providers = ['CPUExecutionProvider']  # Use CPU only on macOS
try:
    opt = ort.SessionOptions()
    session_coco = ort.InferenceSession("yolov10s.onnx", providers=providers, sess_options=opt)
    session_custom = ort.InferenceSession("best.onnx", providers=providers, sess_options=opt)
    print("✅ Hệ hệ thống sẵn sàng")
except Exception as e:
    print(f"❌ Lỗi: {e}"); exit()

class_names_coco = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
class_names_custom = ['door', 'trash_can']

last_custom_results = [] 
frame_count = 0

cap = cv2.VideoCapture(0 if USE_CAM else VIDEO_PATH)
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
danger_line = int(orig_h * 0.75)

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_count += 1
    start_time = time.time()

    # --- TIỀN XỬ LÝ LETTERBOX ---
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

    # --- INFERENCE ---
    results_coco = session_coco.run(None, {session_coco.get_inputs()[0].name: blob})[0][0]
    if frame_count % 2 == 0:
        last_custom_results = session_custom.run(None, {session_custom.get_inputs()[0].name: blob})[0][0]

    # --- MAPPING TỌA ĐỘ ---
    def scale_coords(x1, y1, x2, y2):
        rx1 = int((x1 - dw) / r)
        ry1 = int((y1 - dh) / r)
        rx2 = int((x2 - dw) / r)
        ry2 = int((y2 - dh) / r)
        return (max(0, rx1), max(0, ry1), min(orig_w, rx2), min(orig_h, ry2))

    # Vẽ COCO
    for pred in results_coco:
        x1, y1, x2, y2, score, cls_id = pred
        if score > 0.45:
            ix1, iy1, ix2, iy2 = scale_coords(x1, y1, x2, y2)
            color = (0, 0, 255) if iy2 > danger_line else (0, 255, 0)
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, 2)
            cv2.putText(frame, f"{class_names_coco[int(cls_id)]}", (ix1, iy1-10), 0, 0.5, color, 2)

   # Vẽ Custom với cơ chế "Siết khung" (Tighten Bounding Box)
    for pred in last_custom_results:
        x1, y1, x2, y2, score, cls_id = pred
        label = class_names_custom[int(cls_id)]
        
        if score > (0.4 if label == 'door' else 0.25):
            ix1, iy1, ix2, iy2 = scale_coords(x1, y1, x2, y2)
            
            # --- XỬ LÝ SIÊT KHUNG RIÊNG CHO TRASH_CAN ---
            if label == 'trash_can':
                w_box = ix2 - ix1
                h_box = iy2 - iy1
                
                # Tăng mạnh hệ số siết khung: 
                # Thu nhỏ 20% mỗi bên trái/phải và 15% phía trên/dưới
               # ix1 = int(ix1 + w_box * 0.30)
                #ix2 = int(ix2 - w_box * 0.30)
               # iy1 = int(iy1 + h_box * 0.10)
               # iy2 = int(iy2 - h_box * 0.40) # Giữ phần đáy thấp hơn để bám mặt đất
            
            # Kiểm tra khoảng cách sau khi đã siết khung
            color = (0, 0, 255) if iy2 > danger_line else (0, 255, 255)
            
            cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, 2)
            cv2.putText(frame, f"TARGET: {label.upper()} {score:.2f}", (ix1, iy1-10), 
                        0, 0.5, color, 2)

    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), 0, 0.7, (255, 255, 255), 2)
    cv2.imshow("DIPR Project - Blind Support", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()