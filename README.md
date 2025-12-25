# Object Recognition with MiDaS Depth Estimation - Dự Án Hỗ Trợ Người Khiếm Thị

## Mô tả
Dự án này sử dụng YOLOv10 kết hợp với MiDaS (Monocular Depth Estimation) để nhận diện đối tượng và đo khoảng cách chính xác trong thời gian thực, hỗ trợ người khiếm thị điều hướng an toàn.

## Tính năng mới 🆕
- **MiDaS Depth Estimation**: Đo khoảng cách chính xác bằng deep learning
- **5 mức độ khoảng cách**: Rất gần, Gần, Trung bình, Xa, Rất xa
- **Visualization**: Hiển thị depth map real-time
- **Confidence scoring**: Đánh giá độ tin cậy của depth prediction

## Tính năng cơ bản
- Nhận diện 80 đối tượng COCO (người, xe, động vật, đồ vật...)
- Cảnh báo khoảng cách thông minh với độ chính xác cao
- Hỗ trợ camera real-time và video file
- Tối ưu hiệu suất với gamma correction và LUT

## Yêu cầu hệ thống
- Python 3.11+
- macOS (đã tối ưu)
- Camera (tùy chọn)

## Cài đặt

### Bước 1: Clone dự án và tạo virtual environment
```bash
# Đã có sẵn .venv, kích hoạt:
source .venv/bin/activate
```

### Bước 2: Cài đặt dependencies

**Cài đặt cơ bản (chỉ YOLO):**
```bash
./setup.sh
```

**Cài đặt đầy đủ (YOLO + MiDaS Depth):**
```bash
chmod +x setup_depth.sh
./setup_depth.sh
```

**Cài đặt thủ công:**
```bash
# Dependencies cơ bản
pip install numpy==1.24.3
pip install opencv-python==4.8.1.78
pip install onnxruntime>=1.18.0

# Cho depth estimation
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pillow>=9.5.0
pip install timm>=0.6.0
```

## Cách sử dụng

### Test Camera
```bash
python testcam.py
```

### Chạy dự án cơ bản (YOLO only)
```bash
python model.py
```

### Chạy dự án với Depth Estimation 🆕
```bash
python model_with_depth.py
```

### Test riêng Depth Estimation
```bash
python depth_estimation.py
```

### Cấu hình
Trong `model.py`, bạn có thể điều chỉnh:
- `USE_CAM = True`: Sử dụng camera
- `USE_CAM = False`: Sử dụng video file
- `VIDEO_PATH`: Đường dẫn đến file video

## Điều khiển
### model.py (cơ bản)
- **q**: Thoát chương trình

### model_with_depth.py (nâng cao) 🆕
- **q**: Thoát chương trình  
- **d**: Bật/tắt hiển thị depth map

### testcam.py
- **ESC**: Thoát
- **SPACE**: Chuyển camera tiếp theo

## Màu sắc cảnh báo

### Phiên bản cơ bản (model.py)
- **Xanh lá**: Đối tượng ở xa
- **Đỏ**: Đối tượng gần (NEAR)

### Phiên bản với Depth (model_with_depth.py) 🆕
- **Xanh dương**: Rất xa (>10m)
- **Xanh lá**: Xa (5-10m)
- **Vàng**: Trung bình (2-5m) 
- **Cam**: Gần (1-2m)
- **Đỏ**: Rất gần (<1m)
- **Xám**: Không xác định

## Cấu trúc dự án
```
├── model.py              # File chính (cơ bản)
├── model_with_depth.py   # File chính với depth estimation 🆕
├── depth_estimation.py   # Module MiDaS depth estimation 🆕
├── testcam.py            # Test camera
├── yolov10s.onnx        # Model YOLO AI
├── requirements.txt      # Dependencies  
├── setup.sh             # Script cài đặt cơ bản
├── setup_depth.sh       # Script cài đặt đầy đủ 🆕
└── README.md            # Hướng dẫn này
```

## Troubleshooting

### Lỗi camera không mở được
```bash
# Thử các index camera khác nhau:
# Sửa trong model.py: cv2.VideoCapture(1) hoặc cv2.VideoCapture(2)
```

### Lỗi ONNX Runtime
```bash
# Cài lại onnxruntime:
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime==1.16.3
```

### Lỗi OpenCV
```bash
# Cài lại OpenCV:
pip uninstall opencv-python
pip install opencv-python==4.8.1.78
```

## Thông tin thêm
- Model: YOLOv10s
- Framework: ONNX Runtime
- Platform: Optimized for macOS
- Author: Digital Image Processing Project
