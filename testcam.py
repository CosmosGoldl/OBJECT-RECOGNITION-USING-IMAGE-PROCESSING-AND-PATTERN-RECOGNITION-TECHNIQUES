import cv2

# Thử các index camera khác nhau
for camera_index in [0, 1, 2]:
    cap = cv2.VideoCapture(camera_index)
    
    if cap.isOpened():
        print(f"✅ Camera tìm thấy tại index {camera_index}")
        
        # Test lấy frame
        ret, frame = cap.read()
        if ret:
            print(f"📹 Camera {camera_index} hoạt động bình thường")
            cv2.imshow(f"Camera Index {camera_index}", frame)
            print("Nhấn ESC để thoát, SPACE để thử camera tiếp theo...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                cv2.imshow(f"Camera Index {camera_index}", frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()
                elif key == 32:  # SPACE
                    break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print(f"❌ Không tìm thấy camera tại index {camera_index}")

print("🔍 Hoàn tất kiểm tra camera!")