import cv2

# Mở webcam index 1
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Không mở được camera tại index 1!")
    exit()

# Mở camera để xem
while True:
    ret, frame = cap.read()
    if not ret:
        print("Không lấy được frame!")
        break

    cv2.imshow("DroidCam Preview", frame)

    if cv2.waitKey(1) == 27:  # ESC để thoát
        break

cap.release()
cv2.destroyAllWindows()