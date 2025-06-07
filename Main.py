import cv2
from ultralytics import YOLO 

# Đường dẫn tới model YOLO (best.pt) và link RTSP camera
MODEL_PATH = "best.pt"
RTSP_URL = "rtsp://<user>:<password>@<ip>:<port>/path"  # Ví dụ: rtsp://admin:12345@192.168.1.2:554/stream1

# Load YOLO model
model = YOLO(MODEL_PATH)

# Kết nối camera RTSP
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("❌ Không thể kết nối tới camera. Vui lòng kiểm tra lại đường dẫn RTSP!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không đọc được frame từ camera.")
        break

    # Chạy model detection
    results = model(frame)

    # Vẽ kết quả lên frame
    annotated_frame = results.render()[0]

    # Hiển thị
    cv2.imshow("YOLO Detection - RTSP Camera", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
