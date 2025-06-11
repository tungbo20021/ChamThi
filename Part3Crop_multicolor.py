from ultralytics import YOLO
from PIL import Image, ImageDraw
import os
import numpy as np
import random
import cv2

# Đường dẫn đến ảnh và model
image_path = 'D:/BKHN/20242/AI/ChamThi/Part3_data/part3_image_9.jpg'
model_path = 'D:/BKHN/20242/AI/ChamThi/runs/detect/train3v2/weights/best.pt'

# Load model YOLO đã huấn luyện
model = YOLO(model_path)

# Dự đoán trên ảnh đầu vào
results = model(image_path)

# Phương pháp 1: Sử dụng OpenCV
def visualize_with_opencv():
    # Đọc ảnh gốc
    original_image = cv2.imread(image_path)
    
    # Tạo danh sách màu ngẫu nhiên (định dạng BGR cho OpenCV)
    colors = []
    for _ in range(100):  # Tạo 100 màu khác nhau
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        colors.append(color)
    
    # Lặp qua các kết quả
    for result in results:
        # Lấy các hộp từ kết quả phát hiện
        boxes = result.boxes.xyxy.cpu().numpy()  # Tọa độ tuyệt đối (x1, y1, x2, y2)
        box_conf = result.boxes.conf.cpu().numpy()   # Độ tin cậy của các hộp
        
        # Lặp qua các hộp và vẽ lên ảnh
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            conf = box_conf[i]
            if conf >= 0.85:  # Chỉ vẽ các box có độ tin cậy cao
                # Chọn màu từ danh sách
                color = colors[i % len(colors)]
                
                # Vẽ hộp
                cv2.rectangle(original_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                
                # Hiển thị độ tin cậy
                text = f"Conf: {conf:.2f}"
                cv2.putText(original_image, text, (int(x1), int(y1)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Lưu ảnh với các region được đánh dấu
    output_path = "D:/BKHN/20242/AI/ChamThi/output/multicolor_detected_regions_cv2.jpg"
    cv2.imwrite(output_path, original_image)
    
    # Hiển thị ảnh
    cv2.imshow("Multicolor Detected Regions", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Phương pháp 2: Sử dụng PIL
def visualize_with_pil():
    # Mở ảnh gốc để vẽ lên
    original_image = Image.open(image_path)
    draw = ImageDraw.Draw(original_image)
    
    # Tạo danh sách màu ngẫu nhiên (định dạng RGB cho PIL)
    colors = []
    for _ in range(100):  # Tạo 100 màu khác nhau
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        colors.append(color)
    
    # Lặp qua các kết quả
    for result in results:
        # Lấy các hộp từ kết quả phát hiện
        boxes = result.boxes.xyxy.cpu().numpy()  # Tọa độ tuyệt đối (x1, y1, x2, y2)
        box_conf = result.boxes.conf.cpu().numpy()   # Độ tin cậy của các hộp
        
        # Lặp qua các hộp và vẽ lên ảnh
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            conf = box_conf[i]
            if conf >= 0.85:  # Chỉ vẽ các box có độ tin cậy cao
                # Chọn màu từ danh sách
                color = colors[i % len(colors)]
                
                # Vẽ hộp
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Hiển thị độ tin cậy
                draw.text((x1, y1-15), f"Conf: {conf:.2f}", fill=color)
    
    # Lưu ảnh với các region được đánh dấu
    output_path = "D:/BKHN/20242/AI/ChamThi/output/multicolor_detected_regions_pil.jpg"
    original_image.save(output_path)
    
    # Hiển thị ảnh
    original_image.show()

# Chạy cả hai phương pháp
visualize_with_opencv()
visualize_with_pil()