from ultralytics import YOLO
from PIL import Image
import os
import cv2
import numpy as np

# Đường dẫn đến thư mục chứa ảnh và model
data_dir = 'D:/BKHN/20242/AI/ChamThi/Part3_data'
model_path = 'D:/BKHN/20242/AI/ChamThi/runs/detect/train3v2/weights/best.pt'
output_dir = "D:/BKHN/20242/AI/ChamThi/part3_cropped"

# Kiểm tra sự tồn tại của thư mục và model
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Thư mục ảnh không tồn tại: {data_dir}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"File model không tồn tại: {model_path}")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load model YOLO đã huấn luyện
model = YOLO(model_path)

# Hàm tạo tên file
def init_name(image_name, box_index):
    return f"{os.path.splitext(image_name)[0]}_box_{box_index}.jpg"

# Lấy danh sách tất cả các file ảnh trong thư mục
image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Xử lý từng ảnh
for image_file in image_files:
    image_path = os.path.join(data_dir, image_file)
    print(f"Đang xử lý ảnh: {image_file}")
    
    # Dự đoán trên ảnh đầu vào
    results = model(image_path)
    
    # Tạo thư mục con cho ảnh này (nếu cần)
    image_output_dir = os.path.join(output_dir, os.path.splitext(image_file)[0])
    if not os.path.exists(image_output_dir):
        os.makedirs(image_output_dir)
    
    # Lưu ảnh với các region được đánh dấu
    for r in results:
        im_array = r.plot()
        output_path = os.path.join(image_output_dir, f"detected_{image_file}")
        cv2.imwrite(output_path, im_array)
    
    # Xử lý các kết quả và cắt ảnh
    for result in results:
        # Lấy các hộp từ kết quả phát hiện
        boxes = result.boxes
        box_data = boxes.xywhn.cpu().numpy()
        box_conf = boxes.conf.cpu().numpy()
        
        # Kết hợp dữ liệu hộp và độ tin cậy
        combined_boxes = [(x, y, w, h, conf) for (x, y, w, h), conf in zip(box_data, box_conf)]
        
        # Sắp xếp các hộp dựa trên tọa độ x
        sorted_boxes = sorted(combined_boxes, key=lambda x: x[0])
        
        # Lọc các hộp có độ tin cậy dưới 0.95
        filtered_boxes = [box for box in sorted_boxes if box[4] >= 0.95]
        filtered_boxes = filtered_boxes[:6]  # Giới hạn 6 hộp
        
        # Lấy kích thước của ảnh gốc
        image_size = (result.orig_shape[1], result.orig_shape[0])
        
        # Lặp qua các hộp đã được sắp xếp và cắt ảnh
        for i, box in enumerate(filtered_boxes):
            x, y, w, h, conf = box
            x, y, w, h = map(float, (x, y, w, h))
            
            # Chuyển đổi tọa độ thành số nguyên và đảm bảo hộp nằm trong biên ảnh
            x1 = max(int((x - w / 2) * image_size[0]), 0)
            y1 = max(int((y - h / 2) * image_size[1]), 0)
            x2 = min(int((x + w / 2) * image_size[0]), image_size[0])
            y2 = min(int((y + h / 2) * image_size[1]), image_size[1])
            
            # Cắt ảnh
            image = Image.open(image_path)
            cropped_image = image.crop((x1, y1, x2, y2))
            
            # Tạo tên file hợp lệ
            file_name = init_name(image_file, i)
            file_path = os.path.join(image_output_dir, file_name)
            
            # Lưu ảnh đã cắt
            cropped_image.save(file_path)
            print(f"  Đã lưu: {file_name}")

print("Hoàn thành cắt ảnh và lưu vào thư mục output.")
