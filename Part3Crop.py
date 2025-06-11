from ultralytics import YOLO
from PIL import Image
import os

# Đường dẫn đến ảnh và model
image_path = 'D:/BKHN/20242/AI/ChamThi/cropped_image_3.jpg'
model_path = 'D:/BKHN/20242/AI/ChamThi/runs/detect/train3/weights/best.pt'

# Kiểm tra sự tồn tại của file ảnh và model
if not os.path.exists(image_path):
    raise FileNotFoundError(f"File ảnh không tồn tại: {image_path}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"File model không tồn tại: {model_path}")

# Load model YOLO đã huấn luyện
model = YOLO(model_path)

# Dự đoán trên ảnh đầu vào
results = model(image_path)

# Hàm tạo tên file
def init_name(i):
    return f"crop_box_number_{i}.jpg"

# Lặp qua các kết quả
for result in results:
    # Lấy các hộp từ kết quả phát hiện
    boxes = result.boxes
    box_data = boxes.xywhn.cpu().numpy()  # Chuyển đổi tensor thành numpy array
    box_conf = boxes.conf.cpu().numpy()   # Độ tin cậy của các hộp

    # Kết hợp dữ liệu hộp và độ tin cậy
    combined_boxes = [(x, y, w, h, conf) for (x, y, w, h), conf in zip(box_data, box_conf)]

    # Sắp xếp các hộp dựa trên tọa độ x
    sorted_boxes = sorted(combined_boxes, key=lambda x: x[0])

    # Lọc các hộp có độ tin cậy dưới 0.95
    filtered_boxes = [box for box in sorted_boxes if box[4] >= 0.95]

    # Lấy kích thước của ảnh gốc từ thuộc tính "orig_shape"
    image_size = (result.orig_shape[1], result.orig_shape[0])
    output_dir = "D:/BKHN/20242/AI/ChamThi/output"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
        file_name = init_name(i)
        file_path = os.path.join(output_dir, file_name)

        # Lưu ảnh đã cắt
        cropped_image.save(file_path)

print("Hoàn thành cắt ảnh và lưu vào thư mục output.")
