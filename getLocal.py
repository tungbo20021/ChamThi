from ultralytics import YOLO
from PIL import Image, ImageDraw, ExifTags
import numpy as np
import cv2
import os
import shutil

# Load a pretrained YOLO model
model = YOLO('D:/BKHN/20242/AI/ChamThi/runs/detect/train/weights/best.pt')

# Thư mục chứa ảnh gốc
input_dir = 'Data_raw/Todien'
# Thư mục lưu ảnh đã xử lý
output_base_dir = 'Data_processed'
# Tạo thư mục output nếu chưa tồn tại
if not os.path.exists(output_base_dir):
    os.makedirs(output_base_dir)

def correct_orientation(image):
    """Hiệu chỉnh hướng ảnh dựa trên thông tin EXIF"""
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            break
    
    try:
        exif = image._getexif()
        if exif is not None:
            orientation_value = exif.get(orientation)
            if orientation_value == 3:
                image = image.rotate(180, expand=True)
            elif orientation_value == 6:
                image = image.rotate(270, expand=True)
            elif orientation_value == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # Một số ảnh có thể không có thông tin EXIF
        pass
    
    return image

def transform_image(image_path, model):
    """Phát hiện góc và transform ảnh"""
    # Đọc ảnh và hiệu chỉnh hướng
    image = Image.open(image_path)
    image = correct_orientation(image)
    image_width, image_height = image.size
    
    # Chạy model để phát hiện các góc
    result = model(image_path)
    
    # Khởi tạo biến để lưu các box tốt nhất cho mỗi góc
    corners = {
        "top_left": None,
        "top_right": None,
        "bottom_left": None,
        "bottom_right": None
    }
    corner_distances = {
        "top_left": float('inf'),
        "top_right": float('inf'),
        "bottom_left": float('inf'),
        "bottom_right": float('inf')
    }
    
    # Định nghĩa tọa độ các góc
    corner_coords = {
        "top_left": (0, 0),
        "top_right": (image_width, 0),
        "bottom_left": (0, image_height),
        "bottom_right": (image_width, image_height)
    }
    
    # Tìm các box tốt nhất cho mỗi góc
    for r in result:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu()
            confidence = box.conf[0].item()
            box_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            
            for corner, coord in corner_coords.items():
                distance = np.sqrt((box_center[0] - coord[0]) ** 2 + (box_center[1] - coord[1]) ** 2)
                if distance < corner_distances[corner] and confidence >= 0.8:
                    corners[corner] = box
                    corner_distances[corner] = distance
    
    # Tính toán điểm trung tâm của các box
    center_points = {}
    for corner, box in corners.items():
        if box is not None:
            x1, y1, x2, y2 = box.xyxy[0].cpu()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            center_points[corner] = (center_x, center_y)
    
    # Chuyển đổi ảnh PIL sang định dạng OpenCV
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Thực hiện biến đổi phối cảnh nếu tất cả các góc được phát hiện
    if len(center_points) == 4:
        src_points = np.array([
            center_points["top_left"],
            center_points["top_right"],
            center_points["bottom_right"],
            center_points["bottom_left"]
        ], dtype="float32")
        
        dst_points = np.array([
            [0, 0],
            [1920, 0],
            [1920, 2560],
            [0, 2560]
        ], dtype="float32")
        
        # Tính ma trận biến đổi phối cảnh
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Warp ảnh
        warped_image = cv2.warpPerspective(cv_image, matrix, (1920, 2560))
        
        # Chuyển đổi kết quả trở lại định dạng PIL
        warped_image = Image.fromarray(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
        
        return warped_image, True
    else:
        return image, False

def crop_image(image, output_folder):
    """Cắt ảnh thành 4 phần và lưu"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Định nghĩa các box cắt
    crop_box_0 = (1400, 0, 1920, 800)
    crop_box_1 = (0, 800, 1920, 1400)
    crop_box_2 = (0, 1375, 1920, 1750)
    crop_box_3 = (0, 1750, 1920, 2560)
    
    # Cắt ảnh
    img_cropped_0 = image.crop(crop_box_0)
    img_cropped_1 = image.crop(crop_box_1)
    img_cropped_2 = image.crop(crop_box_2)
    img_cropped_3 = image.crop(crop_box_3)
    
    # Lưu ảnh đã cắt
    img_cropped_0.save(os.path.join(output_folder, 'cropped_image_0.jpg'))
    img_cropped_1.save(os.path.join(output_folder, 'cropped_image_1.jpg'))
    img_cropped_2.save(os.path.join(output_folder, 'cropped_image_2.jpg'))
    img_cropped_3.save(os.path.join(output_folder, 'cropped_image_3.jpg'))

def process_all_images():
    """Xử lý tất cả ảnh trong thư mục input"""
    # Lấy danh sách tất cả các file ảnh
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    # Sắp xếp các file theo tên
    image_files.sort()
    
    # Xử lý từng ảnh
    for i, image_file in enumerate(image_files, 1):
        image_path = os.path.join(input_dir, image_file)
        print(f"Đang xử lý ảnh {i}/{len(image_files)}: {image_file}")
        
        # Tạo thư mục output cho ảnh này
        output_folder = os.path.join(output_base_dir, f"image_{i}")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Transform ảnh
        transformed_image, success = transform_image(image_path, model)
        
        # Lưu ảnh đã transform
        transformed_path = os.path.join(output_folder, "transformed.jpg")
        transformed_image.save(transformed_path)
        
        if success:
            # Cắt ảnh nếu transform thành công
            crop_image(transformed_image, output_folder)
            print(f"  ✓ Đã xử lý và cắt thành công")
        else:
            # Lưu thông tin nếu không transform được
            print(f"  ✗ Không phát hiện đủ 4 góc, không thể transform")
            with open(os.path.join(output_folder, "error.txt"), "w") as f:
                f.write("Không phát hiện đủ 4 góc, không thể transform")

# Chạy hàm xử lý tất cả ảnh
if __name__ == "__main__":
    process_all_images()

