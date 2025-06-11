import os
import shutil
from PIL import Image

# Thư mục nguồn chứa các thư mục image_1, image_2, ...
source_base_dir = 'Data_processed'

# Thư mục đích để lưu tất cả các ảnh cropped_image_3.jpg
target_dir = 'Part3_data'

# Tạo thư mục đích nếu chưa tồn tại
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    print(f"Đã tạo thư mục {target_dir}")

# Lấy danh sách tất cả các thư mục image_X
image_folders = [f for f in os.listdir(source_base_dir) if f.startswith('image_') and os.path.isdir(os.path.join(source_base_dir, f))]

# Sắp xếp các thư mục theo số thứ tự
image_folders.sort(key=lambda x: int(x.split('_')[1]))

# Duyệt qua từng thư mục
for i, folder in enumerate(image_folders, 1):
    source_folder = os.path.join(source_base_dir, folder)
    source_image_path = os.path.join(source_folder, 'cropped_image_3.jpg')
    
    # Kiểm tra xem file cropped_image_3.jpg có tồn tại không
    if os.path.exists(source_image_path):
        # Tạo tên file mới trong thư mục đích
        target_image_path = os.path.join(target_dir, f'part3_image_{i}.jpg')
        
        # Sao chép file
        shutil.copy2(source_image_path, target_image_path)
        print(f"Đã sao chép {source_image_path} -> {target_image_path}")
    else:
        print(f"Không tìm thấy file cropped_image_3.jpg trong thư mục {source_folder}")

print(f"Đã hoàn thành! Tổng số {len(image_folders)} thư mục được kiểm tra.")