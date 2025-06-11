import os
import shutil
from PIL import Image

# Thư mục nguồn chứa các thư mục image_1, image_2, ...
source_base_dir = 'Data_processed'

# Thư mục đích để lưu tất cả các ảnh cropped
target_dir = 'Data_cropped'

# Tạo thư mục đích nếu chưa tồn tại
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    print(f"Đã tạo thư mục {target_dir}")

# Tạo các thư mục con cho từng loại ảnh
part1_dir = os.path.join(target_dir, 'Part1')
part2_dir = os.path.join(target_dir, 'Part2')
part3_dir = os.path.join(target_dir, 'Part3')
todien_dir = os.path.join(target_dir, 'Todien_cropped')  # Thêm thư mục mới cho transformed.jpg

# Tạo các thư mục con nếu chưa tồn tại
for dir_path in [part1_dir, part2_dir, part3_dir, todien_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Đã tạo thư mục {dir_path}")

# Lấy danh sách tất cả các thư mục image_X
image_folders = [f for f in os.listdir(source_base_dir) if f.startswith('image_') and os.path.isdir(os.path.join(source_base_dir, f))]

# Sắp xếp các thư mục theo số thứ tự
image_folders.sort(key=lambda x: int(x.split('_')[1]))

# Khởi tạo biến đếm
count_part1 = 0
count_part2 = 0
count_part3 = 0
count_todien = 0
total_folders = len(image_folders)

# Duyệt qua từng thư mục
for i, folder in enumerate(image_folders, 1):
    source_folder = os.path.join(source_base_dir, folder)
    
    # Kiểm tra và sao chép cropped_image_1.jpg
    source_image_path1 = os.path.join(source_folder, 'cropped_image_1.jpg')
    if os.path.exists(source_image_path1):
        target_image_path1 = os.path.join(part1_dir, f'part1_image_{i}.jpg')
        shutil.copy2(source_image_path1, target_image_path1)
        count_part1 += 1
        print(f"Đã sao chép {source_image_path1} -> {target_image_path1}")
    else:
        print(f"Không tìm thấy file cropped_image_1.jpg trong thư mục {source_folder}")
    
    # Kiểm tra và sao chép cropped_image_2.jpg
    source_image_path2 = os.path.join(source_folder, 'cropped_image_2.jpg')
    if os.path.exists(source_image_path2):
        target_image_path2 = os.path.join(part2_dir, f'part2_image_{i}.jpg')
        shutil.copy2(source_image_path2, target_image_path2)
        count_part2 += 1
        print(f"Đã sao chép {source_image_path2} -> {target_image_path2}")
    else:
        print(f"Không tìm thấy file cropped_image_2.jpg trong thư mục {source_folder}")
    
    # Kiểm tra và sao chép cropped_image_3.jpg
    source_image_path3 = os.path.join(source_folder, 'cropped_image_3.jpg')
    if os.path.exists(source_image_path3):
        target_image_path3 = os.path.join(part3_dir, f'part3_image_{i}.jpg')
        shutil.copy2(source_image_path3, target_image_path3)
        count_part3 += 1
        print(f"Đã sao chép {source_image_path3} -> {target_image_path3}")
    else:
        print(f"Không tìm thấy file cropped_image_3.jpg trong thư mục {source_folder}")
    
    # Kiểm tra và sao chép transformed.jpg
    source_transformed_path = os.path.join(source_folder, 'transformed.jpg')
    if os.path.exists(source_transformed_path):
        target_transformed_path = os.path.join(todien_dir, f'todien_image_{i}.jpg')
        shutil.copy2(source_transformed_path, target_transformed_path)
        count_todien += 1
        print(f"Đã sao chép {source_transformed_path} -> {target_transformed_path}")
    else:
        print(f"Không tìm thấy file transformed.jpg trong thư mục {source_folder}")
    
    # Hiển thị tiến trình
    print(f"Đã xử lý: {i}/{total_folders} thư mục ({(i/total_folders*100):.1f}%)")

# Hiển thị thống kê
print("\nKết quả thu thập:")
print(f"- Part1: {count_part1}/{total_folders} ảnh")
print(f"- Part2: {count_part2}/{total_folders} ảnh")
print(f"- Part3: {count_part3}/{total_folders} ảnh")
print(f"- Todien: {count_todien}/{total_folders} ảnh")
print(f"Tổng số thư mục đã kiểm tra: {total_folders}")
