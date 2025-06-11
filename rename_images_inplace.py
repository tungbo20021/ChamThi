import os
import shutil
from PIL import Image

def rename_images_inplace():
    """
    Quét tất cả ảnh trong thư mục Data_raw và đổi tên thành image1.jpg, image2.jpg, v.v. tại chỗ
    """
    # Đường dẫn thư mục
    BASE_DIR = "D:/BKHN/20242/AI/ChamThi"
    DATA_RAW_DIR = os.path.join(BASE_DIR, "Data_raw")
    
    # Kiểm tra thư mục tồn tại
    if not os.path.exists(DATA_RAW_DIR):
        print(f"Thư mục {DATA_RAW_DIR} không tồn tại!")
        return
    
    # Lấy danh sách tất cả các file ảnh trong thư mục Data_raw
    image_files = []
    for root, _, files in os.walk(DATA_RAW_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_files.append(os.path.join(root, file))
    
    # Sắp xếp các file theo tên để đảm bảo thứ tự nhất quán
    image_files.sort()
    
    # Tạo thư mục tạm để lưu trữ ảnh trước khi đổi tên
    temp_dir = os.path.join(BASE_DIR, "Temp_Rename")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    
    # Sao chép tất cả ảnh vào thư mục tạm với tên mới
    total_images = len(image_files)
    new_paths = []
    
    for i, image_path in enumerate(image_files, 1):
        # Lấy phần mở rộng của file gốc
        _, ext = os.path.splitext(image_path)
        
        # Tạo tên file mới
        new_name = f"image{i}{ext.lower()}"
        temp_path = os.path.join(temp_dir, new_name)
        
        # Sao chép file vào thư mục tạm
        shutil.copy2(image_path, temp_path)
        new_paths.append((temp_path, os.path.join(DATA_RAW_DIR, new_name)))
        print(f"[{i}/{total_images}] Đã chuẩn bị đổi tên: {os.path.basename(image_path)} -> {new_name}")
    
    # Xóa các file gốc
    for image_path in image_files:
        os.remove(image_path)
        print(f"Đã xóa file gốc: {image_path}")
    
    # Di chuyển các file từ thư mục tạm về thư mục gốc
    for temp_path, final_path in new_paths:
        shutil.move(temp_path, final_path)
        print(f"Đã di chuyển: {os.path.basename(temp_path)} -> {final_path}")
    
    # Xóa thư mục tạm
    os.rmdir(temp_dir)
    
    print(f"\nĐã hoàn thành đổi tên {total_images} ảnh trong thư mục {DATA_RAW_DIR}!")

if __name__ == "__main__":
    rename_images_inplace()