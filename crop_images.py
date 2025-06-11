import os
from PIL import Image

def Crop1(urlimage, url):
    """
    Cắt ảnh thành 4 phần với kích thước cho Part1
    
    Args:
        urlimage: Đường dẫn đến ảnh cần cắt
        url: Thư mục đích để lưu ảnh đã cắt
    """
    img = Image.open(urlimage)
    crop_box_0 = (0, 0, 500, 600)
    crop_box_1 = (500, 0, 955, 600)
    crop_box_2 = (955, 0, 1405, 600)
    crop_box_3 = (1405, 0, 1920, 600)

    # Crop the image
    img_cropped_0 = img.crop(crop_box_0)
    img_cropped_1 = img.crop(crop_box_1)
    img_cropped_2 = img.crop(crop_box_2)
    img_cropped_3 = img.crop(crop_box_3)

    # Đảm bảo thư mục đích tồn tại
    if not os.path.exists(url):
        os.makedirs(url)

    # Lấy tên file gốc (không có phần mở rộng)
    base_name = os.path.splitext(os.path.basename(urlimage))[0]
    
    # Save the cropped image với tên file bao gồm tên file gốc
    img_cropped_0.save(os.path.join(url, f'{base_name}_cropped_0.jpg'))
    img_cropped_1.save(os.path.join(url, f'{base_name}_cropped_1.jpg'))
    img_cropped_2.save(os.path.join(url, f'{base_name}_cropped_2.jpg'))
    img_cropped_3.save(os.path.join(url, f'{base_name}_cropped_3.jpg'))
    
    return [
        os.path.join(url, f'{base_name}_cropped_0.jpg'),
        os.path.join(url, f'{base_name}_cropped_1.jpg'),
        os.path.join(url, f'{base_name}_cropped_2.jpg'),
        os.path.join(url, f'{base_name}_cropped_3.jpg')
    ]

def Crop2(urlimage, url):
    """
    Cắt ảnh thành 4 phần với kích thước cho Part2
    
    Args:
        urlimage: Đường dẫn đến ảnh cần cắt
        url: Thư mục đích để lưu ảnh đã cắt
    """
    img = Image.open(urlimage)
    # Kích thước khác cho Part2
    crop_box_0 = (0, 0, 500, 380)
    crop_box_1 = (500, 0, 955, 380)
    crop_box_2 = (955, 0, 1405, 380)
    crop_box_3 = (1405, 0, 1920, 380)

    # Crop the image
    img_cropped_0 = img.crop(crop_box_0)
    img_cropped_1 = img.crop(crop_box_1)
    img_cropped_2 = img.crop(crop_box_2)
    img_cropped_3 = img.crop(crop_box_3)

    # Đảm bảo thư mục đích tồn tại
    if not os.path.exists(url):
        os.makedirs(url)

    # Lấy tên file gốc (không có phần mở rộng)
    base_name = os.path.splitext(os.path.basename(urlimage))[0]
    
    # Save the cropped image với tên file bao gồm tên file gốc
    img_cropped_0.save(os.path.join(url, f'{base_name}_cropped_0.jpg'))
    img_cropped_1.save(os.path.join(url, f'{base_name}_cropped_1.jpg'))
    img_cropped_2.save(os.path.join(url, f'{base_name}_cropped_2.jpg'))
    img_cropped_3.save(os.path.join(url, f'{base_name}_cropped_3.jpg'))
    
    return [
        os.path.join(url, f'{base_name}_cropped_0.jpg'),
        os.path.join(url, f'{base_name}_cropped_1.jpg'),
        os.path.join(url, f'{base_name}_cropped_2.jpg'),
        os.path.join(url, f'{base_name}_cropped_3.jpg')
    ]

def process_folder(input_folder, output_folder, crop_function):
    """
    Xử lý tất cả các ảnh trong một thư mục
    
    Args:
        input_folder: Thư mục chứa ảnh cần cắt
        output_folder: Thư mục đích để lưu ảnh đã cắt
        crop_function: Hàm cắt ảnh sẽ được sử dụng
    """
    # Đảm bảo thư mục đích tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Đã tạo thư mục {output_folder}")
    
    # Lấy danh sách tất cả các file ảnh
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Sắp xếp các file theo tên
    image_files.sort()
    
    total_images = len(image_files)
    processed_images = 0
    
    # Xử lý từng ảnh
    for i, image_file in enumerate(image_files, 1):
        image_path = os.path.join(input_folder, image_file)
        
        try:
            # Cắt ảnh và lưu trực tiếp vào thư mục output
            cropped_images = crop_function(image_path, output_folder)
            processed_images += 1
            print(f"[{i}/{total_images}] Đã cắt thành công: {image_file}")
            print(f"  Đã lưu 4 ảnh con vào: {output_folder}")
        except Exception as e:
            print(f"[{i}/{total_images}] Lỗi khi xử lý ảnh {image_file}: {str(e)}")
    
    print(f"\nĐã hoàn thành! Đã xử lý {processed_images}/{total_images} ảnh.")
    return processed_images

def main():
    # Thư mục chứa ảnh gốc
    data_cropped_dir = "Data_cropped"
    part1_dir = os.path.join(data_cropped_dir, "Part1")
    part2_dir = os.path.join(data_cropped_dir, "Part2")
    
    # Thư mục đích để lưu ảnh đã cắt
    output_base_dir = "Data_subcropped"
    part1_output_dir = os.path.join(output_base_dir, "Part1")
    part2_output_dir = os.path.join(output_base_dir, "Part2")
    
    # Kiểm tra xem thư mục nguồn có tồn tại không
    if not os.path.exists(part1_dir):
        print(f"Thư mục {part1_dir} không tồn tại!")
        return
    
    if not os.path.exists(part2_dir):
        print(f"Thư mục {part2_dir} không tồn tại!")
        return
    
    # Xử lý Part1 với hàm Crop1
    print("\n=== Đang xử lý ảnh trong thư mục Part1 ===")
    part1_processed = process_folder(part1_dir, part1_output_dir, Crop1)
    
    # Xử lý Part2 với hàm Crop2
    print("\n=== Đang xử lý ảnh trong thư mục Part2 ===")
    part2_processed = process_folder(part2_dir, part2_output_dir, Crop2)
    
    # Hiển thị thống kê
    print("\n=== Thống kê ===")
    print(f"Part1: Đã xử lý {part1_processed} ảnh")
    print(f"Part2: Đã xử lý {part2_processed} ảnh")
    print(f"Tổng cộng: {part1_processed + part2_processed} ảnh")

if __name__ == "__main__":
    main()

