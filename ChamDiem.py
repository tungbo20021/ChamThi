from ultralytics import YOLO
from PIL import Image, ImageDraw, ExifTags
import numpy as np
import cv2
import math
import os
import pandas as pd
from datetime import datetime

# Định nghĩa các biến toàn cục
key_map_21 = ["Đ", "S", "Đ", "S"]
key_map_22 = ["A", "B", "C", "D"]

# Đường dẫn thư mục
BASE_DIR = "D:/BKHN/20242/AI/ChamThi"
DATA_RAW_DIR = os.path.join(BASE_DIR, "Data_raw")
MODELS_DIR = os.path.join(BASE_DIR, "runs/detect")
OUTPUT_DIR = os.path.join(BASE_DIR, "Results")
TEMP_DIR = os.path.join(BASE_DIR, "Temp")

# Tạo các thư mục nếu chưa tồn tại
for dir_path in [OUTPUT_DIR, TEMP_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Load các model YOLO
detect_square_model = YOLO(os.path.join(MODELS_DIR, 'trainv2/weights/best.pt'))
detect_rect_model = YOLO(os.path.join(MODELS_DIR, 'train3v2/weights/best.pt'))
detect_circle_model = YOLO(os.path.join(MODELS_DIR, 'train4/weights/best.pt'))
detect_circle_model_update = YOLO(os.path.join(MODELS_DIR, 'train5/weights/best.pt'))
detect_circle_model_update1 = YOLO(os.path.join(MODELS_DIR, 'train6/weights/best.pt'))

# **************************************************************************************************
# ********************************* FUNC BONUS *****************************************************
# **************************************************************************************************

def init_name(i):
    return f"crop_box_number_{i}.jpg"

def convert_to_gray(image_path, output_path):
    image = Image.open(image_path)
    gray_image = image.convert('L')
    gray_image.save(output_path)

def Tranformer(image_path, model, savefolder):

    result = model(image_path)

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
    image = Image.open(image_path)
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            break
    exif = image._getexif()
    if exif is not None:
        orientation_value = exif.get(orientation)
        if orientation_value == 3:
            image = image.rotate(180, expand=True)
        elif orientation_value == 6:
            image = image.rotate(270, expand=True)
        elif orientation_value == 8:
            image = image.rotate(90, expand=True)

    image_width, image_height = image.size
    corner_coords = {
        "top_left": (0, 0),
        "top_right": (image_width, 0),
        "bottom_left": (0, image_height),
        "bottom_right": (image_width, image_height)
    }
    for r in result:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].item()
            box_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            for corner, coord in corner_coords.items():
                distance = np.sqrt((box_center[0] - coord[0]) ** 2 + (box_center[1] - coord[1]) ** 2)
                if distance < corner_distances[corner] and confidence >= 0.75:
                    corners[corner] = box
                    corner_distances[corner] = distance
    center_points = {}
    for corner, box in corners.items():
        if box is not None:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            center_points[corner] = (center_x, center_y)

    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

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

        matrix = cv2.getPerspectiveTransform(src_points, dst_points)

        warped_image = cv2.warpPerspective(cv_image, matrix, (1920, 2560))

        warped_image = Image.fromarray(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))

        warped_image.save(savefolder)
    else:
        print("Not all corners were detected.")

def Crop1(urlimage, url):
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

    # Save the cropped image
    img_cropped_0.save(url + '/cropped_image_0.jpg')
    img_cropped_1.save(url + '/cropped_image_1.jpg')
    img_cropped_2.save(url + '/cropped_image_2.jpg')
    img_cropped_3.save(url + '/cropped_image_3.jpg')

def Crop2(urlimage, url):
    img = Image.open(urlimage)
    crop_box_0 = (0, 0, 500, 380)
    crop_box_1 = (500, 0, 955, 380)
    crop_box_2 = (955, 0, 1405, 380)
    crop_box_3 = (1405, 0, 1920, 380)

    # Crop the image
    img_cropped_0 = img.crop(crop_box_0)
    img_cropped_1 = img.crop(crop_box_1)
    img_cropped_2 = img.crop(crop_box_2)
    img_cropped_3 = img.crop(crop_box_3)

    # Save the cropped image
    img_cropped_0.save(url + '/cropped_image_0.jpg')
    img_cropped_1.save(url + '/cropped_image_1.jpg')
    img_cropped_2.save(url + '/cropped_image_2.jpg')
    img_cropped_3.save(url + '/cropped_image_3.jpg')

def Crop(urlimage, urlfolder):
    img = Image.open(urlimage)
    crop_box_0 = (1400, 0, 1920, 800)
    crop_box_1 = (0, 800, 1920, 1400)
    crop_box_2 = (0, 1375, 1920, 1750)
    crop_box_3 = (0, 1750, 1920, 2560)

    img_cropped_0 = img.crop(crop_box_0)
    img_cropped_1 = img.crop(crop_box_1)
    img_cropped_2 = img.crop(crop_box_2)
    img_cropped_3 = img.crop(crop_box_3)
    
    img_cropped_0.save(urlfolder +'/cropped_image_0.jpg')
    img_cropped_1.save(urlfolder +'/cropped_image_1.jpg')
    img_cropped_2.save(urlfolder +'/cropped_image_2.jpg')
    img_cropped_3.save(urlfolder +'/cropped_image_3.jpg')

def subdiv_part3(image_path, model, output_dir):
    # if not os.path.exists(image_path):
    #     raise FileNotFoundError(f"File ảnh không tồn tại: {image_path}")
    # if not os.path.exists(model_path):
    #     raise FileNotFoundError(f"File model không tồn tại: {model_path}")
    # Dự đoán trên ảnh đầu vào
    results = model(image_path)

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
        filtered_boxes = [box for box in sorted_boxes if box[4] >= 0.9][:6]

        # Lấy kích thước của ảnh gốc từ thuộc tính "orig_shape"
        image_size = (result.orig_shape[1], result.orig_shape[0])

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

# **************************************************************************************************
# ********************************* FUNC PART3 *****************************************************
# **************************************************************************************************

def get_three_lowest_y_coords(coords):
    return sorted(coords, key=lambda x: x[0][1])[:3]

def calculate_average_y(groups):
    rows = []
    for i in range(0, len(groups), 4):
        group = groups[i:i+4]
        avg_y = np.mean([item[0][1] for item in group])
        rows.append(avg_y)
    return rows

def get_result_value(closest_index, y, decimal, matrix):
    if closest_index == 0:
        return (0, 0)
    elif closest_index == 1:
        decimal_y1, decimal_y2 = [item[0][1] for item in decimal]
        y1_diff = abs(y - decimal_y1)
        y2_diff = abs(y - decimal_y2)
        if y1_diff < y2_diff:
            return (1, 1)
        else:
            return (1, 2)
    else:
        min_diff = float('inf')
        result_value = None
        for i in range(4):
            matrix_y = matrix[closest_index - 2][i][1]
            diff = abs(y - matrix_y)
            if diff < min_diff:
                min_diff = diff
                result_value = (closest_index, i)
        return result_value

def find_closest_row_index(y, row_circle):
    min_diff = float('inf')
    closest_index = None
    
    for i, row_y in enumerate(row_circle):
        diff = abs(y - row_y)
        if diff < min_diff:
            min_diff = diff
            closest_index = i
    
    return closest_index

def extract_first_element_tuples(finally_result):
    return [item[0] for item in finally_result]

def checking_part3(image_path, model):
    result = model(image_path)

    circle = []
    choice = []
    finally_result = []
    key_map = ["-", ".", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    combined = []
    for r in result:
        boxes = r.boxes.xywh.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy()
        centers = np.column_stack((boxes[:, 0], boxes[:, 1]))
        for center, confidence, class_id, box in zip(centers, confidences, class_ids, boxes):
            if class_id == 1:
                combined.append((tuple(center), confidence, class_id, box))
            elif class_id == 0 and confidence >= 0.9:
                choice.append((tuple(center), confidence, class_id, box))
    # Sắp xếp danh sách theo độ tin cậy giảm dần
    combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)

    smallest_area = float('inf')

    # Duyệt qua tất cả các phần tử trong combined_sorted
    for boxmin in combined_sorted:
        # Tính diện tích của hộp hiện tại
        box_area = calculate_area(boxmin[3][2], boxmin[3][3])
        
        # So sánh với diện tích nhỏ nhất hiện tại
        if box_area < smallest_area:
            smallest_area = box_area
    highest_conf_box_area = smallest_area

    # Lọc các bound box dựa trên sự chênh lệch diện tích
    filtered_combined = []
    removed_count = 0
    for box in combined_sorted:
        area = calculate_area(box[3][2], box[3][3])
        if abs(area - highest_conf_box_area) > 10000:
            removed_count += 1
        else:
            filtered_combined.append(box)
    filtered_combined_with_conf = []
    for box in filtered_combined:
        center, confidence, class_id, coords = box
        if confidence >= 0.8:
            filtered_combined_with_conf.append(box)

    circle.extend(filtered_combined_with_conf)
    # Lưu trữ vào danh sách circle
    
    lowest_y_circle = get_three_lowest_y_coords(circle)
    negative = lowest_y_circle[0]
    decimal = sorted(lowest_y_circle[1:], key=lambda x: x[0][0])
    circle = [item for item in circle if item not in lowest_y_circle]
    choice.sort(key=lambda x: x[0][1])

    sorted_choice = []
    for i in range(0, len(choice), 4):
        group = choice[i:i+4]
        group.sort(key=lambda x: x[0][0])
        sorted_choice.extend(group)

    circle.sort(key=lambda x: x[0][1])

    sorted_circle = []
    for i in range(0, len(circle), 4):
        group = circle[i:i+4]
        group.sort(key=lambda x: x[0][0])
        sorted_circle.extend(group)
    row_circle = [negative[0][1]]
    row_circle.extend(calculate_average_y(sorted_circle))
    row_circle.insert(1, np.mean([item[0][1] for item in decimal]))

    matrix = np.array([item[0] for item in sorted_circle]).reshape(-1, 4, 2)
    
    for item in sorted_choice:
        y = item[0][1]
        x = item[0][0]
        closest_index = find_closest_row_index(y, row_circle)
        result_value = get_result_value(closest_index, y, decimal, matrix)
        finally_result.append((result_value))
    result_array = extract_first_element_tuples(finally_result)

    string_array = []
    for i in range(len(result_array)):
        string_array.append(key_map[result_array[i]])

    result_string = "".join([string_array[i] for i in range(len(string_array))])
    # print(result_string)
    return result_string

# **************************************************************************************************
# ********************************* FUNC PART2 *****************************************************
# **************************************************************************************************

def generate_output(key_map_21, key_map_22, combined_indices, offset):
        result = []
        for i, (index_21, index_22) in enumerate(combined_indices):
            prefix = str(1 + offset) if index_21 in [0, 1] else str(2 + offset)
            output_string = f"{prefix}{key_map_22[index_22]}-{key_map_21[index_21]}"
            result.append(output_string)
        return result

def sorting_ans_p2(key_map_21, key_map_22, combined_indices_list):
    all_outputs = []
    for loop_index, combined_indices in enumerate(combined_indices_list):
        offset = loop_index * 2
        output = generate_output(key_map_21, key_map_22, combined_indices, offset)
        all_outputs.extend(output)

    # Sắp xếp lại kết quả theo tiền tố
    all_outputs.sort(key=lambda x: int(x[0]))
    return all_outputs

def filter_boxes_by_area(combined_sorted):
    smallest_area = float('inf')

    # Duyệt qua tất cả các phần tử trong combined_sorted
    for boxmin in combined_sorted:
        # Tính diện tích của hộp hiện tại
        box_area = calculate_area(boxmin[3][2], boxmin[3][3])
        
        # So sánh với diện tích nhỏ nhất hiện tại
        if box_area < smallest_area:
            smallest_area = box_area
    highest_conf_box_area = smallest_area

    # Lọc các bound box dựa trên sự chênh lệch diện tích
    filtered_combined = []
    removed_count = 0
    for box in combined_sorted:
        area = calculate_area(box[3][2], box[3][3])
        if abs(area - highest_conf_box_area) > 10000:
            removed_count += 1
        else:
            filtered_combined.append(box)

    return filtered_combined

def checking_part2(img, model):
    circle = []
    choice = []
    combined = []
    combined_choice = []

    # result = model(image_path)
    # model = YOLO('D:/BKHN/20242/AI/ChamThi/runs/detect/train5/weights/best.pt')
    # result = model('D:/BKHN/20242/AI/ChamThi/Part1/cropped_image_1.jpg')
    result = model(img)
    for r in result:
        boxes = r.boxes.xywh.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy()
        centers = np.column_stack((boxes[:, 0], boxes[:, 1]))
        for center, confidence, class_id, box in zip(centers, confidences, class_ids, boxes):
            if class_id == 1:
                combined.append((tuple(center), confidence, class_id, box))
            elif class_id == 0 and confidence >= 0.9:
                combined_choice.append((tuple(center), confidence, class_id, box))
    # Sắp xếp danh sách theo độ tin cậy giảm dần
    combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)

    # Lưu trữ vào danh sách circle
    choice.extend(filter_boxes_by_area(combined_choice))
    circle.extend(filter_boxes_by_area(combined_sorted))

    sorted_choice = sort_objects(choice, 4)
    sorted_circle = sort_objects(circle, 4)
    #print(len(sorted_choice))
    row_circle = calculate_average_y(sorted_circle)

    matrix = process_circle(sorted_circle)

    x = find_min_x_indices(sorted_choice, find_closest_indices(sorted_choice, row_circle), matrix)
    y = find_closest_indices(sorted_choice, row_circle)
    combined_indices = [(x, y) for x, y in zip(x, y)]
    combined_indices.sort(key=lambda x: x[1])    
    return(combined_indices)

# **************************************************************************************************
# ********************************* FUNC PART1 *****************************************************
# **************************************************************************************************

def calculate_area(w, h):
    return w * h

# def process_circle(circle):
#     coordinates = [item[0] for item in circle]
#     num_elements = len(coordinates)
#     num_rows = num_elements // 4
#     if num_elements % 4 != 0:
#         num_rows += 1

#     matrix = np.zeros((num_rows * 4, 2))
#     for i, coord in enumerate(coordinates):
#         matrix[i] = coord
#     matrix = matrix[:num_elements].reshape(num_rows, 4, 2)
#     return matrix
def process_circle(circle):
    coordinates = [item[0] for item in circle]
    num_elements = len(coordinates)
    
    # Đảm bảo số phần tử là bội số của 4
    if num_elements % 4 != 0:
        # Tính số hàng cần thiết
        num_rows = num_elements // 4 + 1
        # Thêm các phần tử giả để đủ số lượng
        padding_needed = num_rows * 4 - num_elements
        for _ in range(padding_needed):
            coordinates.append((0, 0))  # Thêm tọa độ giả
        num_elements = len(coordinates)
    else:
        num_rows = num_elements // 4
    
    # Tạo ma trận và điền dữ liệu
    matrix = np.zeros((num_elements, 2))
    for i, coord in enumerate(coordinates):
        matrix[i] = coord
    
    # Reshape thành ma trận 3D
    matrix = matrix.reshape(num_rows, 4, 2)
    return matrix

def sort_objects(objects, group_size):
    objects.sort(key=lambda x: x[0][1])
    sorted_objects = []
    for i in range(0, len(objects), group_size):
        group = objects[i:i+group_size]
        group.sort(key=lambda x: x[0][0])
        sorted_objects.extend(group)
    return sorted_objects

def calculate_average_y(groups):
    rows = []
    for i in range(0, len(groups), 4):
        group = groups[i:i+4]
        avg_y = np.mean([item[0][1] for item in group])
        rows.append(avg_y)
    return rows

def process_answers(positions, answers):
    combined2 = list(zip(positions, answers))
    result_dict = {}
    for (a, b), answer in combined2:
        if b not in result_dict:
            result_dict[b] = []
        result_dict[b].append(answer)

    # Tạo kết quả cuối cùng
    final_result = []
    for question in sorted(result_dict.keys()):
        answers_list = result_dict[question]
        if len(answers_list) == 0:
            final_result.append(f"{question} - null")
        elif len(answers_list) == 1:
            final_result.append(f"{question} - {answers_list[0]}")
        else:
            final_result.append(f"{question} - ({', '.join(answers_list)})")

    return final_result

def find_closest_indices(sorted_choice, row_circle):
    closest_indices = []
    for choice_coord in sorted_choice:
        choice_y = choice_coord[0][1]
        min_error = float('inf')
        closest_index = None
        for i, circle_y in enumerate(row_circle):
            error = abs(choice_y - circle_y)
            if error < min_error:
                min_error = error
                closest_index = i
        closest_indices.append(closest_index)
    return closest_indices

def find_min_x_indices(sorted_choice, closest_indices, matrix):
    min_x_indices = []
    for i, choice_coord in enumerate(sorted_choice):
        choice_x = choice_coord[0][0]
        row_index = closest_indices[i]
        row = matrix[row_index]
        row_x_errors = [abs(choice_x - x) for x in row[:, 0]]
        min_x_error = min(row_x_errors)
        min_x_index = row_x_errors.index(min_x_error)
        min_x_indices.append(min_x_index)
    return min_x_indices

def update_indices(result_part1):
    updated_results = []
    for i, sublist in enumerate(result_part1):
        updated_sublist = []
        offset = 10 * i + 1
        for item in sublist:
            index, value = item.split(' - ')
            new_index = int(index) + offset
            updated_sublist.append(f"{new_index} - {value}")
        updated_results.extend(updated_sublist)
    return updated_results

def checking_part1(img, model):
    circle = []
    choice = []
    key_map_12 = ["A", "B", "C", "D"]
    result_final = []
    combined = []

    # result = model(image_path)
    # model = YOLO('D:/BKHN/20242/AI/ChamThi/runs/detect/train5/weights/best.pt')
    # result = model('D:/BKHN/20242/AI/ChamThi/Part1/cropped_image_1.jpg')
    result = model(img)
    for r in result:
        boxes = r.boxes.xywh.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy()
        centers = np.column_stack((boxes[:, 0], boxes[:, 1]))
        for center, confidence, class_id, box in zip(centers, confidences, class_ids, boxes):
            if class_id == 1:
                combined.append((tuple(center), confidence, class_id, box))
            elif class_id == 0 and confidence >= 0.95:
                choice.append((tuple(center), confidence, class_id, box))
    # Sắp xếp danh sách theo độ tin cậy giảm dần
    combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)

    smallest_area = float('inf')

    # Duyệt qua tất cả các phần tử trong combined_sorted
    for boxmin in combined_sorted:
        # Tính diện tích của hộp hiện tại
        box_area = calculate_area(boxmin[3][2], boxmin[3][3])
        
        # So sánh với diện tích nhỏ nhất hiện tại
        if box_area < smallest_area:
            smallest_area = box_area
    highest_conf_box_area = smallest_area

    # Lọc các bound box dựa trên sự chênh lệch diện tích
    filtered_combined = []
    removed_count = 0
    for box in combined_sorted:
        area = calculate_area(box[3][2], box[3][3])
        if abs(area - highest_conf_box_area) > 10000:
            removed_count += 1
        else:
            filtered_combined.append(box)

    # Lưu trữ vào danh sách circle
    circle.extend(filtered_combined)

    sorted_choice = sort_objects(choice, 4)
    sorted_circle = sort_objects(circle, 4)

    row_circle = calculate_average_y(sorted_circle)

    matrix = process_circle(sorted_circle)
    #print(find_closest_indices(sorted_choice, row_circle))
    #print(find_min_x_indices(sorted_choice, find_closest_indices(sorted_choice, row_circle), matrix))

    x = find_min_x_indices(sorted_choice, find_closest_indices(sorted_choice, row_circle), matrix)
    y = find_closest_indices(sorted_choice, row_circle)
    combined_indices = [(x, y) for x, y in zip(x, y)]
    combined_indices.sort(key=lambda x: x[1])

    #print(combined_indices)

    for xloc, yloc in combined_indices:
        value = key_map_12[xloc]
        result_final.append(value)
    
    return(process_answers(combined_indices, result_final))


# **************************************************************************************************
# ********************************* BATCH PROCESSING & EXCEL EXPORT ********************************
# **************************************************************************************************

def process_all_images_and_export():
    """
    Xử lý tất cả ảnh trong thư mục Data_raw và lưu kết quả vào Excel với mỗi câu ở một cột riêng
    
    Returns:
        Đường dẫn đến file Excel kết quả
    """
    # Kiểm tra thư mục tồn tại
    if not os.path.exists(DATA_RAW_DIR):
        print(f"Thư mục {DATA_RAW_DIR} không tồn tại!")
        return
    
    # Tạo thư mục cho ảnh đã xử lý
    processed_images_dir = os.path.join(OUTPUT_DIR, "Processed_Images")
    if not os.path.exists(processed_images_dir):
        os.makedirs(processed_images_dir)
    
    # Lấy danh sách tất cả các file ảnh trong thư mục Data_raw
    image_files = []
    for root, _, files in os.walk(DATA_RAW_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    # Tạo DataFrame để lưu kết quả
    results_data = {
        'Image_Name': [],
        'Image_Path': []
    }
    
    # Xác định số lượng câu tối đa cho mỗi phần
    max_part1_questions = 40  # Giả sử tối đa 40 câu cho phần 1
    
    # Tạo các cột cho Part 1
    for i in range(1, max_part1_questions + 1):
        results_data[f'P1_Q{i}'] = []
    
    # Tạo các cột cho Part 2
    # Part 2 có định dạng 1A, 1B, 1C, 1D, 2A, 2B, 2C, 2D, v.v.
    max_part2_numbers = 8  # Giả sử có 8 số (1-8)
    part2_letters = ['A', 'B', 'C', 'D']  # Các chữ cái cho mỗi số
    for num in range(1, max_part2_numbers + 1):
        for letter in part2_letters:
            results_data[f'P2_{num}{letter}'] = []
    
    # Tạo các cột cho Part 3
    max_part3_questions = 6  # Giả sử tối đa 10 câu cho phần 3
    for i in range(1, max_part3_questions + 1):
        results_data[f'P3_Q{i}'] = []
    
    # Xử lý từng ảnh
    total_images = len(image_files)
    for i, image_path in enumerate(image_files, 1):
        image_name = os.path.basename(image_path)
        print(f"[{i}/{total_images}] Đang xử lý ảnh: {image_name}")
        
        try:
            # Tạo thư mục riêng cho ảnh này
            image_output_dir = os.path.join(processed_images_dir, os.path.splitext(image_name)[0])
            if not os.path.exists(image_output_dir):
                os.makedirs(image_output_dir)
            
            # Đường dẫn cho các file tạm thời
            temp_gray_path = os.path.join(image_output_dir, "gray.jpg")
            temp_warped_path = os.path.join(image_output_dir, "transformed.jpg")
            temp_folder = os.path.join(image_output_dir, "crops")
            if not os.path.exists(temp_folder):
                os.makedirs(temp_folder)
            
            # Xử lý ảnh
            convert_to_gray(image_path, temp_gray_path)
            Tranformer(temp_gray_path, detect_square_model, temp_warped_path)
            Crop(temp_warped_path, temp_folder)
            
            # Xử lý Part1
            part1_folder = os.path.join(image_output_dir, "Part1")
            if not os.path.exists(part1_folder):
                os.makedirs(part1_folder)
            Crop1(os.path.join(temp_folder, 'cropped_image_1.jpg'), part1_folder)
            
            # Xử lý Part2
            part2_folder = os.path.join(image_output_dir, "Part2")
            if not os.path.exists(part2_folder):
                os.makedirs(part2_folder)
            Crop2(os.path.join(temp_folder, 'cropped_image_2.jpg'), part2_folder)
            
            # Xử lý Part3
            part3_folder = os.path.join(image_output_dir, "Part3")
            if not os.path.exists(part3_folder):
                os.makedirs(part3_folder)
            subdiv_part3(os.path.join(temp_folder, 'cropped_image_3.jpg'), detect_rect_model, part3_folder)
            
            # Lấy kết quả từ các phần
            part1_results = []
            part1_image_files = [f for f in os.listdir(part1_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_file in part1_image_files:
                part1_results.append(checking_part1(os.path.join(part1_folder, img_file), detect_circle_model_update))
            
            part2_results = []
            part2_image_files = [f for f in os.listdir(part2_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_file in part2_image_files:
                part2_results.append(checking_part2(os.path.join(part2_folder, img_file), detect_circle_model_update1))
            
            part3_results = []
            part3_image_files = [f for f in os.listdir(part3_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for img_file in part3_image_files:
                part3_results.append(checking_part3(os.path.join(part3_folder, img_file), detect_circle_model))
            
            # Chuẩn bị dữ liệu cho Excel
            results_data['Image_Name'].append(image_name)
            results_data['Image_Path'].append(image_path)
            
            # Xử lý kết quả Part 1
            part1_updated = update_indices(part1_results)
            part1_dict = {}
            
            # Chuyển đổi kết quả thành dictionary với key là số câu
            for item in part1_updated:
                if isinstance(item, str):
                    parts = item.split(' - ')
                    if len(parts) == 2:
                        question_num = int(parts[0])
                        answer = parts[1]
                        part1_dict[question_num] = answer
            
            # Điền kết quả vào các cột tương ứng
            for q_num in range(1, max_part1_questions + 1):
                if q_num in part1_dict:
                    results_data[f'P1_Q{q_num}'].append(part1_dict[q_num])
                else:
                    results_data[f'P1_Q{q_num}'].append('')
            
            # Xử lý kết quả Part 2
            part2_sorted = sorting_ans_p2(key_map_21, key_map_22, part2_results)
            part2_dict = {}
            
            # Chuyển đổi kết quả thành dictionary với key là mã câu (ví dụ: "1A", "2B")
            for item in part2_sorted:
                if isinstance(item, str):
                    # Format là "1A-Đ", "1B-S", v.v.
                    # Lấy 2 ký tự đầu làm mã câu (ví dụ: "1A")
                    question_code = item[:2]
                    answer = item[3:]  # Phần sau dấu "-"
                    part2_dict[question_code] = answer
            
            # Điền kết quả vào các cột tương ứng
            for num in range(1, max_part2_numbers + 1):
                for letter in part2_letters:
                    question_code = f"{num}{letter}"
                    if question_code in part2_dict:
                        results_data[f'P2_{question_code}'].append(part2_dict[question_code])
                    else:
                        results_data[f'P2_{question_code}'].append('')
            
            # Xử lý kết quả Part 3
            part3_dict = {}
            
            # Chuyển đổi kết quả thành dictionary với key là số câu
            for i, answer in enumerate(part3_results, 1):
                part3_dict[i] = answer
            
            # Điền kết quả vào các cột tương ứng
            for q_num in range(1, max_part3_questions + 1):
                if q_num in part3_dict:
                    results_data[f'P3_Q{q_num}'].append(part3_dict[q_num])
                else:
                    results_data[f'P3_Q{q_num}'].append('')
            
            print(f"  ✓ Đã xử lý thành công ảnh {image_name}")
            
        except Exception as e:
            print(f"  ✗ Lỗi khi xử lý ảnh {image_name}: {str(e)}")
            
            # Thêm dòng cho ảnh bị lỗi với các giá trị trống
            results_data['Image_Name'].append(image_name)
            results_data['Image_Path'].append(image_path)
            
            # Điền giá trị lỗi cho tất cả các câu
            for q_num in range(1, max_part1_questions + 1):
                results_data[f'P1_Q{q_num}'].append(f"ERROR: {str(e)}")
            
            for num in range(1, max_part2_numbers + 1):
                for letter in part2_letters:
                    results_data[f'P2_{num}{letter}'].append(f"ERROR: {str(e)}")
            
            for q_num in range(1, max_part3_questions + 1):
                results_data[f'P3_Q{q_num}'].append(f"ERROR: {str(e)}")
    
    # Tạo DataFrame từ dữ liệu đã thu thập
    results_df = pd.DataFrame(results_data)
    
    # Sắp xếp DataFrame theo tên ảnh
    results_df = results_df.sort_values(by='Image_Name')
    
    # Tạo tên file Excel với timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_path = os.path.join(OUTPUT_DIR, f"Results_{timestamp}.xlsx")
    
    # Lưu DataFrame vào file Excel
    results_df.to_excel(excel_path, index=False)
    
    print(f"\nĐã hoàn thành xử lý {total_images} ảnh!")
    print(f"Kết quả đã được lưu vào: {excel_path}")
    
    return excel_path

# Chạy chương trình khi file được thực thi trực tiếp
if __name__ == "__main__":
    print("\nBắt đầu xử lý tất cả ảnh trong thư mục Data_raw...")
    # Xử lý ảnh và xuất kết quả
    process_all_images_and_export()
