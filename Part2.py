from ultralytics import YOLO
import numpy as np
import cv2
import os

# Load a pretrained YOLO model
model = YOLO('D:/BKHN/20242/AI/ChamThi/runs/detect/train5/weights/best.pt')
image_path = 'D:/BKHN/20242/AI/ChamThi/Part2/cropped_image_3.jpg'
result = model(image_path)

# Khởi tạo danh sách
circle = []
choice = []
key_map_21 = ["Đ", "S", "Đ", "S"]
key_map_22 = ["A", "B", "C", "D"]
result_final = []
combined = []

# Tạo thư mục output nếu chưa tồn tại
output_dir = 'D:/BKHN/20242/AI/ChamThi/Part2'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Đọc hình ảnh để vẽ lên
image = cv2.imread(image_path)
if image is None:
    print(f"Không thể đọc hình ảnh từ {image_path}")
else:
    # Tạo bản sao của hình ảnh để vẽ lên
    visualization = image.copy()
    
    # Màu sắc cho các lớp khác nhau
    circle_color = (0, 255, 0)  # Xanh lá cho circle (class_id=1)
    choice_color = (0, 0, 255)  # Đỏ cho choice (class_id=0)

def calculate_area(w, h):
    return w * h

def process_circle(circle):
    coordinates = [item[0] for item in circle]
    num_elements = len(coordinates)
    num_rows = num_elements // 4
    if num_elements % 4 != 0:
        num_rows += 1
    matrix = np.zeros((num_rows * 4, 2))
    for i, coord in enumerate(coordinates):
        matrix[i] = coord
    matrix = matrix[:num_elements].reshape(num_rows, 4, 2)
    return matrix

def sort_objects(objects, group_size):
    objects.sort(key=lambda x: x[0][1])
    sorted_objects = []
    for i in range(0, len(objects), group_size):
        group = objects[i:i+group_size]
        group.sort(key=lambda x: x[0][0])
        sorted_objects.extend(group)
    return sorted_objects

def calculate_average_y(groups, group_size):
    rows = []
    for i in range(0, len(groups), group_size):
        group = groups[i:i+4]
        avg_y = np.mean([item[0][1] for item in group])
        rows.append(avg_y)
    return rows

for r in result:
    boxes = r.boxes.xywh.cpu().numpy()
    confidences = r.boxes.conf.cpu().numpy()
    class_ids = r.boxes.cls.cpu().numpy()
    centers = np.column_stack((boxes[:, 0], boxes[:, 1]))
    
    # Chuyển đổi từ xywh sang xyxy để vẽ bounding box
    boxes_xyxy = []
    for box in boxes:
        x, y, w, h = box
        x1 = int(x - w/2)
        y1 = int(y - h/2)
        x2 = int(x + w/2)
        y2 = int(y + h/2)
        boxes_xyxy.append([x1, y1, x2, y2])
    
    # Lưu thông tin và vẽ bounding box
    for (x1, y1, x2, y2), center, confidence, class_id, box in zip(boxes_xyxy, centers, confidences, class_ids, boxes):
        # Chọn màu dựa trên class_id
        color = circle_color if class_id == 1 else choice_color
        
        if class_id == 1:
            combined.append((tuple(center), confidence, class_id, box))
            # Vẽ bounding box cho circle không có label
            if 'visualization' in locals():
                cv2.rectangle(visualization, (x1, y1), (x2, y2), color, 2)
        elif class_id == 0 and confidence >= 0.9:
            choice.append((tuple(center), confidence, class_id, box))
            # Vẽ bounding box và label cho choice
            if 'visualization' in locals():
                cv2.rectangle(visualization, (x1, y1), (x2, y2), color, 2)
                
                # Hiển thị thông tin chỉ cho choice
                label = f"Choice: {confidence:.2f}"
                
                # Đặt văn bản ở vị trí phù hợp
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                
                # Vẽ nền cho văn bản
                cv2.rectangle(visualization, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
                
                # Vẽ văn bản
                cv2.putText(visualization, label, (x1, y1 - 5), font, font_scale, (255, 255, 255), font_thickness)

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
        
        # Đánh dấu các circle bị loại bỏ nếu hình ảnh đã được đọc thành công
        if 'visualization' in locals():
            x, y = int(box[0][0]), int(box[0][1])
            cv2.drawMarker(visualization, (x, y), (0, 0, 0), cv2.MARKER_CROSS, 20, 3)
    else:
        filtered_combined.append(box)

# Lưu trữ vào danh sách circle
circle.extend(filtered_combined)

# Thêm thông tin về số lượng circle bị loại bỏ nếu hình ảnh đã được đọc thành công
if 'visualization' in locals():
    info_text = f"Removed Circles: {removed_count}/{len(combined_sorted)}"
    cv2.putText(visualization, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Thêm thông tin về diện tích nhỏ nhất
    info_text = f"Smallest Circle Area: {smallest_area:.0f}"
    cv2.putText(visualization, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

sorted_choice = sort_objects(choice, 4)
sorted_circle = sort_objects(circle, 4)
row_circle = calculate_average_y(sorted_circle, 4)

matrix = process_circle(sorted_circle)

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

def generate_output(key_map_21, key_map_22, combined_indices, offset):
    result = []
    
    # Kiểm tra xem combined_indices có phải là danh sách các cặp không
    if not isinstance(combined_indices, list):
        print(f"Lỗi: combined_indices không phải là danh sách: {combined_indices}")
        return result
    
    for i, indices in enumerate(combined_indices):
        # Kiểm tra xem indices có phải là tuple hoặc list không
        if not isinstance(indices, (tuple, list)) or len(indices) != 2:
            print(f"Lỗi: indices không phải là cặp giá trị: {indices}")
            continue
        
        index_21, index_22 = indices
        
        # Kiểm tra xem index_21 và index_22 có nằm trong phạm vi hợp lệ không
        if not (0 <= index_21 < len(key_map_21) and 0 <= index_22 < len(key_map_22)):
            print(f"Lỗi: index_21={index_21} hoặc index_22={index_22} nằm ngoài phạm vi")
            continue
        
        prefix = str(1 + offset) if index_21 in [0, 1] else str(2 + offset)
        output_string = f"{prefix}{key_map_22[index_22]}-{key_map_21[index_21]}"
        result.append(output_string)
    
    return result

closest_indices = find_closest_indices(sorted_choice, row_circle)
min_x_indices = find_min_x_indices(sorted_choice, closest_indices, matrix)

# Kiểm tra kết quả
print("closest_indices:", closest_indices)
print("min_x_indices:", min_x_indices)

# Tạo combined_indices
combined_indices = []
for i in range(len(min_x_indices)):
    combined_indices.append((min_x_indices[i], closest_indices[i]))

# Kiểm tra combined_indices
print("combined_indices trước khi sắp xếp:", combined_indices)

# Sắp xếp combined_indices
combined_indices.sort(key=lambda x: x[1])
print("combined_indices sau khi sắp xếp:", combined_indices)

# Vòng lặp để tạo all_outputs
all_outputs = []
for loop_index, indices in enumerate(combined_indices):
    offset = loop_index * 2
    output = generate_output(key_map_21, key_map_22, [indices], offset)
    all_outputs.extend(output)

# Sắp xếp lại kết quả theo tiền tố
all_outputs.sort(key=lambda x: int(x[0]))

print(all_outputs)

# Hiển thị và lưu hình ảnh nếu đã được đọc thành công
if 'visualization' in locals():
    # Hiển thị hình ảnh
    cv2.imshow("Part2 Visualization", visualization)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Lưu hình ảnh
    output_path = os.path.join(output_dir, f"part2_visualization_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, visualization)
    print(f"Đã lưu hình ảnh kết quả tại: {output_path}")

# for xloc, yloc in combined_indices:
#     value = key_map_12[xloc]
#     result_final.append(value)

# def process_answers(positions, answers):
#     combined2 = list(zip(positions, answers))
#     result_dict = {}
#     for (a, b), answer in combined2:
#         if b not in result_dict:
#             result_dict[b] = []
#         result_dict[b].append(answer)

#     # Tạo kết quả cuối cùng
#     final_result = []
#     for question in sorted(result_dict.keys()):
#         answers_list = result_dict[question]
#         if len(answers_list) == 0:
#             final_result.append(f"{question} - null")
#         elif len(answers_list) == 1:
#             final_result.append(f"{question} - {answers_list[0]}")
#         else:
#             final_result.append(f"{question} - ({', '.join(answers_list)})")

#     return final_result

# print(process_answers(combined_indices, result_final))


