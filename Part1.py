from ultralytics import YOLO
import numpy as np

# Load a pretrained YOLO model
model = YOLO('D:/BKHN/20242/AI/ChamThi/runs/detect/train5/weights/best.pt')
result = model('D:/BKHN/20242/AI/ChamThi/Data_raw/Todien/5.jpg')

# Khởi tạo danh sách
circle = []
choice = []
key_map_12 = ["A", "B", "C", "D"]
result_final = []
combined = []

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

def calculate_average_y(groups):
    rows = []
    for i in range(0, len(groups), 4):
        group = groups[i:i+4]
        avg_y = np.mean([item[0][1] for item in group])
        rows.append(avg_y)
    return rows

for r in result:
    boxes = r.boxes.xywh.cpu().numpy()
    confidences = r.boxes.conf.cpu().numpy()
    class_ids = r.boxes.cls.cpu().numpy()
    centers = np.column_stack((boxes[:, 0], boxes[:, 1]))
    for center, confidence, class_id, box in zip(centers, confidences, class_ids, boxes):
        if class_id == 1:
            combined.append((tuple(center), confidence, class_id, box))
        elif class_id == 0 and confidence >= 0.98:
            choice.append((tuple(center), confidence, class_id, box))
# Sắp xếp danh sách theo độ tin cậy giảm dần
combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)

# Lấy bound box có độ tin cậy cao nhất và tính diện tích
# highest_conf_box_1 = combined_sorted[1]
# highest_conf_box_0 = combined_sorted[0]
# highest_conf_box_2 = combined_sorted[2]
# highest_conf_box_area_0 = calculate_area(highest_conf_box_0[3][2], highest_conf_box_0[3][3])
# highest_conf_box_area_1 = calculate_area(highest_conf_box_1[3][2], highest_conf_box_1[3][3])
# highest_conf_box_area_2 = calculate_area(highest_conf_box_2[3][2], highest_conf_box_2[3][3])

# if(highest_conf_box_area_0 > highest_conf_box_area_1):
#     highest_conf_box_area = highest_conf_box_area_1
# else:
#     highest_conf_box_area = highest_conf_box_area_0

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
print(removed_count)
print(len(circle))

sorted_choice = sort_objects(choice, 4)
sorted_circle = sort_objects(circle, 4)
print(sorted_choice)
print(len(sorted_circle))
row_circle = calculate_average_y(sorted_circle)

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

print(find_closest_indices(sorted_choice, row_circle))
print(find_min_x_indices(sorted_choice, find_closest_indices(sorted_choice, row_circle), matrix))

x = find_min_x_indices(sorted_choice, find_closest_indices(sorted_choice, row_circle), matrix)
y = find_closest_indices(sorted_choice, row_circle)
combined_indices = [(x, y) for x, y in zip(x, y)]
combined_indices.sort(key=lambda x: x[1])

print(combined_indices)


