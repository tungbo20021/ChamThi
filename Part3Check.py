from ultralytics import YOLO
from PIL import Image
import numpy as np
import math

# Load a pretrained YOLO model (recommended for training)
model = YOLO('D:/BKHN/20242/AI/ChamThi/runs/detect/train5/weights/best.pt')

result = model('D:/BKHN/20242/AI/ChamThi/output/crop_box_number_2.jpg')

circle = []
choice = []
finally_result = []
key_map = ["-", ".", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

for r in result:
    boxes = r.boxes.xyxy.cpu().numpy()
    confidences = r.boxes.conf.cpu().numpy()
    class_ids = r.boxes.cls.cpu().numpy()

    centers = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2))

    for center, confidence, class_id in zip(centers, confidences, class_ids):
        if confidence < 0.85:
            continue
        if class_id == 1:
            circle.append((tuple(center), confidence, class_id))
        elif class_id == 0:
            choice.append((tuple(center), confidence, class_id))
print(len(circle))
def sort_objects(objects, group_size):
    objects.sort(key=lambda x: x[0][1])
    sorted_objects = []
    for i in range(0, len(objects), group_size):
        group = objects[i:i+group_size]
        group.sort(key=lambda x: x[0][0])
        sorted_objects.extend(group)
    return sorted_objects

def get_three_lowest_y_coords(coords):
    return sorted(coords, key=lambda x: x[0][1])[:3]

lowest_y_circle = get_three_lowest_y_coords(circle)
negative = lowest_y_circle[0]
decimal = sorted(lowest_y_circle[1:], key=lambda x: x[0][0])
circle = [item for item in circle if item not in lowest_y_circle]

sorted_choice = sort_objects(choice, 4)
sorted_circle = sort_objects(circle, 4)

def calculate_average_y(groups):
    rows = []
    for i in range(0, len(groups), 4):
        group = groups[i:i+4]
        avg_y = np.mean([item[0][1] for item in group])
        rows.append(avg_y)
    return rows

row_circle = [negative[0][1]]
row_circle.extend(calculate_average_y(sorted_circle))
row_circle.insert(1, np.mean([item[0][1] for item in decimal]))
row_choice = calculate_average_y(sorted_choice)

matrix = np.array([item[0] for item in sorted_circle]).reshape(-1, 4, 2)

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

for item in sorted_choice:
    y = item[0][1]
    x = item[0][0]
    closest_index = find_closest_row_index(y, row_circle)
    result_value = get_result_value(closest_index, y, decimal, matrix)
    finally_result.append((result_value))

def extract_first_element_tuples(finally_result):
    return [item[0] for item in finally_result]

result_array = extract_first_element_tuples(finally_result)

string_array = []
for i in range(len(result_array)):
    string_array.append(key_map[result_array[i]])

result_string = "".join([string_array[i] for i in range(len(string_array))])
print(result_string)
