from ultralytics import YOLO
from PIL import Image


# Load a pretrained YOLO model (recommended for training)
model = YOLO('D:/BKHN/20242/AI/ChamThi/runs/detect/train5/weights/best.pt')

result = model('D:/BKHN/20242/AI/ChamThi/Part2/cropped_image_0.jpg')

for r in result:
    print(r.boxes)
    im_array = r.plot()
    im = Image.fromarray(im_array[..., ::-1])
    im.show()
    im.save('result.jpg')