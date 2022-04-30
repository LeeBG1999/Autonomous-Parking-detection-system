import time
import cv2
import argparse
import os
import json
import numpy as np
from detect_car.Config import my_scale
import core.utils as utils
from PIL import Image

weight_path = r"data/yolov4.weights"
cfg_path = r"data/yolov4.cfg"
classes_path = r"data/classes/coco.names"

list_coordinate_car = []
colorListLine = utils.bgr2rgb((255, 255, 0))  # yellow
colorListLine2 = utils.bgr2rgb((255, 0, 0))
color_rectangle = utils.bgr2rgb((255, 0, 0))
color_meters = utils.bgr2rgb((0,0,0))


def draw_cars(img, list_coordinate_car, color_rectangle, color_meters):
    for coordinate_car in list_coordinate_car:
        x = coordinate_car[0]
        y = coordinate_car[1]
        x_plus_w = coordinate_car[2]
        y_plus_h = coordinate_car[3]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color_rectangle, 2)
        cv2.putText(img, str(x_plus_w - x), (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_meters, 2)

def on_segment(p, q, r):
    if max(p[0], q[0]) >= r[0] >= min(p[0], q[0]) and max(p[1], q[1]) >= r[1] >= min(p[1], q[1]):
        return True
    return False

def intersects(seg1, seg2):
    p1 = seg1[0]
    q1 = seg1[1]
    p2 = seg2[0]
    q2 = seg2[1]
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(p1, q1, p2):
        return True
    if o2 == 0 and on_segment(p1, q1, q2):
        return True
    if o3 == 0 and on_segment(p2, q2, p1):
        return True
    if o4 == 0 and on_segment(p2, q2, q1):
        return True
    return False

def orientation(p, q, r):
    val = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]))
    if val == 0:
        return 0
    return 1 if val > 0 else -1


def filter_cars_by_intersect(x, y, x_plus_w, y_plus_h, listAxisPoint, listAxisPoint_2,
                             list_coordinate_car, class_ids, boxes, confidences):
    # count = 0

    for i in range(len(listAxisPoint) - 1):

        segment_one = [[x, y], [x, y_plus_h]]
        segment_two = [listAxisPoint[i], listAxisPoint[i + 1]]
        segment_three = [listAxisPoint_2[i], listAxisPoint_2[i + 1]]
        # count+=1
        # print("count: ",count)
        print("line: ",i+1)
        print(intersects(segment_one, segment_two))
        if intersects(segment_one, segment_two) and not intersects(segment_one, segment_three):
            # count+=1
            # print("count: ",count)
            # print(x, x_plus_w)
            list_coordinate_car.append([x, y, x_plus_w, y_plus_h])
            print(len(list_coordinate_car))
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])
            break


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h,count):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color_rectangle, 2)

    cv2.putText(img, label+" "+str(count), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# -----------------------------------------------------------------------------------------------------
# input
f2 = open("detect_car/list_axis_point.txt", "r")

two_list_axis = f2.readlines()
listAxisPoint = json.loads(two_list_axis[0])
listAxisPoint_2 = json.loads(two_list_axis[1])

input_folder = 'data/01.2021-left'
output_folder = 'detect_car/result_detect_car'
f1 = open("detect_car/list_coordinate_car.txt", "w+")

# -----------------------------------------------------------------------------------------------------
list_img_name = os.listdir(input_folder)
list_img_name.sort(key=utils.sort_name_input_images2)
for count, filename in enumerate(list_img_name):
    start = time.time()
    image = cv2.imread(os.path.join(input_folder, filename))
    image = utils.rescale_frame(image, my_scale)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(image, (608, 608))

    image_data = image_data / 255.

    images_data = []
    for i in range(1):
        images_data.append(image_data)

# -----------------------------------------------------------------------------------------------------
    Width = image.shape[1]
    # print(Width)
    Height = image.shape[0]

    # Width = int(image.shape[1]*my_scale)
    #
    # Height = int(image.shape[0]*my_scale)

    # image = cv2.resize(image, (Width,Height), interpolation=cv2.INTER_AREA)
    scale = 0.00392

    classes = None

    with open(classes_path, 'r') as f:
        # classes = ["car","truck"]
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(weight_path, cfg_path)

    blob = cv2.dnn.blobFromImage(image, scale, (608, 608), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # Thực hiện xác định bằng HOG và SVM
    start = time.time()

    for out in outs:
        count=0
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.7:
                if class_id not in [2, 7]:
                    continue
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

                filter_cars_by_intersect(x, y, x+w, y+h, listAxisPoint, listAxisPoint_2,
                                         list_coordinate_car, class_ids, boxes, confidences)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    count_1=0
    for i in indices:
        count_1 += 1
        i = i[0]
        box = boxes[i]
        print("car number: ",count,box)
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h),count_1)
    # draw_cars(image, list_coordinate_car, color_rectangle, color_meters)

    image = cv2.resize(image, (Width, Height), interpolation=cv2.INTER_AREA)
    # cv2.imshow("object detection", image)


    a_line = json.dumps(list_coordinate_car) + "\n"
    f1.write(a_line)
    list_coordinate_car = []  # dont delete this line
    image = Image.fromarray(image.astype(np.uint8))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    if count == 0 or not True:
        param = (image, colorListLine, listAxisPoint)
        param2 = (image, colorListLine2, listAxisPoint_2)
        utils.draw_line_2(param)
        utils.draw_line_2(param2)
        cv2.imshow(filename, image)
        cv2.waitKey(0)

    # cv2.imwrite(os.path.join(output_folder, filename), image)
    # cv2.destroyAllWindows()
    # end = time.time()
    # print("Execution time: " + str(end - start))
    end = time.time()
    print("YOLO Execution time: " + str(end - start))

    cv2.waitKey()

    cv2.imwrite("object-detection.jpg", image)
    cv2.destroyAllWindows()
