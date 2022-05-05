import json
import os
import time
import cv2
import pandas as pd
import numpy as np
from absl import flags
from absl.flags import FLAGS
import core.utils as utils
from core.utils import bgr2rgb, sort_name_input_images, line, \
    intersection_of_line, intersects
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.python.saved_model import tag_constants
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-608',
                    'path to weights file')
flags.DEFINE_integer('size', 608, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')

flags.DEFINE_string('images', 'data/01.2021-left', 'path to input image')
flags.DEFINE_string('output_folder', 'detect_car/result_detect_car', 'path to output folder')

flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('count', False, 'count objects within images')
flags.DEFINE_boolean('dont_show', True, 'dont show image output')
flags.DEFINE_boolean('info', False, 'print info on detections')
flags.DEFINE_boolean('crop', False, 'crop detections from images')
flags.DEFINE_boolean('ocr', False, 'perform generic OCR on detection regions')
flags.DEFINE_boolean('plate', False, 'perform license plate recognition')

# folder = 'data/01.2021-left'
folder = 'detect_car/testing'
result_path = 'detect_space/result_detect_space'

f1 = open("fragment/list_fragment.txt", "r")
list_fragment = json.loads(f1.read())
# [0,83,186,315,487,682,893,1113,1344]

f2 = open("detect_car/list_axis_point.txt", "r")
# f2 = open("detect_car/list_axis_point_R.txt", "r")
two_list_axis = f2.readlines()
listAxisPoint = json.loads(two_list_axis[0])
listAxisPoint_2 = json.loads(two_list_axis[1])
# [[0, 420], [25, 420], [108, 428], [193, 429], [287, 444], [802, 483], [1032, 513], [1344, 513]]
# [[0, 450], [25, 450], [108, 466], [193, 472], [287, 492], [802, 564], [1032, 589], [1344, 589]]

f3 = open("detect_space/list_xl_xs_min_space_between_car.txt", "r")
# [[43, 62, -5], [79, 94, -6], [101, 108, -9], [133, 147, -8], [158, 182, -1], [162, 174, 6], [167, 206, 1], [165, 192, 24]]
list_xl_xs = json.loads(f3.read())

#------------------------------------------------------------------
# config

color_tiny_car = bgr2rgb((102, 0, 102))
color_red = bgr2rgb((255, 0, 0))
color_large_car = bgr2rgb((0, 255, 153))
color_rectangle = bgr2rgb((255, 0, 0))
color_meters = bgr2rgb((0, 204, 0))
color_black = (0,0,0)
color_white = (255,255,255)
input_size = 608
#=-----------------------------------------------------------------
list_img_name = os.listdir(folder)
list_img_name.sort(key=sort_name_input_images)

# ----------------------------------------------------------------------
def get_real_xl_xs(x_location, frag_line):
    if frag_line < len(list_fragment) - 2 \
    and x_location + list_xl_xs[frag_line][0] > list_fragment[frag_line + 1]:
        return int(
            list_fragment[frag_line + 1] - x_location + (1 - (list_fragment[frag_line + 1] - x_location) / list_xl_xs[frag_line][1]) * list_xl_xs[frag_line + 1][1]), \
               int(list_fragment[frag_line + 1] - x_location + (1 - (list_fragment[frag_line + 1] - x_location) / list_xl_xs[frag_line][0]) * list_xl_xs[frag_line + 1][
                   0]), frag_line + 1
    else:
        return list_xl_xs[frag_line][1], list_xl_xs[frag_line][0], frag_line


def get_real_min_space(x_location, frag_line):
    #  trả về giá trị của min space và index của segment line
    if frag_line < len(list_fragment) - 2 and x_location + list_xl_xs[frag_line][2] > list_fragment[frag_line + 1]:
        return int(list_fragment[frag_line + 1] - x_location + (1 - (list_fragment[frag_line + 1] - x_location) / list_xl_xs[frag_line][2]) * list_xl_xs[frag_line + 1][
            2]), frag_line + 1
    else:
        return list_xl_xs[frag_line][2], frag_line


def draw(x_location, x_plus_w, frag_line, image, is_show_rectangle=True, color=color_rectangle):
    first_min_space, frag_line = get_real_min_space(x_location, frag_line)
    x_location += first_min_space
    xl_car, xs_car, frag_line = get_real_xl_xs(x_location, frag_line)
    min_space, frag_line = get_real_min_space(x_location + xs_car, frag_line)

    # cache
    first_xl_car = xl_car
    first_x = x_location
    second_min_space = min_space
    # cache

    list_x_space = []

# ------------------------------------------------------------------------------------------------
#     VS size

    while x_location + xs_car + min_space < x_plus_w:
        if x_location < 0:
            list_x_space.append([0, x_location + xs_car])
            x_location = 0
        else:
            list_x_space.append([x_location, x_location + xs_car])
        x_location += xs_car

# ------------------------------------------------------------------------------------------------
#     VX size(average)
#     while x_location + (xs_car+xl_car)/2 + min_space < x_plus_w:
#         if x_location < 0:
#             list_x_space.append([0, x_location + int((xs_car+xl_car)/2)])
#         else:
#             list_x_space.append([x_location, x_location + int((xs_car+xl_car)/2)])
#         x_location += int((xs_car+xl_car)/2)
# #
#------------------------------------------------------------------------------------------------

        xl_car, xs_car, frag_line = get_real_xl_xs(x_location, frag_line)
        x_location += min_space
        min_space, frag_line = get_real_min_space(x_location, frag_line)

    if len(list_x_space) == 1 and first_min_space + first_x + first_xl_car + second_min_space < x_plus_w:
        list_x_space[0][1] = first_x + first_xl_car + second_min_space
        some_space = do_draw(list_x_space, "L", image, color_large_car)
    else:
        some_space = do_draw(list_x_space, "S", image, color_tiny_car)
    return some_space


def get_height_of_space(x_space):
    for i in range(len(listAxisPoint) - 1):
        if listAxisPoint[i][0] <= x_space < listAxisPoint[i + 1][0]:
            return int((listAxisPoint_2[i][1] - listAxisPoint[i][1] + listAxisPoint_2[i + 1][1] - listAxisPoint[i + 1][1]) / 2)


def get_y_center_of_space(x_space):
    a1 = [x_space, 0]
    b1 = [x_space, 1000]
    L1 = line(a1, b1)  # so 1000 nay khong quan trong, mien lon hon 0 la dc
    for i in range(len(listAxisPoint) - 1):
        a2 = [listAxisPoint[i][0], listAxisPoint[i][1]]
        b2 = [listAxisPoint[i + 1][0], listAxisPoint[i + 1][1]]
        L2 = line(a2, b2)
        if intersection_of_line(L1, L2) is not False and intersects([a1, b1], [a2, b2]):
            return intersection_of_line(L1, L2)[1]


def do_draw(list_x_space, size_in_string, image, color):
    some_space = []

    for i in range(len(list_x_space)):
        x_space = list_x_space[i][0]
        x_space_plus_w = list_x_space[i][1]

        height_space = get_height_of_space(x_space)
        y_center_space = get_y_center_of_space(x_space)

        y_space = int(y_center_space - height_space / 2)
        y_space_plus_h = int(y_center_space + height_space / 2)

        cv2.rectangle(image, (x_space, y_space), (x_space_plus_w, y_space_plus_h), color, 2)
        cv2.putText(image, size_in_string, (x_space + 10, y_space + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color_black, 2)
        some_space.append([x_space, y_space, x_space_plus_w, y_space_plus_h])
    return some_space
#-------------------------------------------------------------------------------------------

def main(N_ignore_car):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    N_car = []
    N_space = []
    _list_image_names = []
    count = 0
    _n_of_images = len(list_img_name)
    saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-608', tags=[tag_constants.SERVING])


    start = time.time()
    for _index, filename in enumerate(list_img_name):
        start_1 = time.time()
        if _index%1000 == 0 or _index == _n_of_images-1:
            print("Processing...", round(_index*100/_n_of_images, 2), "% (", _index, "/", _n_of_images, "images)")
        _list_image_names.append(filename)
        image = cv2.imread(os.path.join(folder, filename))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_data = image.copy() / 255
        image_data = cv2.resize(image_data, (input_size, input_size), interpolation=cv2.INTER_AREA)
        # a_line = json.loads(lines[_index])
        count += 1
        # n_car_in_a_image = len(a_line)
#------------------------------------------------------------------
        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

        infer = saved_model_loaded.signatures['serving_default']
        batch_data = tf.constant(images_data)
        pred_bbox = infer(batch_data)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]

        # run non max suppression on detections
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.45,
            score_threshold=0.30
        )

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = image.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

        # hold all detection data in one variable
        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

        allowed_classes = ['car', 'bus', 'truck']

        print(_index, ": ", filename)
        # if _index == 0:
        #     param = (image, colorListLine, listAxisPoint)
        #     param2 = (image, colorListLine2, listAxisPoint_2)
        #     utils.draw_line_2(param)
        #     utils.draw_line_2(param2)
        list_coordinate_car=[]
        utils.process_bbox(pred_bbox, listAxisPoint, listAxisPoint_2, list_coordinate_car,
                           allowed_classes)
        list_coordinate_car.sort(key=utils.my_func_sort)

        # iou filter
        list_index_filter = []
        if len(list_coordinate_car) > 1:
            for i in range(len(list_coordinate_car) - 1):
                my_iou, b = utils.get_iou(list_coordinate_car[i], list_coordinate_car[i + 1])
                if my_iou > 0.7:
                    if b:
                        list_index_filter.append(i + 1)
                    else:
                        list_index_filter.append(i)
        for i in range(len(list_index_filter) - 1, -1, -1):
            list_coordinate_car.pop(list_index_filter[i])
# ------------------------------------------------------------------

        N_car.append(len(list_coordinate_car))
        list_space = []
        if len(list_coordinate_car) == 0:
            some_space = draw(0, listAxisPoint[- 1][0], 0, image)
            list_space += some_space
        for index, coordinate_car in enumerate(list_coordinate_car):
            cv2.rectangle(image, (coordinate_car[0], coordinate_car[1]), (coordinate_car[2], coordinate_car[3]),
                          color_red, 2)

            cv2.rectangle(image, (coordinate_car[0], coordinate_car[1] - 20),
                          (int(coordinate_car[0]) + (len(str(coordinate_car[-1][1]) + "_" + str(round(coordinate_car[-1][0], 2)))) * 10,
                           int(coordinate_car[1])),color_white, -1)
            cv2.putText(image, str(coordinate_car[-1][1]) + "_" + str(round(coordinate_car[-1][0], 2))
                        , (coordinate_car[0], coordinate_car[1] -5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_black, 2)

            cv2.rectangle(image, (coordinate_car[0], coordinate_car[3] ),
                          (int(coordinate_car[0]) + (len(str(coordinate_car[2] - coordinate_car[0]))) * 12,
                           int(coordinate_car[3])+ 20), color_white, -1)
            cv2.putText(image, str(coordinate_car[2] - coordinate_car[0]), (coordinate_car[0], coordinate_car[3] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_black, 2)

            if index == 0:
                x = 0
                x_plus_w = coordinate_car[0]

                some_space = draw(x, x_plus_w, 0, image)
                list_space += some_space

            if index != len(list_coordinate_car) - 1:
                x = list_coordinate_car[index][2]
                x_plus_w = list_coordinate_car[index + 1][0]

                for i in range(len(list_fragment) - 1):
                    if list_fragment[i] <= x < list_fragment[i + 1]:
                        some_space = draw(x, x_plus_w, i, image)
                        list_space += some_space

            if index == len(list_coordinate_car) - 1:
                x = coordinate_car[2]
                x_plus_w = listAxisPoint[len(listAxisPoint) - 1][0]

                some_space = draw(x, x_plus_w, len(list_fragment) - 2, image)
                list_space += some_space
        N_space.append(len(list_space))
        cv2.imwrite(os.path.join(result_path, filename), image)
        end_1 = time.time()
        proc_time = round(end_1-start_1,2)
        print("processing image: ",proc_time)

        if _index==0:
            w = int(original_w*0.7)
            h = int(original_h*0.7)
            image = cv2.resize(image,( w, h))
            cv2.imshow(filename, image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    end = time.time()

    print("Final Result Execution time: " + str(round(end - start,2)) + " seconds")
    print("---------------------------")
    print("processed images: ", count)



    df = pd.DataFrame(list(set(zip(_list_image_names, N_car,N_space))))
    df.rename(columns={"0": "Image", "1": "Detected Parking vehicle", "2":"Vacant parking slot","3": "Total slot"})
    df.to_excel("detect_space/detected_value_1.xlsx")
    # print(df)
    # print(len(N_car))
    print(len(list_img_name))
    # return N_car, N_space


if __name__ == "__main__":
    main(0)
