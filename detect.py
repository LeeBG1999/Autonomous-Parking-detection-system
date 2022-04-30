import json
import os
import time
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags
from absl.flags import FLAGS
import core.utils as utils
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from detect_car.Config import my_scale, stride_of_load_input_image

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

# ------------________________________________________________________________________________
list_coordinate_car = []
colorListLine = (255, 255, 0)   # yellow
colorListLine2 = (255, 0, 0)    # red
color_rectangle = (255, 0, 0)   # red
color_meters = (0,0,0)          # black

# ------------________________________________________________________________________________
f2 = open("detect_car/list_axis_point.txt", "r")
two_list_axis = f2.readlines()
listAxisPoint = json.loads(two_list_axis[0])
listAxisPoint_2 = json.loads(two_list_axis[1])

# ------------________________________________________________________________________________
# if not os.path.exists("detect_car/result_detect_car_new"):
#     os.mkdir("detect_car/result_detect_car_new")
# output_folder = 'detect_car/result_detect_car_new'
# f1 = open("detect_car/list_coordinate_car_new.txt", "w+")

input_folder = 'data/01.2021-left'
output_folder = 'detect_car/result_detect_car'
f1 = open("detect_car/list_coordinate_car.txt", "w+")

# ------------________________________________________________________________________________


def main(_argv):
    global list_coordinate_car
    global listAxisPoint_2
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    # session = InteractiveSession(config=config)
    # STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size

    list_img_name = os.listdir(input_folder)
    list_img_name.sort(key=utils.sort_name_input_images2)
    # load model
    # print(list_img_name)
    saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])

    checking_image = "ch01_00000000000021100.jpg"
    # loop through images in list and run Yolov4 model on each
    for count, filename in enumerate(list_img_name):
        start = time.time()
        # if filename != checking_image:
        #     continue
        # if count % stride_of_load_input_image != 0:
        #     continue
        image = cv2.imread(os.path.join(input_folder, filename))
        image = utils.rescale_frame(image, my_scale)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_data = cv2.resize(image, (input_size, input_size))

        image_data = image_data / 255.

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
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = image.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

        # hold all detection data in one variable
        pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

        allowed_classes = ['car', 'bus', 'truck']

        print(count,": ",filename[-13:])
        if count == 0:
            param = (image, colorListLine, listAxisPoint)
            param2 = (image, colorListLine2, listAxisPoint_2)
            utils.draw_line_2(param)
            utils.draw_line_2(param2)
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

        # param = (image, colorListLine, listAxisPoint)
        # param2 = (image, colorListLine2, listAxisPoint_2)
        # utils.draw_line_2(param)
        # utils.draw_line_2(param2)
        utils.draw_cars(image, list_coordinate_car, color_rectangle, color_meters)

        a_line = json.dumps(list_coordinate_car) + "\n"
        f1.write(a_line)
        list_coordinate_car = []  # dont delete this line
        image = Image.fromarray(image.astype(np.uint8))
        image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
        if count == 0 or not FLAGS.dont_show:
            cv2.imshow(filename, image)
            cv2.waitKey(0)

        cv2.imwrite(os.path.join(output_folder, filename), image)
        cv2.destroyAllWindows()
        end = time.time()
        print("Execution time: " + str(end - start))


    f1.close()
    f2.close()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
