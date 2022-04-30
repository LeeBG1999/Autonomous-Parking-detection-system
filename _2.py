import math
import os

# comment out below line to enable tensorflow logging outputs


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from core.utils import get_iou, intersects, intersection_of_line, line, rescale_frame
from detect_car import Config
import json

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 608, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
# flags.DEFINE_string('video', './data/video/_10s_left_camera.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('video', './data/video/tran_dai_nghia_c9_10s.mp4', 'path to input video or set to 0 for webcam')
# flags.DEFINE_string('video', './data/video/tran_dai_nghia_c9_10s_R.mp4', 'path to input video or set to 0 for webcam')
flags.DEFINE_string('output', './tracking_result/result.avi', 'path to output video')
# flags.DEFINE_string('output', './tracking_result/result_R.avi', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.50, 'score threshold')
flags.DEFINE_boolean('dont_show', True, 'dont show video output')
flags.DEFINE_boolean('info', False, 'show detailed info of tracked objects')
flags.DEFINE_boolean('count', False, 'count objects being tracked on screen')

colorListLine2 = utils.bgr2rgb((0,0,128))
colorListLine = utils.bgr2rgb((0,0,128))
output_path = "paper_image/navy_parking_lane.jpg"
f = open("detect_car/list_axis_point.txt", "w+")
an_pha = 0.8


def main(_argv):
    # Definition of the parameters
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    # initialize deep sort
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    # calculate cosine distance metric
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # initialize tracker
    tracker = Tracker(metric)

    # load configuration for object detector
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    # load tflite model if flag is set
    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    # otherwise load standard tensorflow saved model
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # begin video capture
    try:
        vid = cv2.VideoCapture(int(video_path))
    except:
        vid = cv2.VideoCapture(video_path)

    out = None

    # get video ready to save locally if flag is set
    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH) * Config.my_scale)
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT) * Config.my_scale)
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    frame_num = 0
    # while video is running

    dict_fix_car = {}
    dict_cars = {}
    frame = None
    while True:
        return_value, my_frame = vid.read()
        if return_value:
            frame_num += 1
            if frame_num % 10 != 0:
                continue
            print('Frame #: ', frame_num)
            frame = cv2.cvtColor(my_frame, cv2.COLOR_BGR2RGB)
            frame = rescale_frame(frame, Config.my_scale)

        else:
            print('Frame #: ', frame_num)
            print('Video has ended or failed, try a different video format!')
            break

        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        start_time = time.time()

        # run detections on tflite if flag is set
        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            # run detections using yolov3 if flag is set
            if FLAGS.model == 'yolov3' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=tf.constant([input_size, input_size]))
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        # convert data to numpy arrays and slice out unused elements
        num_objects = valid_detections.numpy()[0]
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:int(num_objects)]
        scores = scores.numpy()[0]
        scores = scores[0:int(num_objects)]
        classes = classes.numpy()[0]
        classes = classes[0:int(num_objects)]

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        original_h, original_w, _ = frame.shape
        bboxes = utils.format_boxes_2(bboxes, original_h, original_w)

        # store all predictions in one parameter for simplicity when calling functions
        pred_bbox = [bboxes, scores, classes, num_objects]

        # read in all class names from config
        class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # by default allow all classes in .names file
        allowed_classes = list(class_names.values())

        # custom allowed classes (uncomment line below to customize tracker for only people)
        allowed_classes = ['car', 'bus']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_indx = int(classes[i])
            class_name = class_names[class_indx]
            if class_name not in allowed_classes:
                deleted_indx.append(i)
            else:
                names.append(class_name)
        names = np.array(names)
        count = len(names)
        if FLAGS.count:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,
                        (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))
        # delete detections that are not in allowed_classes
        bboxes = np.delete(bboxes, deleted_indx, axis=0)
        scores = np.delete(scores, deleted_indx, axis=0)

        # encode yolo detections and feed to tracker
        features = encoder(frame, bboxes)
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(bboxes, scores, names, features)]

        # initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # run non-maxima supression
        boxs = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        # update tracks
        if len(dict_cars) == 0:
            dict_cars = {}
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                dict_cars[str(track.track_id)] = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
        else:
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                class_name = track.get_class()

                # draw bbox on screen
                if str(track.track_id) in dict_cars:
                    if get_iou(dict_cars[str(track.track_id)],
                                        [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])[0] > 0.8:
                        dict_fix_car[str(track.track_id)] = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                        color = colors[int(track.track_id) % len(colors)]
                        color = [i * 255 for i in color]
                        # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                        # cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
                        #               (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])),
                        #               color, -1)
                        # cv2.putText(frame, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0,
                        #             0.75, (255, 255, 255), 2)
                    else:
                        if str(track.track_id) in dict_fix_car:
                            dict_fix_car.pop(str(track.track_id))

        # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
        print("FPS: %.2f" % fps)
        # result = np.asarray(frame)

        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not FLAGS.dont_show:
            cv2.imshow("Output Video", result)

        # if output flag is set, save video file
        if FLAGS.output:
            out.write(result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

    # sort dict car
    dict_fix_car = {k: v for k, v in sorted(dict_fix_car.items(), key=lambda item: item[1])}
    print("dict_fix_car" + str(dict_fix_car))

    # map dict to list
    list_fix_car = []
    for key in dict_fix_car:
        car = dict_fix_car[key]
        list_fix_car.append(car)
    # create list axis point
    listAxisPoint = []
    listAxisPoint2 = []
    print("list_fix_car" + str(list_fix_car))

    i = 0
    h = (list_fix_car[i][3] - list_fix_car[i][1])
    listAxisPoint.append([0, int((list_fix_car[i][1] + list_fix_car[i][3]) / 2)])
    listAxisPoint2.append([0, int((list_fix_car[i][1] + list_fix_car[i][3]) / 2 + (h * an_pha))])

    for i in range(len(list_fix_car)):
        h = (list_fix_car[i][3] - list_fix_car[i][1])

        listAxisPoint.append([int((list_fix_car[i][0] + list_fix_car[i][2]) / 2),
                              int((list_fix_car[i][1] + list_fix_car[i][3]) / 2)])
        listAxisPoint2.append([int((list_fix_car[i][0] + list_fix_car[i][2]) / 2),
                               int((list_fix_car[i][1] + list_fix_car[i][3]) / 2 + (h * an_pha))])

    i = len(list_fix_car) - 1
    h = (list_fix_car[i][3] - list_fix_car[i][1])
    listAxisPoint.append([int(frame.shape[1]), int((list_fix_car[i][1] + list_fix_car[i][3]) / 2)])
    listAxisPoint2.append([int(frame.shape[1]), int((list_fix_car[i][1] + list_fix_car[i][3]) / 2 + (h * an_pha))])

    print(listAxisPoint)
    print(listAxisPoint2)
    a_line = json.dumps(listAxisPoint) + "\n"
    a_line2 = json.dumps(listAxisPoint2) + "\n"
    f.write(a_line)
    f.write(a_line2)

    param = (frame, colorListLine, listAxisPoint)
    param2 = (frame, colorListLine2, listAxisPoint2)
    utils.draw_line_without_point(param)
    utils.draw_line_without_point(param2)
    # frame = rescale_frame(fa)
    cv2.imshow("frame: " + str(frame_num), frame)
    cv2.imwrite(output_path,frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    f.close()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
