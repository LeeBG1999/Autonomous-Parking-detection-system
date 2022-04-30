import os
from core import utils
from detect_car.Config import *
import cv2
import json

stride_of_load_input_image = 30
input_folder = "data/01.2021-left"
list_img_name = os.listdir(input_folder)
list_img_name.sort(key=utils.sort_name_input_images2)
color_red = utils.bgr2rgb((255, 0, 0))

# output_folder = "label_image/result_label_image_shadow"
output_folder = "label_image/labeled_images"
# list_img_to_re_draw = ["2377"]
list_img_to_re_draw = []

f1_folder = "label_image/coordinate.txt"
f1 = open(f1_folder, "r")
lines = f1.readlines()
f1 = open(f1_folder, "w+")
rectangle = [0, 0, 0, 0]
drawing = False  # true if mouse is pressed
ix, iy = -1, -1


def label(event, x, y, flags, param):
    global ix, iy, drawing, rectangle
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        rectangle = [0, 0, 0, 0]

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            rectangle = [ix, iy, x, y]

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


def my_func_sort(e):
    return e[0]

alpha = 0.4
beta = 0.6
c = -1
for count, filename in enumerate(list_img_name):

    if count % stride_of_load_input_image != 0:
        continue
    c += 1

    if len(lines) != 0:
        b = False
        for i in range(len(list_img_to_re_draw)):
            if filename == list_img_to_re_draw[i] + ".jpg":
                b = True
                break
        if not b:  # khong can ve lai
            a_line = json.loads(lines[c])
            a_line.sort(key=my_func_sort)
            for rectangle in a_line:
                if rectangle[2] < rectangle[0]:
                    rectangle[2], rectangle[0] = rectangle[0], rectangle[2]
                if rectangle[3] < rectangle[1]:
                    rectangle[3], rectangle[1] = rectangle[1], rectangle[3]
                if rectangle[0] < 0:
                    rectangle[0] = 0
            f1.write(json.dumps(a_line) + '\n')

            continue

    image = cv2.imread(os.path.join(input_folder, filename))
    image = utils.rescale_frame(image, my_scale)
    list_truth_coordinate_car = []

    cv2.namedWindow(winname=filename)
    cv2.setMouseCallback(filename, label)
    while True:
        overlay = image.copy()
        if rectangle != [0, 0, 0, 0]:
            cv2.rectangle(overlay, (rectangle[0], rectangle[1]), (rectangle[2], rectangle[3]), (0, 200, 0), -1)
        k = cv2.waitKey(10)
        new_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        cv2.imshow(filename, new_image)

        if k == 32:
            new_image = cv2.addWeighted(overlay, beta, image, 1 - beta, 0)
            cv2.imwrite(os.path.join(output_folder, filename), new_image)
            image = new_image
            if rectangle[2] < rectangle[0]:
                rectangle[2], rectangle[0] = rectangle[0], rectangle[2]
            if rectangle[3] < rectangle[1]:
                rectangle[3], rectangle[1] = rectangle[1], rectangle[3]
            if rectangle[0] < 0:
                rectangle[0] = 0
            list_truth_coordinate_car.append(rectangle)
            rectangle = [0, 0, 0, 0]
        if k == 13:  # next (press enter button)
            break
    list_truth_coordinate_car.sort(key=my_func_sort)
    print(list_truth_coordinate_car)
    a_line = json.dumps(list_truth_coordinate_car) + "\n"

    f1.write(a_line)
    cv2.destroyAllWindows()

f1.close()


