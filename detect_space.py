import json
import os
import time
import cv2
from detect_car import Config
from core.utils import bgr2rgb, rescale_frame, sort_name_input_images, sort_name_input_images2, line, \
    intersection_of_line, intersects
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler as scaling

# folder = 'data/01.2021-left'

folder = 'detect_car/testing'

result_path = 'detect_space/result_detect_space'

# if not os.path.exists(result_path):
#     os.mkdir(result_path)

# result_path = 'detect_space/result_detect_space_R'

# f = open("detect_car/list_coordinate_car.txt", "r")
f = open("detect_car/coordinate_car_for_testing.txt", "r")
lines = f.readlines()

f1 = open("fragment/list_fragment.txt", "r")
# f1 = open("fragment/list_fragment_R.txt", "r")
list_fragment = json.loads(f1.read())
# [0,83,186,315,487,682,893,1113,1344]

f2 = open("detect_car/list_axis_point.txt", "r")
# f2 = open("detect_car/list_axis_point_R.txt", "r")
two_list_axis = f2.readlines()
list_point = json.loads(two_list_axis[0])
list_point2 = json.loads(two_list_axis[1])
# [[0, 420], [25, 420], [108, 428], [193, 429], [287, 444], [802, 483], [1032, 513], [1344, 513]]
# [[0, 450], [25, 450], [108, 466], [193, 472], [287, 492], [802, 564], [1032, 589], [1344, 589]]


f3 = open("detect_space/list_xl_xs_min_space_between_car.txt", "r")
# [[43, 62, -5], [79, 94, -6], [101, 108, -9], [133, 147, -8], [158, 182, -1], [162, 174, 6], [167, 206, 1], [165, 192, 24]]
# f3 = open("detect_space/list_xl_xs_min_space_between_car_R.txt", "r")
list_xl_xs = json.loads(f3.read())

color_meters = bgr2rgb((0, 204, 0))
color_tiny_car = bgr2rgb((102, 0, 102))
color_red = bgr2rgb((255, 0, 0))
color_large_car = bgr2rgb((0, 255, 153))
color_rectangle = bgr2rgb((0, 153, 255))

list_img_name = os.listdir(folder)
list_img_name.sort(key=sort_name_input_images)
# list_img_name.sort(key=sort_name_input_images2)
# ----------------------------------------------------------------------
# def min_max_scaling(size_list):
#     transformed = scaling(feature_range=(1, 10))
#     return transformed.fit_transform(size_list)
# def spliting(datas):
#     xs = []
#     xl = []
#     minspace = []
#     for data in datas:
#         xs.append(data[0])
#         xl.append(data[1])
#         minspace.append(data[2])
#     xs = min_max_scaling(np.array(xs))
#     xl = min_max_scaling(np.array(xl))
#     minspace = min_max_scaling(np.array(minspace))
#     print(xs,xl,minspace)
#     return list((set(zip(xs, xl, minspace))))
#

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

    # while x_location + xs_car + min_space < x_plus_w:
    #     if x_location < 0:
    #         list_x_space.append([0, x_location + xs_car])
    #         x_location = 0
    #     else:
    #         list_x_space.append([x_location, x_location + xs_car])
    #     x_location += xs_car

# ------------------------------------------------------------------------------------------------
#     VX size(average)
    while x_location + (xs_car+xl_car)/2 + min_space < x_plus_w:
        if x_location < 0:
            list_x_space.append([0, x_location + int((xs_car+xl_car)/2)])
        else:
            list_x_space.append([x_location, x_location + int((xs_car+xl_car)/2)])
        x_location += int((xs_car+xl_car)/2)
#
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
    for i in range(len(list_point) - 1):
        if list_point[i][0] <= x_space < list_point[i + 1][0]:
            return int((list_point2[i][1] - list_point[i][1] + list_point2[i + 1][1] - list_point[i + 1][1]) / 2)


def get_y_center_of_space(x_space):
    a1 = [x_space, 0]
    b1 = [x_space, 1000]
    L1 = line(a1, b1)  # so 1000 nay khong quan trong, mien lon hon 0 la dc
    for i in range(len(list_point) - 1):
        a2 = [list_point[i][0], list_point[i][1]]
        b2 = [list_point[i + 1][0], list_point[i + 1][1]]
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
        cv2.putText(image, size_in_string, (x_space + 10, y_space + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_red, 2)
        some_space.append([x_space, y_space, x_space_plus_w, y_space_plus_h])
    return some_space


def main(N_ignore_car):
    N_car = []
    N_space = []
    _name = []
    c = 0
    start = time.time()
    _l = len(list_img_name)
    for count, filename in enumerate(list_img_name):
        # if count % Config.stride_of_load_input_image != 0:
        #     continue
        if count%1000 == 0 or count == _l-1:
            print("Processing...", round(count*100/_l, 2), "% (", count, "/", _l, "images)")
        _name.append(filename)
        # print("Image " + filename)
        # start = time.time()
        image = cv2.imread(os.path.join(folder, filename))
        image = rescale_frame(image, Config.my_scale)
        a_line = json.loads(lines[c])
        c += 1
        n_car_in_a_image = len(a_line)
        # ------------ ignore car--------------
        # n = N_ignore_car
        # if n != 0:
        #     if N_ignore_car > n_car_in_a_image:
        #         list_index_ignore_car = random.sample(range(n_car_in_a_image), n_car_in_a_image)
        #     else:
        #         list_index_ignore_car = random.sample(range(n_car_in_a_image), N_ignore_car)
        #     list_index_ignore_car.sort(reverse=True)
        #     for index_ignore_car in list_index_ignore_car:
        #         a_line.pop(index_ignore_car)
        #         n_car_in_a_image -= 1
        # ------------ ignore car--------------
        N_car.append(n_car_in_a_image)
        list_space = []
        if len(a_line) == 0:
            some_space = draw(0, list_point[- 1][0], 0, image)
            list_space += some_space
        for index, coordinate_car in enumerate(a_line):
            cv2.rectangle(image, (coordinate_car[0], coordinate_car[1]), (coordinate_car[2], coordinate_car[3]),
                          color_red, 2)
            if index == 0:
                x = 0
                x_plus_w = coordinate_car[0]

                some_space = draw(x, x_plus_w, 0, image)
                list_space += some_space

            if index != len(a_line) - 1:
                x = a_line[index][2]
                x_plus_w = a_line[index + 1][0]

                for i in range(len(list_fragment) - 1):
                    if list_fragment[i] <= x < list_fragment[i + 1]:
                        some_space = draw(x, x_plus_w, i, image)
                        list_space += some_space

            if index == len(a_line) - 1:
                x = coordinate_car[2]
                x_plus_w = list_point[len(list_point) - 1][0]

                some_space = draw(x, x_plus_w, len(list_fragment) - 2, image)
                list_space += some_space
        N_space.append(len(list_space))
        cv2.imwrite(os.path.join(result_path, filename), image)

    end = time.time()

    print("Final Result Execution time: " + str(round(end - start,2)) + " seconds")
    print("---------------------------")
    print("processed images: ", c)

        # cv2.imshow(filename, image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    _total = []
    for i in range(len(N_car)):
        _total.append(N_car[i] + N_space[i])
    df = pd.DataFrame(list(set(zip(_name, N_car,N_space, _total))))
    df.rename(columns={"0": "Image", "1": "Detected Parking vehicle", "2":"Vacant parking slot","3": "Total slot"})
    df.to_excel("detect_space/detected_value.xlsx")
    # print(df)
    # print(len(N_car))
    print(len(list_img_name))
    # return N_car, N_space


if __name__ == "__main__":
    main(0)
