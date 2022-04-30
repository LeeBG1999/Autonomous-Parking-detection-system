import json
import cv2
from core.utils import bgr2rgb, rescale_frame
from detect_car import Config
mode = "average"
# mode = "min"
# mode = "max"

# color_line_fragment = bgr2rgb((0, 102, 204))
# color_line_fragment =bgr2rgb((131,230,101))
color_line_fragment =bgr2rgb((0,0,128))

# input_image_path = "paper_image/transparented_navy_parking_lane.jpg"
# output_image_path = "paper_image/segmented_transparented_navy_parking_lane.jpg"

# input_image_path = "detect_car/result_detect_car/ch01_00000000006043700.jpg"
# input_image_path = "detect_car/result_detect_car/ch01_00000000006010700.jpg"

#---------------------------------------------------------------------
# original path
input_image_path = "data/01.2021-left/ch01_00000000000000000.jpg"
output_image_path = "fragment/image_draw_fragment.jpg"
#---------------------------------------------------------

# f = open("detect_car/list_coordinate_car.txt", "r")
# f1 = open("fragment/list_fragment.txt", "w+")

f = open("detect_car/coordinate_car_for_training.txt", "r")
f1 = open("fragment/list_fragment.txt", "w+")

lines = f.readlines()
list_coordinate_car = []


image = cv2.imread(input_image_path)
image = rescale_frame(image, 0.7)


def get_X_mid(car, x):
    return car[0] - x


def get_X_size(car):
    return car[2] - car[0]


for i in range(len(lines)):
    a_line = json.loads(lines[i])
    if len(a_line) != 0:
        list_coordinate_car += a_line

# print("list_coordinate_car" + str(list_coordinate_car))

arr = [0]

while True:
    my_max = 0
    my_min = 100000  # 1 so rat lon
    my_sum = 0
    count_for_sum = 0
    is_complete = None
    for i, coordinate_car in enumerate(list_coordinate_car):
        if get_X_size(coordinate_car) > get_X_mid(coordinate_car, arr[len(arr) - 1]) > 0:
            if my_max < get_X_size(coordinate_car):
                my_max = get_X_size(coordinate_car)
            if my_min > get_X_size(coordinate_car):
                my_min = get_X_size(coordinate_car)
            my_sum += get_X_size(coordinate_car)
            count_for_sum += 1
            is_complete = False
    if is_complete is False:
        if mode == "max":
            arr.append(my_max + arr[len(arr) - 1])
        elif mode == "min":
            arr.append(my_min + arr[len(arr) - 1])
        elif mode == "average":
            arr.append(int(my_sum/count_for_sum) + arr[len(arr) - 1])
    else:
        break
print(image.shape)
print(arr)
if image.shape[1] - arr[-1] < 200 or image.shape[1] -arr[-2] < 200:
    arr =arr[:-1]
    # arr[-1] = int(image.shape[1])
    arr.append(int(image.shape[1]))
else:
    arr.append(int(image.shape[1]))
print("list fragment is: " + str(arr))

a_line = json.dumps(arr) + "\n"
f1.write(a_line)
f1.close()

# create image draw fragment
# arr = [0,83,186,315,487,682,893,1113,1344]
for fragment in arr:
    cv2.line(image, (fragment, 0), (fragment, image.shape[0]), color_line_fragment, 2)
cv2.imshow("fragment created", image)
cv2.imwrite(output_image_path, image)
cv2.waitKey(0)
cv2.destroyAllWindows()
