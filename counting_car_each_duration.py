import datetime
import json
import os
import cv2
from core import utils
from detect_car.Config import stride_of_load_input_image

f = open("detect_car/list_coordinate_car.txt", "r")
lines = f.readlines()

input_folder = "data/01.2021-left"
# input_folder = "data/01.2021-right"

list_img_name = os.listdir(input_folder)

list_img_name.sort(key=utils.sort_name_input_images2)
time = datetime.datetime(2021, 6, 1, 16, 40, 0)  # anh thu 4
list_counting_car = [0] * 5
list_counting_image = [0] * 5
data_path = 'data'
data_type = ['poor', 'rich']
hour_periods = ['11pm_6am','6am_10am','10am_2pm','2pm_6pm','6pm_11pm']
for _type in data_type:
    if not os.path.isdir(data_path+'/'+_type):
        os.mkdir(data_path+'/'+_type)
for _type in data_type:
    for hour_period in hour_periods:
        if not os.path.isdir(data_path+'/'+_type+'/'+hour_period):
            os.mkdir(data_path+'/'+_type+'/'+hour_period)

for count, filename in enumerate(list_img_name):
    if count < 3:  # 3 anh dau tien khong lay vi no khong tuan theo quy luat thoi gian cac anh cach nhau 5p!
        continue
    if count % stride_of_load_input_image != 0:
        time_change = datetime.timedelta(minutes=5)
        time += time_change
        continue
    a_line = json.loads(lines[int(count / stride_of_load_input_image)])  # list coor car in one image
    if 23 <= time.hour <= 23 or 0 <= time.hour < 6:  # 11 h dem den 6 h sang hom sau
        list_counting_car[0] += len(a_line)
        list_counting_image[0] += 1
    elif 6 <= time.hour < 10:  # 6 h den 10 h sang
        list_counting_car[1] += len(a_line)
        list_counting_image[1] += 1
    if 10 <= time.hour < 14:
        list_counting_car[2] += len(a_line)
        list_counting_image[2] += 1
    if 14 <= time.hour < 18:
        list_counting_car[3] += len(a_line)
        list_counting_image[3] += 1
    else:
        list_counting_car[4] += len(a_line)
        list_counting_image[4] += 1
    time_change = datetime.timedelta(minutes=5)  # moi anh cach nhau 5p
    time += time_change

print(list_counting_car)
print(list_counting_image)

list_counting_car_average = [i / j for i, j in zip(list_counting_car, list_counting_image)]
print(list_counting_car_average)
# [60, 35, 32, 33, 170]