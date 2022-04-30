# import numpy as np
# from core import utils
# from detect_car.Config import *
import cv2
import pandas as pd
import os

def rescale_frame(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimension = (width, height)
    return cv2.resize(frame, dimension, interpolation=cv2.INTER_AREA)



input_folder = "detect_space_VS_left_view_fixed/result_detect_space_left_view"
df = pd.read_excel("detect_space_VS_left_view_fixed/detected_value_on_VS_left_view_fixed_1.xlsx")

noticed_list = ["occlusion", "yolo undetected", "bus parking", "half vehicle undetected","yolo wrong detected"]
index = 0


list_img_name = os.listdir(input_folder)
list_img_name.sort()

# print(df["Images"][2])
# print(df["Images"][3])
length = 5296
count = 0
for i in range(length):
    if df["Note"][i] == noticed_list[index]:
        count += 1
        image = cv2.imread(input_folder+"/"+df["Images"][i])
        print("image number ",i,"(",count,"): ",df["Images"][i])
        # image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        image = rescale_frame(image, 0.7)
        window_name = df["Note"][i]
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()