import json
import pandas as pd
import os
from core.utils import sort_name_input_images, get_iou
from detect_car import Config

folder_path = "classify_shadow_images/not_shadow"
list_img_name = os.listdir(folder_path)
list_img_name.sort(key=sort_name_input_images)

f1 = open("detect_car/list_coordinate_car_not_shadow.txt", "r")

f2 = open("detect_car/list_coordinate_car_mask_not_shadow.txt", "r")

f3 = open("label_image/list_truth_coordinate_car_not_shadow.txt", "r")


lines1 = f1.readlines()
lines2 = f2.readlines()
lines3 = f3.readlines()

imageID = []
car_detected = []
groundTruth = []
detected_by_Yolo4 = []
boundingBox_by_Yolo4 = []
detected_by_M_RCNN = []
boundingBox_by_M_RCCN = []
c = 0
acc_yolo = [0, 0]
acc_mask = [0, 0]
for count, filename in enumerate(list_img_name):
    if count % Config.stride_of_load_input_image != 0:
        continue
    print(filename)
    a_line1 = json.loads(lines1[c])
    a_line2 = json.loads(lines2[c])
    a_line3 = json.loads(lines3[c])

    c += 1
    n = len(a_line3)
    for i in range(n):
        is_have_car = [False, False]
        if i == 0:
            imageID.append(filename)
            car_detected.append(len(a_line3))
        else:
            imageID.append("")
            car_detected.append("")

        for j in range(len(a_line1)):
            if get_iou(a_line1[j], a_line3[i])[0] > 0.7:
                detected_by_Yolo4.append("T")
                boundingBox_by_Yolo4.append(a_line1[j][2] - a_line1[j][0])
                is_have_car[0] = True
                acc_yolo[0] += 1
                break
        if not is_have_car[0]:
            acc_yolo[1] += 1
            detected_by_Yolo4.append("F")
            boundingBox_by_Yolo4.append("")
        for j in range(len(a_line2)):
            if get_iou(a_line2[j], a_line3[i])[0] > 0.7:
                detected_by_M_RCNN.append("T")
                boundingBox_by_M_RCCN.append(a_line2[j][2] - a_line2[j][0])
                is_have_car[1] = True
                acc_mask[0] += 1
                break
        if not is_have_car[1]:
            detected_by_M_RCNN.append("F")
            boundingBox_by_M_RCCN.append("")
            acc_mask[1] += 1
        groundTruth.append(a_line3[i][2] - a_line3[i][0])

df = pd.DataFrame({'imageID': imageID,
                   '#car detected': car_detected,
                   'groundTruth': groundTruth,
                   'detected by Yolo4': detected_by_Yolo4,
                   'boundingBox by Yolo4': boundingBox_by_Yolo4,
                   'detected by M-RCNN': detected_by_M_RCNN,
                   'boundingBox by M-RCCN': boundingBox_by_M_RCCN
                   })
writer = pd.ExcelWriter('valuation/compare_yolo_and_mask_not_shadow.xlsx', engine='xlsxwriter')
df.to_excel(writer, index=False)
writer.save()
print("finish compare")
print("accuracy of yolo is: " + str(acc_yolo[0] / (acc_yolo[0] + acc_yolo[1]) * 100))
print("accuracy of mask r-cnn is: " + str(acc_mask[0] / (acc_mask[0] + acc_mask[1]) * 100))