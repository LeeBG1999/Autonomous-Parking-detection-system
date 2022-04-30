#import
import os
import cv2
import json


#----------------------------------------------------------------
#calibration

training_limited = 150
limited_cars_per_image = 8
_filter = True

#----------------------------------------------------------------
#path

data_path = "detect_car/result_detect_car"
detected_image_for_training_Path = "detect_car/training"
detected_image_for_testing_Path ="detect_car/testing"
coordinate_car_file = open("detect_car/list_coordinate_car.txt", "r")
coordinate_for_testing = open("detect_car/coordinate_car_for_testing.txt", "w+")
coordinate_for_training = open("detect_car/coordinate_car_for_training.txt", "w+")

lines = coordinate_car_file.readlines()


#--------------------------------------------------------------------
#checking


if not os.path.exists(detected_image_for_training_Path):
    os.mkdir(detected_image_for_training_Path)
    print("Folder for training dataset is created")
else:
    print("Folder for training dataset is already existed")

if not os.path.exists(detected_image_for_testing_Path):
    os.mkdir(detected_image_for_testing_Path)
    print("\nFolder for testing dataset is created")
else:
    print("Folder for testing dataset is already existed")

#----------------------------------------------------------------
#main code
list_img_name = os.listdir(data_path)
coordinate_car_for_testing = []
coordinate_car_for_training = []
satisfying_count = 0
testing_image_count = 0
index =[]
print("Munber of images: ",len(lines),"\n")
for i in range(len(lines)):
    a_line = json.loads(lines[i])
    if len(a_line) >= limited_cars_per_image:
        satisfying_count += 1
print("\nNumber of images that satisfy the condition: ",satisfying_count)
print("Number of images that in need: ",training_limited,"\n")

if _filter:
    satisfying_count = 0
    for i in range(len(lines)):
        a_line = json.loads(lines[i])
        if list_img_name[i] == "ch01_00000000003032700.png":
            continue
        elif len(a_line) >= limited_cars_per_image:


            satisfying_count += 1
            if satisfying_count % 50 == 0 or satisfying_count == training_limited:
                print(f"Training images processing...{satisfying_count}/{training_limited}")

            if satisfying_count > training_limited:
                break
            else:
                coordinate_car_for_training.append(a_line)
                image = cv2.imread(os.path.join(data_path, list_img_name[i]))
                cv2.imwrite(os.path.join(detected_image_for_training_Path, list_img_name[i]), image)
                index.append(i)
                a_line = json.dumps(a_line) + "\n"
                coordinate_for_training.write(a_line)
    print("Training images processing is done!!!\n")
    coordinate_for_training.close()

    testing_total = len(lines)-len(index)
    for i in range(len(lines)):
        if list_img_name[i] == "ch01_00000000003032700.png":
            continue
        elif i not in index:
            testing_image_count += 1
            if testing_image_count % 50 == 0 or testing_image_count == testing_total:
                print(f"Testing images processing...{testing_image_count}/{testing_total}")

            a_line = json.loads(lines[i])
            image = cv2.imread(os.path.join(data_path, list_img_name[i]))
            cv2.imwrite(os.path.join(detected_image_for_testing_Path, list_img_name[i]), image)
            coordinate_car_for_testing.append(a_line)
            a_line = json.dumps(a_line) + "\n"
            coordinate_for_testing.write(a_line)

    coordinate_for_testing.close()
else:
    coordinate_for_training.close()
    coordinate_for_testing.close()
#----------------------------------------------------------------
# manipulate/visualize/...
print("Done!!!")
#----------------------------------------------------------------
#output