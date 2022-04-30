import json
import statistics


# ----------------- create list_xl_xs_min_space.txt ------------------
# nếu cần chính xác hơn thì chỉ tính những ảnh có nhiều xe
min_count_car = 7
vehicle_width_ratio_in_frag = 0.7

# f = open("detect_car/list_coordinate_car.txt", "r")
# lines = f.readlines()
# f2 = open("fragment/list_fragment.txt", "r")

f = open("detect_car/coordinate_car_for_training.txt", "r")
lines = f.readlines()
f2 = open("fragment/list_fragment.txt", "r")


list_fragment = json.loads(f2.read())
list_car_each_segment = [[]*1 for n in range(len(list_fragment) - 1)]


list_xl_xs = [0] * (len(list_fragment) - 1)
list_counting = [0] * (len(list_fragment) - 1)
list_min_space = [0] * (len(list_fragment) - 1)
list_xl_xs_min_space_between_car = [[]] * (len(list_fragment) - 1)


#  fixxxxxxxxxxx
_under_dog = []



for i in range(len(lines)):
    a_line = json.loads(lines[i])
    for j in range(len(a_line)):
        x = a_line[j][0]
        x_plus_w = a_line[j][2]
        for k in range(len(list_fragment) - 1):
            if (list_fragment[k] <= x and x_plus_w <= list_fragment[k + 1]) \
                    or (list_fragment[k] <= x and (list_fragment[k + 1] - x) > (x_plus_w - x) * vehicle_width_ratio_in_frag) \
                    or (x_plus_w <= list_fragment[k+1] and (x_plus_w - list_fragment[k]) > (x_plus_w - x) * vehicle_width_ratio_in_frag):
                list_car_each_segment[k].append(x_plus_w - x)

# --------------------------------------------------------------------------------
#                 if k == 6:
#                     if x_plus_w - x < 79:
#                         _under_dog.append(i)
#                         _under_dog.append(x_plus_w - x)
# --------------------------------------------------------------------------------

    for j in range(len(a_line) - 1):
        for k in range(len(list_fragment) - 1):
            if list_fragment[k] < a_line[j][2] < list_fragment[k+1] \
                    and list_fragment[k] < a_line[j+1][0] < list_fragment[k+1]\
                    and len(a_line) > min_count_car:
                if j >0 :
                    list_min_space[k] += min((a_line[j+1][0] - a_line[j][2]),(a_line[j][0]-a_line[j-1][2]))
                else:
                    list_min_space[k] += a_line[j+1][0] - a_line[j][2]
                list_counting[k] += 1

print(list_car_each_segment)
for count, car_each_segment in enumerate(list_car_each_segment):
    print("count: ", count)
    print(car_each_segment)

    car_each_segment.sort()
    try:
        xs = statistics.mean(car_each_segment[:len(car_each_segment) // 2])
    except:
        xs = list_xl_xs_min_space_between_car[count-1][0] * list_fragment[count-1]/list_fragment[count]
    try:
        xl = statistics.mean(car_each_segment[len(car_each_segment)//2:])
    except:
        xl = list_xl_xs_min_space_between_car[count-1][0] * list_fragment[count-1]/list_fragment[count]

    try:
        # if list_min_space[count]/list_counting[count] >0:
            list_xl_xs_min_space_between_car[count] = [int(xs), int(xl), int(min(list_min_space[count]/list_counting[count], xs*0.15))]
        # else:
        #     list_xl_xs_min_space_between_car[count] = [int(xs), int(xl),1]

    except ZeroDivisionError:
            list_xl_xs_min_space_between_car[count] = [int(xs), int(xl), int(xs*0.15)]


print(list_xl_xs_min_space_between_car)

f3 = open("detect_space/list_xl_xs_min_space_between_car.txt", "w+")
a_line = json.dumps(list_xl_xs_min_space_between_car)
f3.write(a_line)

f.close()
f2.close()
f3.close()

