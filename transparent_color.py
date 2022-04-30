# import cv2
# import
# image = cv2.imread("paper_image/intro.jpg")
# overlay = image.copy()
#
# x, y, w, h = 200,200,200,200  # Rectangle parameters
# cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 200, 0), -1)  # A filled rectangle
#
# alpha = 0.3  # Transparency factor.
#
# # Following line overlays transparent rectangle over the image
# image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
# cv2.imshow("transparented", image_new)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# f.close()
# ______________________________________________________________________________________
import cv2
import json
from core.utils import intersects, intersection_of_line, line

input_path = "paper_image/navy_parking_lane.jpg"
output_path = "paper_image/transparented_navy_parking_lane.jpg"

RGB_color = (0, 200, 0)

def draw_line(transparing_image, vertical_line, parking_line,alpha = 0.3 ):
    l1 = line(vertical_line[0], vertical_line[1])
    l2_1 = line(parking_line[0][0],parking_line[0][1])
    l2_2 = line(parking_line[1][0], parking_line[1][1])

    head = intersection_of_line(l1, l2_1)
    tail = intersection_of_line(l1, l2_2)
    cv2.rectangle(transparing_image, (head[0], head[1]), (tail[0]+3, tail[1]), RGB_color, 1)


f = open("detect_car/list_axis_point.txt", "r")
two_list_axis = f.readlines()
list_point = json.loads(two_list_axis[0])
list_point2 = json.loads(two_list_axis[1])

print(list_point)
print(list_point2)
# [[0, 420], [25, 420], [108, 428], [193, 429], [287, 444], [802, 483], [1032, 513], [1344, 513]]
# [[0, 450], [25, 450], [108, 466], [193, 472], [287, 492], [802, 564], [1032, 589], [1344, 589]]
image = cv2.imread(input_path)
# print(image.shape)
transparing_image = image.copy()
alpha = 0.3  # Transparency factor.
# vertical_line = [[0,0], [0, image.shape[0]]]

indx = 0
i = 0
parking_line = [list_point[indx : indx+2], list_point2[indx : indx+2]]
# for i in range(image.shape[1]-1):
print(image.shape[1]-5 )

while i < image.shape[1]-5 :

    vertical_line = [[i, 0], [i, image.shape[0]]]
    while indx < len(list_point)-1:
        print("i: ", i)
        print("line: ", indx)
        if intersects(vertical_line, parking_line[0]):
            draw_line(transparing_image, vertical_line, parking_line )
            i += 2
            vertical_line = [[i, 0], [i, image.shape[0]]]
        else:
            indx += 1
            parking_line = [list_point[indx : indx+2], list_point2[indx : indx+2]]

# Following line overlays transparent rectangle over the image
image_new = cv2.addWeighted(transparing_image, alpha, image, 1 - alpha, 0)
cv2.imshow("transparented", image_new)
cv2.imwrite(output_path,image_new)
cv2.waitKey(0)
cv2.destroyAllWindows()
f.close()
