import cv2
import numpy as np

img_circle = cv2.imread("resources/hsv_circle.png")
circle_center = (200 - 26, 200 - 26)
circle_radius = 200 - 26
circle_min = (0, 0)
circle_max = (circle_min[0] + 2 * circle_radius, circle_min[1] + 2 * circle_radius)
circle_x = circle_center[0]
circle_y = circle_center[1]
interval = 30
line_width = 30
line_min = (circle_max[0] + interval, 0)
line_max = (line_min[0] + line_width, circle_max[1])

window_name = "demo"
window = cv2.namedWindow(window_name)
interval_space = np.zeros((2 * circle_radius, interval, 3), np.uint8)
value_bar = np.zeros((2 * circle_radius, line_width, 3), np.uint8)
im = np.zeros((0, 0, 0))
h = np.nan
s = np.nan
v = np.nan


def get_angle(coords, center):
    c = 1
    if coords[0] < 0:
        c = -1
    if coords[0] - center[0] == 0:
        return 0
    tan_a = (coords[1] - center[1]) / (coords[0] - center[0])
    angle = np.pi / 2 + np.arctan(tan_a)
    if coords[0] - center[0] < 0:
        angle += np.pi
    return angle


def get_distance(coords, center):
    distance = np.sqrt((coords[0] - center[0]) ** 2 + (coords[1] - center[1]) ** 2)
    return distance


def get_height(coords, horizont):
    return coords[1] - horizont


def draw_value_bar(image, hue, saturation, pt1, pt2):
    rect = np.zeros((pt2[0] - pt1[0], pt2[1] - pt1[1], 3), np.uint8)
    for j in range(0, rect.shape[1]):
        for i in range(0, rect.shape[0]):
            rect[i][j] = [hue, saturation, (j - line_min[1]) / line_max[1] * 255]
    rect = cv2.cvtColor(rect, cv2.COLOR_HSV2BGR)
    for i in range(pt1[0], pt2[0]):
        for j in range(pt1[1], pt2[1]):
            image[j][i] = rect[i - pt1[0]][j - pt1[1]]


def hsvCallback(event, x, y, flags, param):
    global im, h, s, v, circle_x, circle_y
    if event == cv2.EVENT_LBUTTONUP \
            and circle_min[0] <= x <= circle_max[0] \
            and circle_min[1] <= y <= circle_max[1] \
            and (((x - circle_center[0]) * (x - circle_center[0]) + (y - circle_center[1]) * (y - circle_center[1])) <= (circle_radius * circle_radius)):
                # and ((x - circle_center[0]) ** 2) + ((y - circle_center[1]) ** 2) <= circle_radius ** 2:
        im = np.concatenate((np.concatenate((img_circle, interval_space), axis=1), value_bar), axis=1)
        cv2.circle(im, (x, y), 3, (0, 0, 0), thickness=1)
        circle_x, circle_y = x, y
        h = get_angle((x, y), circle_center) / 2 / np.pi * 180
        # print("h", h)
        s = get_distance((x, y), circle_center) / circle_radius * 255
        draw_value_bar(im, h, s, line_min, line_max)
    elif h != np.nan \
            and s != np.nan \
            and event == cv2.EVENT_LBUTTONUP \
            and line_min[0] < x <= line_max[0] \
            and line_min[1] < y <= line_max[1]:
        im = np.concatenate((np.concatenate((img_circle, interval_space), axis=1), value_bar), axis=1)
        draw_value_bar(im, h, s, line_min, line_max)
        cv2.circle(im, (circle_x, circle_y), 3, (0, 0, 0), thickness=1)
        cv2.rectangle(im, (line_min[0] - 3, y - 1), (line_min[0], y + 1), (0, 0, 0), thickness=1)
        v = get_height((x, y), line_min[1]) / (line_max[1] - line_min[1]) * 255
        hsv = np.uint8([[[h, s, v]]])
        print("HSV: ", hsv)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        print("RGB: ", rgb)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)
        print("LAB: ", lab)
        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        print("YCrCb: ", ycrcb)


if __name__ == '__main__':
    interval_space.fill(255)
    for y in range(value_bar.shape[1]):
        for x in range(value_bar.shape[0]):
            value_bar[x][y] = [0, 0, (y - line_min[1]) / line_max[1] * 255]
    cv2.cvtColor(value_bar, cv2.COLOR_HSV2BGR)
    im = np.concatenate((np.concatenate((img_circle, interval_space), axis=1), value_bar), axis=1)

    cv2.setMouseCallback(window_name, hsvCallback)
    while True:
        cv2.imshow(window_name, im)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('c') or key == 27:  # ASCII ESCape
            break

    cv2.destroyAllWindows()
