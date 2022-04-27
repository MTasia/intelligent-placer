import cv2 as cv
import numpy as np
from copy import copy
from form import get_objects, detection


def try_insert(x_start: int, y_start: int, list: np.array, img: np.array):
    height = img[0, :].size
    width = img[:, 0].size

    list_copy = copy(list)

    for y in range(y_start, y_start + height):
        for x in range(x_start, x_start + width):
            if list_copy[x, y] == 0 and img[x - x_start, y - y_start] != 0:
                return [], False
            if list_copy[x, y] != 0 and img[x - x_start, y - y_start] != 0:
                list_copy[x, y] = 0

    return list_copy, True


def find(elements: np.array, list: np.array, x_start: int, y_start: int):
    if len(elements) == 0:
        return True

    height = list[0, :].size
    width = list[:, 0].size

    img_height = elements[0][0, :].size
    img_width = elements[0][:, 0].size

    while height - y_start > img_height or width - x_start > img_width:
        new_list, result = try_insert(x_start, y_start, list, elements[0])
        if not result:
            if width - x_start > img_width:
                x_start += 1
            else:
                x_start = 0
                y_start += 1
        else:
            result = find(elements[1:], new_list, 0, 0)
            if result:
                return True
            else:
                if width - x_start > img_width:
                    x_start += 1
                else:
                    y_start += 1

    return False


def algorithm(elements: np.array, poly: np.array, size):
    list = np.zeros((size[0], size[1]))
    cv.drawContours(list, [poly], -1, 255, -1)

    x, y = np.where(list != 0)
    list = list[min(x): max(x), min(y): max(y)]

    return find(elements, list, 0, 0)


def get_mask(image):
    src = image
    gr = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    bl = cv.medianBlur(gr, 19)
    canny = cv.Canny(bl, 0, 5)

    contours, hierarchy = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    pap_list = \
        sorted(contours, key=lambda tup: max(tup[:, 0, 0]) + max(tup[:, 0, 1]) - min(tup[:, 0, 0]) - min(tup[:, 0, 1]),
               reverse=True)[0]

    max(pap_list[:, 0, 0])
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv.drawContours(image=mask, contours=[pap_list], color=(255, 255, 255), thickness=cv.FILLED, contourIdx=0)

    return mask


def only_mask(src):
    height = src[:, 0, 0].size
    width = src[0, :, 0].size

    return src[30: height - 30, 30: width - 30, :]


def intelligent_placer(img_path: str, poly: np.array):
    objects = get_objects()
    need = detection(cv.imread(img_path))

    meets = {}
    for elem in objects:
        meet = []
        for item in need:
            if item.size != 0 and item[:, 0, 0].size * item[0, :, 0].size >= 8000:
                meet.append(objects[elem].detection(item))
        meets[elem] = meet

    needs = []
    for i in range(len(meets[next(iter(meets))])):
        need = next(iter(meets))
        for meet in meets:
            if meets[meet][i] > meets[need][i]:
                need = meet
        if meets[need][i] != 0:
            meets.pop(need)
            needs.append(need)

    elements = []
    size = np.zeros(2, dtype=int)
    for need in needs:
        need_map = objects[need].get_map(10)
        size[0], size[1] = need_map[:, 0].size, need_map[0, :].size
        x, y = np.where(need_map != 0)
        elements.append(need_map[min(x): max(x), min(y): max(y)])

    for i in range(len(poly)):
        poly[i, 0] = int(poly[i, 0] * (size[0] / 10))
        poly[i, 1] = int(poly[i, 1] * (size[1] / 10))

    return algorithm(elements, poly, size)
