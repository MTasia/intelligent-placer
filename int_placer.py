from copy import copy
import cv2 as cv
import numpy as np


class Form:
    def __init__(self, name_def, img_def, start_img_def):
        self.name = name_def
        self.img = cv.resize(img_def, (1000, 1000))
        self.data = []
        self.start_img = start_img_def
        x, y = np.where(img_def[:, :, 0] != 0)

    def get_map(self, e: int):
        width_map = 1000 // e
        height_map = 1000 // e

        need_map = np.zeros((height_map, width_map))
        for i in range(0, height_map):
            for j in range(0, width_map):
                if len(np.where(self.img[e * i: e * (i + 1), e * j: e * (j + 1)] != 0)[0]) == 0:
                    need_map[i, j] = 0
                else:
                    need_map[i, j] = 255

        return need_map

    def detection(self, src):
        orb = cv.ORB_create()
        a1, b1 = orb.detectAndCompute(src, None)
        a2, b2 = orb.detectAndCompute(self.start_img, None)
        matches = cv.BFMatcher().knnMatch(b1, b2, k=2)

        need_matches = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                need_matches.append([m])

        return len(need_matches)

    def get_image(self):
        return self.img

    def get_size(self):
        return min(self.img[:, 0, 0].size, self.img[0, :, 0].size), max(self.img[:, 0, 0].size, self.img[0, :, 0].size)


def list_paper(image):
    mask = get_mask(image)[:, :, 0]
    x, y = np.where(mask != 0)
    u = min(x)
    r = max(y)
    b = max(x)
    l = min(y)
    image = image[u:b, l:r]
    return image


def get_mask(image):
    line, post = cv.findContours(cv.Canny(cv.medianBlur(cv.cvtColor(image, cv.COLOR_BGR2GRAY), 19), 0, 5),
                                         cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    need_list = \
    sorted(line, key=lambda tup: max(tup[:, 0, 0]) + max(tup[:, 0, 1]) - min(tup[:, 0, 0]) - min(tup[:, 0, 1]),
           reverse=True)[0]
    max(need_list[:, 0, 0])
    need_mask = np.zeros(image.shape, dtype=np.uint8)
    cv.drawContours(image=need_mask, contours=[need_list], color=(255, 255, 255), thickness=cv.FILLED, contourIdx=0)
    return need_mask


def get_mask(image):
    line, post = cv.findContours(cv.Canny(cv.medianBlur(cv.cvtColor(image, cv.COLOR_BGR2GRAY), 19), 0, 5),
                                         cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    need_list = \
    sorted(line, key=lambda tup: max(tup[:, 0, 0]) + max(tup[:, 0, 1]) - min(tup[:, 0, 0]) - min(tup[:, 0, 1]),
           reverse=True)[0]
    max(need_list[:, 0, 0])
    need_mask = np.zeros(image.shape, dtype=np.uint8)
    cv.drawContours(image=need_mask, contours=[need_list], color=(255, 255, 255), thickness=cv.FILLED, contourIdx=0)
    return need_mask


def get_objects():
    photos = []
    for i in range(1, 10):
        photos.append('photo' + str(i))
    elements = {}
    for name in photos:
        start_img = cv.imread('./photos/objects/' + name + '.jpg')
        need_mask = get_mask(start_img)[:, :, 0]
        x, y = np.where(need_mask != 0)
        u = min(x)
        r = max(y)
        b = max(x)
        l = min(y)
        start_img = start_img[u:b, l:r]
        need_mask = start_img[30: start_img[:, 0, 0].size - 30, 30: start_img[0, :, 0].size - 30, :]
        line, post = cv.findContours(cv.Canny(cv.GaussianBlur(src=need_mask, ksize=(9, 9), sigmaX=10, dst=50), 10, 100, apertureSize=3), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS)
        need_mask = np.zeros(start_img.shape, dtype=np.uint8)
        cv.drawContours(image=need_mask, contours=line, color=(255, 255, 255), thickness=cv.FILLED, contourIdx=-1)
        elements[name] = Form(name_def=name, img_def=need_mask, start_img_def=start_img)
    return elements


def detection(imfge):
    paper = list_paper(imfge)[30: list_paper(imfge)[:, 0, 0].size - 30, 30: list_paper(imfge)[0, :, 0].size - 30, :]
    ker = cv.getStructuringElement(cv.MORPH_RECT, (35, 35))
    can = cv.Canny(paper, 150, 255)
    to_close = cv.morphologyEx(can, cv.MORPH_CLOSE, ker)
    need_cont = cv.findContours(to_close, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    cv.drawContours(to_close, need_cont, -1, (255, 255, 255), -1)
    res = []
    for cont in need_cont:
        image = np.zeros(to_close.shape, dtype=np.uint8)
        cv.drawContours(image, [cont], -1, (255, 255, 255), -1)
        x, y = np.where(image != 0)
        image = paper[min(x): max(x), min(y): max(y)]
        res.append(image)
    return res


def try_insert(st_x: int, st_y: int, list: np.array, img: np.array):
    copies = copy(list)
    for y in range(st_y, st_y + img[0, :].size):
        for x in range(st_x, st_x + img[:, 0].size):
            if copies[x, y] == 0 and img[x - st_x, y - st_y] != 0:
                return [], False
            if copies[x, y] != 0 and img[x - st_x, y - st_y] != 0:
                copies[x, y] = 0
    return copies, True


def find(elements: np.array, list: np.array, st_x: int, st_y: int):
    if len(elements) == 0:
        return True
    while list[0, :].size - st_y > elements[0][0, :].size or list[:, 0].size - st_x > elements[0][:, 0].size:
        needs, res = try_insert(st_x, st_y, list, elements[0])
        if res:
            res = find(elements[1:], needs, 0, 0)
            if res:
                return True
            else:
                if list[:, 0].size - st_x > elements[0][:, 0].size:
                    st_x += 1
                else:
                    st_y += 1
        else:
            if list[:, 0].size - st_x > elements[0][:, 0].size:
                st_x += 1
            else:
                st_x = 0
                st_y += 1
    return False


def algorithm(elements: np.array, poly: np.array, size):
    llist = np.zeros((size[0], size[1]))
    cv.drawContours(llist, [poly], -1, 255, -1)
    x, y = np.where(llist != 0)
    llist = llist[min(x): max(x), min(y): max(y)]
    return find(elements, llist, 0, 0)


def intelligent_placer(img_path: str, poly: np.array):
    objects = get_objects()
    meets = {}
    for elem in objects:
        meet = []
        for i in detection(cv.imread(img_path)):
            if i.size != 0 and i[:, 0, 0].size * i[0, :, 0].size >= 8000:
                meet.append(objects[elem].detection(i))
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
    for i in needs:
        need_map = objects[i].get_map(10)
        size[0], size[1] = need_map[:, 0].size, need_map[0, :].size
        x, y = np.where(need_map != 0)
        elements.append(need_map[min(x): max(x), min(y): max(y)])
    for i in range(len(poly)):
        poly[i, 0] = int(poly[i, 0] * (size[0] / 10))
        poly[i, 1] = int(poly[i, 1] * (size[1] / 10))

    return algorithm(elements, poly, size)
