import cv2 as cv
import numpy as np


class Form:
    def __init__(self, img, start_img, name):
        x, y = np.where(img[:, :, 0] != 0)
        self.img = cv.resize(img, (1000, 1000))
        self.start_img = start_img
        self.name = name
        self.data = []

    def get_image(self):
        return self.img

    def detection(self, src):
        orb = cv.ORB_create()
        a1, b1 = orb.detectAndCompute(src, None)
        a2, b2 = orb.detectAndCompute(self.start_img, None)

        bf_matcher = cv.BFMatcher()
        matches = bf_matcher.knnMatch(b1, b2, k=2)

        need_matches = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                need_matches.append([m])

        return len(need_matches)

    def get_size(self):
        return min(self.img[:, 0, 0].size, self.img[0, :, 0].size), max(self.img[:, 0, 0].size, self.img[0, :, 0].size)

    def get_map(self, eps: int):
        height = 1000 // eps
        width = 1000 // eps

        map = np.zeros((height, width))

        for i in range(0, height):
            for j in range(0, width):
                if len(np.where(self.img[eps * i: eps * (i + 1), eps * j: eps * (j + 1)] != 0)[0]) != 0:
                    map[i, j] = 255
                else:
                    map[i, j] = 0

        return map


def list_paper(img):
    mask = get_mask(img)
    mask = mask[:, :, 0]

    x, y = np.where(mask != 0)

    up = min(x)
    bottom = max(x)
    left = min(y)
    right = max(y)
    img = img[up:bottom, left:right]

    return img


def only_mask(src):
    height = src[:, 0, 0].size
    width = src[0, :, 0].size

    return src[30: height - 30, 30: width - 30, :]


def get_mask(image):
    cvt = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.medianBlur(cvt, 19)
    cv_canny = cv.Canny(blur, 0, 5)
    outline, hierarchy = cv.findContours(cv_canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    pap_list = \
        sorted(outline, key=lambda tup: max(tup[:, 0, 0]) + max(tup[:, 0, 1]) - min(tup[:, 0, 0]) - min(tup[:, 0, 1]),
               reverse=True)[0]
    max(pap_list[:, 0, 0])
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv.drawContours(image=mask, contours=[pap_list], color=(255, 255, 255), thickness=cv.FILLED, contourIdx=0)

    return mask


def get_objects():
    photos = []
    for i in range(1, 10):
        photos.append('photo' + str(i))

    elements = {}
    for name in photos:
        start_img = cv.imread('./photos/objects/' + name + '.jpg')
        mask = get_mask(start_img)
        mask = mask[:, :, 0]

        x, y = np.where(mask != 0)
        up = min(x)
        bottom = max(x)
        left = min(y)
        right = max(y)
        start_img = start_img[up:bottom, left:right]
        height = start_img[:, 0, 0].size
        width = start_img[0, :, 0].size
        mask = start_img[30: height - 30, 30: width - 30, :]
        detected_edges = cv.GaussianBlur(src=mask, ksize=(9, 9), sigmaX=10, dst=50)
        canny = cv.Canny(detected_edges, 10, 100, apertureSize=3)
        outline, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_KCOS)

        mask = np.zeros(start_img.shape, dtype=np.uint8)
        cv.drawContours(image=mask, contours=outline, color=(255, 255, 255), thickness=cv.FILLED, contourIdx=-1)
        elements[name] = Form(img=mask, start_img=start_img, name=name)

    return elements


def detection(img):
    paper = list_paper(img)
    paper = only_mask(paper)
    canny = cv.Canny(paper, 150, 255)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (35, 35))
    closed = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel)

    contours = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    cv.drawContours(closed, contours, -1, (255, 255, 255), -1)

    result = []
    for contour in contours:
        image = np.zeros(closed.shape, dtype=np.uint8)
        cv.drawContours(image, [contour], -1, (255, 255, 255), -1)
        x, y = np.where(image != 0)
        image = paper[min(x): max(x), min(y): max(y)]
        result.append(image)

    return result
