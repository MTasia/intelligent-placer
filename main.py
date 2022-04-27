import numpy as np
from int_placer import intelligent_placer


def get_poly(path):
    with open(path) as f:
        n = int(f.readline())
        poly = []
        for i in range(n):
            x, y = f.readline().split(' ')
            poly.append([int(x), int(y)])
    return poly


def print_ans(answer):
    if answer:
        print("yes")
    else:
        print("no")


if __name__ == '__main__':
    for i in range(1, 2):
        img_path = 'photos/tests/test' + str(i) + '.jpg'
        poly_point = get_poly('photos/tests/poly' + str(i) + '.txt')
        poly = np.array(poly_point)
        ans = intelligent_placer(img_path, poly)
        print("test number " + str(i))
        print_ans(ans)

# print(intelligent_placer('photos/tests/test1.jpg', np.array([[0, 10], [10, 10], [10, 0]])))
# print(intelligent_placer('photos/tests/test2.jpg', np.array([[0, 0], [0, 2], [2, 2], [2, 0]])))
# print(intelligent_placer('photos/tests/test4.jpg', np.array([[3, 10], [0, 4], [5, 0], [10, 4], [8, 10]])))
# print(intelligent_placer('photos/tests/test5.jpg', np.array([[3, 10], [0, 4], [5, 0], [10, 4], [8, 10]])))
