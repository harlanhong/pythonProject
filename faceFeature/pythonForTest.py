import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    i = 584
    while i <= 1270:
        print(i)
        img = cv2.imread("C:\\Users\Harlan\Downloads\\result\\result\\1 (" + str(i) + ")_New.jpg", 0)
        img = cv2.medianBlur(img,3);
        cv2.imshow("hah",img)
        cv2.imwrite("C:\\Users\\Harlan\\Downloads\\result\\result\\1 (" + str(i) + ")_New-1.jpg", img)
        i = i + 1
