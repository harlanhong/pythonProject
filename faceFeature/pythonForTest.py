import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv2.imread("img/15.jpg",0)
    edges = cv2.Canny(img,30,90)
    cv2.imshow("img",img)
    cv2.imshow("edges",edges)
    cv2.waitKey(0)

    cv2.waitKey(0)
