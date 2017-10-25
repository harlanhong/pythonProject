import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = cv2.imread("img/53.jpg",1)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ycrbr = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb);
    y,cr,cb = cv2.split(ycrbr);
    cv2.imshow("y",y)
    cv2.imshow("gray",gray)
    cv2.waitKey(0)
