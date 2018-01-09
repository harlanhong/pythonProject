import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    img = cv2.imread('B01/B01-01.bmp', 0)  # 直接读为灰度图像
    img = img[143:891, 393:1239]
    # 为了方便，我们新建一个与input一样大小的图片;
    sp = img.shape
    x = np.arange(sp[1])
    y = np.arange(sp[0])
    X, Y = np.meshgrid(x, y)
    Z=img
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z');
    plt.show()
cv2.waitKey(0)