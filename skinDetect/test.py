import cv2
import matplotlib
import numpy as np
import math

import skimage
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == '__main__':
    # img_man = cv2.imread('woman.jpg', 0)  # 直接读为灰度图像
    # plt.subplot(121), plt.imshow(img_man, 'gray'), plt.title('origial')
    # plt.xticks([]), plt.yticks([])
    # # --------------------------------
    # rows, cols = img_man.shape
    # mask = np.ones(img_man.shape, np.uint8)
    # mask[rows / 2 - 30:rows / 2 + 30, cols / 2 - 30:cols / 2 + 30] = 0
    # # --------------------------------
    # f1 = np.fft.fft2(img_man)
    # f1shift = np.fft.fftshift(f1)
    # f1shift = f1shift * mask
    # f2shift = np.fft.ifftshift(f1shift)  # 对新的进行逆变换
    # img_new = np.fft.ifft2(f2shift)
    # # 出来的是复数，无法显示
    # img_new = np.abs(img_new)
    # # 调整大小范围便于显示
    # img_new = (img_new - np.amin(img_new)) / (np.amax(img_new) - np.amin(img_new))
    # plt.subplot(122), plt.imshow(img_new, 'gray'), plt.title('Highpass')
    # plt.xticks([]), plt.yticks([])
    myfont = matplotlib.font_manager.FontProperties(fname='c:\\windows\\fonts\\simfang.ttf')

    img = cv2.imread("2.2/8.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(221), plt.title('原图', fontproperties=myfont)
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])

    img = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(222), plt.title('核结构', fontproperties=myfont)
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])

    img = cv2.imread("2.2/dilate.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(223), plt.title('膨胀例图', fontproperties=myfont)
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])

    img = cv2.imread("2.2/erode.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(224), plt.title('腐蚀例图', fontproperties=myfont)
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])



    plt.show()
   #
   #  img = cv2.imread("2.3/1.jpg")
   #  dst = cv2.GaussianBlur(img,ksize=(5, 5), sigmaX=1.0,sigmaY=1.0)
   #  cv2.imshow("dst",dst)
   #  cv2.imwrite("2.3/2.jpg",dst)
   #  cv2.imshow("img",img)
   #  cv2.waitKey(0)