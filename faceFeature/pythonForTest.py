import cv2
import math

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import faceFeature.newThresholdMethod as newOne
import faceFeature.HistUtil as histUtil

def DrawHist(hist, color):
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    histImg = np.zeros([512,256], np.uint8)

    hpt = int(0.9* 256);
    #cv2.line(histImg, (0, 256), (256, 256), [0,0,255])
    for h in range(256):
            intensity = int(hist[h]*hpt/maxVal)
            if intensity!=0:
                cv2.line(histImg,(h,256), (h,256-intensity), color)
    return histImg;

def fourier(hist,Mode,alterComponent = 0,radio = 0):
    ft = hist.copy()
    N = 256
    Cu = 1
    #高通滤波
    if Mode == 0:
        for u in range(N):
            if u == 0:
                Cu= 1.0/math.sqrt(2)
            sum = 0
            for x in range(N):
                theta = ((2*x)+1)*u*math.pi/(2*N)
                if theta == 0 or u<alterComponent:
                    sum += (hist[x]*math.cos(theta))
            temp = Cu*math.sqrt(2.0/N)*sum
            ft[u] = int(temp)
    else: #低通滤波
        sum_direct = 0
        for u in range(N):
            if u == 0:
                Cu= 1.0/math.sqrt(2)
            sum = 0
            for x in range(N):
                theta = ((2*x)+1)*u*math.pi/(2*N)
                if theta == 0:
                    sum_direct+=(hist[x]*math.cos(theta))
                sum += (hist[x]*math.cos(theta))
            temp = Cu*math.sqrt(2.0/N)*sum
            ft[u] = int(temp)
        sum_direct = Cu*math.sqrt(2.0/N)*sum_direct
        for u in range(N):
            if ft[u]>sum_direct*radio:
                ft[u]=0
    return ft





if __name__ == '__main__':
    myfont = matplotlib.font_manager.FontProperties(fname='c:\\windows\\fonts\\simfang.ttf')
    img = cv2.imread("skeleton/src1.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(231) ,plt.title('原图',fontproperties=myfont)
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])
    img = cv2.imread("skeleton/g1.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(232),plt.title('肤色概率图',fontproperties=myfont)
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])
    img = cv2.imread("skeleton/b1.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(233),plt.title('肤色图',fontproperties=myfont)
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])
    img = cv2.imread("skeleton/src2.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(234),plt.title('原图',fontproperties=myfont)
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])
    img = cv2.imread("skeleton/g2.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(235),plt.title('肤色概率图',fontproperties=myfont)
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])
    img = cv2.imread("skeleton/b2.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(236),plt.title('肤色图',fontproperties=myfont)
    plt.imshow(img)
    plt.xticks([]), plt.yticks([])
    #
    # img = cv2.imread("skeleton/52.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.subplot(245)
    # plt.imshow(img)
    # plt.xticks([]), plt.yticks([])
    # img = cv2.imread("skeleton/52_.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.subplot(246)
    # plt.imshow(img)
    # plt.xticks([]), plt.yticks([])
    # img = cv2.imread("skeleton/41.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.subplot(247)
    # plt.imshow(img)
    # plt.xticks([]), plt.yticks([])
    # img = cv2.imread("skeleton/41_.jpg")
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.subplot(248)
    # plt.imshow(img)
    # plt.xticks([]), plt.yticks([])
    plt.show()


    # img_man = cv2.imread('skeleton/5.png', 0)  # 直接读为灰度图像
    #
    #
    # img_man = cv2.imread('skeleton/test.jpg', 0)  # 直接读为灰度图像
    # plt.subplot(232), plt.imshow(img_man, 'gray'), plt.title('原图',fontproperties=myfont)
    # plt.xticks([]), plt.yticks([])
    # # --------------------------------
    # rows, cols = img_man.shape
    # mask = np.zeros(img_man.shape, np.uint8)
    # #mask[int(rows / 2) - 30:int(rows / 2) + 30, int(cols / 2) - 30:int(cols / 2) + 30] = 1
    # for x in range(rows):
    #     for y in range(cols):
    #         if np.sqrt(np.power(x-rows/2,2)+np.power(y-cols/2,2))<50-5:
    #             mask[x,y]=255
    #         elif np.sqrt(np.power(x-rows/2,2)+np.power(y-cols/2,2))>50+5:
    #             mask[x,y]=255
    #         else:
    #             mask[x,y]=0
    # # --------------------------------
    # plt.subplot(231), plt.imshow(mask, 'gray'), plt.title('理想带阻滤波器', fontproperties=myfont)
    # plt.xticks([]), plt.yticks([])
    # f1 = np.fft.fft2(img_man)
    # f1shift = np.fft.fftshift(f1)
    # f1shift = f1shift * mask
    # f2shift = np.fft.ifftshift(f1shift)  # 对新的进行逆变换
    # img_new = np.fft.ifft2(f2shift)
    # # 出来的是复数，无法显示
    # img_new = np.abs(img_new)
    # # 调整大小范围便于显示
    # img_new = (img_new - np.amin(img_new)) / (np.amax(img_new) - np.amin(img_new))
    # plt.subplot(233), plt.imshow(img_new, 'gray'), plt.title('理想带阻滤波结果图',fontproperties=myfont)
    # plt.xticks([]), plt.yticks([])
    #
    # img_man = cv2.imread('skeleton/test.jpg', 0)  # 直接读为灰度图像
    # plt.subplot(235), plt.imshow(img_man, 'gray'), plt.title('原图', fontproperties=myfont)
    # plt.xticks([]), plt.yticks([])
    # # --------------------------------
    # rows, cols = img_man.shape
    # mask = np.zeros(img_man.shape, np.uint8)
    # # mask[int(rows / 2) - 30:int(rows / 2) + 30, int(cols / 2) - 30:int(cols / 2) + 30] = 1
    # for x in range(rows):
    #     for y in range(cols):
    #         if np.sqrt(np.power(x - rows / 2, 2) + np.power(y - cols / 2, 2)) < 50 - 5:
    #             mask[x, y] = 0
    #         elif np.sqrt(np.power(x - rows / 2, 2) + np.power(y - cols / 2, 2)) > 50 + 5:
    #             mask[x, y] = 0
    #         else:
    #             mask[x, y] = 255
    # # --------------------------------
    # plt.subplot(234), plt.imshow(mask, 'gray'), plt.title('理想带通滤波器', fontproperties=myfont)
    # plt.xticks([]), plt.yticks([])
    # f1 = np.fft.fft2(img_man)
    # f1shift = np.fft.fftshift(f1)
    # f1shift = f1shift * mask
    # f2shift = np.fft.ifftshift(f1shift)  # 对新的进行逆变换
    # img_new = np.fft.ifft2(f2shift)
    # # 出来的是复数，无法显示
    # img_new = np.abs(img_new)
    # # 调整大小范围便于显示
    # img_new = (img_new - np.amin(img_new)) / (np.amax(img_new) - np.amin(img_new))
    # plt.subplot(236), plt.imshow(img_new, 'gray'), plt.title('理想带通滤波结果图', fontproperties=myfont)
    # plt.xticks([]), plt.yticks([])
    # plt.show()


