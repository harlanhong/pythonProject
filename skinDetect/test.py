import cv2
import matplotlib
import numpy as np
import math

import skimage
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#OSTU大津法
def GetOSTUThreshold(histGram):
    X, Y, Amount = 0,0,0;
    MinValue, MaxValue=0,0;
    Threshold = 0;
    for MinValue in range(256):
        if histGram[MinValue]!=0:
            break;
    for MaxValue in range(255,-1,-1):
        if histGram[MaxValue]!=0:
            break;
    if MaxValue == MinValue: return MaxValue; # 图像中只有一个颜色
    if MinValue + 1 == MaxValue: return MinValue; # 图像中只有二个颜色
    for Y in range(MinValue,MaxValue+1):Amount += histGram[Y];#像素个数
    PixelIntegral = 0;
    for Y in range(MinValue, MaxValue + 1): PixelIntegral += histGram[Y] * Y;#像素总值
    SigmaB = -1;
    PixelBack = 0
    PixelIntegralBack=0
    for Y in range(MinValue, MaxValue + 1):
        PixelBack = PixelBack + histGram[Y];
        PixelFore = Amount - PixelBack;
        OmegaBack = float(PixelBack / Amount);
        OmegaFore = float(PixelFore / Amount);
        PixelIntegralBack += histGram[Y] * Y;
        PixelIntegralFore = PixelIntegral - PixelIntegralBack;
        MicroBack = float(PixelIntegralBack / PixelBack);
        MicroFore = float(PixelIntegralFore / PixelFore);
        Sigma = OmegaBack * OmegaFore * (MicroBack - MicroFore) * (MicroBack - MicroFore);
        if (Sigma > SigmaB):
            SigmaB = Sigma;
            Threshold = Y;
    return Threshold;
#手动画出直方图和阈值
def calcAndDrawHist(image, color):
    hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)
    histImg = np.zeros([256, 256, 3], np.uint8)
    hpt = int(0.9 * 256);
    for h in range(256):
        intensity = int(hist[h] * hpt / maxVal)
        cv2.line(histImg, (h, 256), (h, 256 - intensity), color)
    thresh = GetOSTUThreshold(hist)
    cv2.line(histImg, (thresh, 255), (thresh,0), [0,0,255])
    return thresh,histImg;
if __name__ == '__main__':
    img = cv2.imread('2.3/11.jpg', 0)  # 直接读为灰度图像
    thresh,histimg = calcAndDrawHist(img,[255,255,255])
    cv2.imshow("HistByHand",histimg)
    cv2.imwrite("2.3/11_HistByHand.jpg",histimg)
    cv2.imshow("lena",img)
    cv2.imwrite("2.3/11_Gray.jpg",img)

    ret,result = cv2.threshold(img,thresh,255,cv2.THRESH_BINARY)
    cv2.imshow("binary",result)
    cv2.imwrite("2.3/11_binary.jpg",result)

    #使用python画出直方图
    plt.figure("7_HistByPython")
    arr = img.flatten()
    n, bins, patches = plt.hist(arr, bins=256, normed=1, facecolor='green', alpha=0.75)
    plt.show()

    cv2.waitKey(0)
