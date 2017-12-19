import cv2
import math
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
   t = math.floor(3.455)
   print(type(t))



