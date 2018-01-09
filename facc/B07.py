import cv2
import numpy as np
#去除指定大小的区域
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.pyplot as plt
import facc.HistUtil as histUtil

def delete_jut(src,uthreshold,vthreshold,type):
    threshold = 0
    dst = src.copy()
    sp = src.shape
    height = sp[0]
    width = sp[1]
    k = 0
    #type = 0 就是消除黑色的突出部 type=1就是消除白色的突出部
    mode = (1-type)*255
    modeInver = 255-mode
    for i in range(height-1):
        for j in range(width-1):
            #行消除
            if dst[i,j]== mode and dst[i,j+1] == modeInver:
                if j+uthreshold >= width:
                    for k in range(j+1,width):
                        dst[i,k] = mode
                else:
                    for k in range(j+2,j+uthreshold+1):
                        if dst[i,k] == mode:
                            break
                    if dst[i,k]  == mode:
                        for h in range(j+1,k):
                            dst[i,h] = mode
            #列消除
            if dst[i,j] == mode and dst[i+1,j] == modeInver:
                if i+vthreshold >= height:
                    for k in range(i+1,height):
                        dst[k,j] = mode
                else:

                    for k in range(i+2,i+vthreshold+1):
                        if dst[k,j] == mode:
                            break
                    if dst[k,j] == mode:
                        for h in range(i+1,k):
                            dst[h,j] = mode
    #根据需求，强制性去除上下四行
    for i in range(2):
        for j in range(width):
            dst[i,j] = mode;
    for i in range(height-1,height-3,-1):
        for j in range(width):
            dst[i,j] = mode;
    return dst
if __name__ == '__main__':
    img = cv2.imread("B07/B07-02.bmp",0)
    ROI = img[445:635,560:1050]
    #rows from 445 to 635 ,cols from 560 to 1050
    cv2.imshow("ROI",ROI)

    #转换成二值图，方便获取轮廓
    ret,ROIbin = cv2.threshold(ROI,100,255,cv2.THRESH_BINARY)
    cv2.imshow("ROIbin",ROIbin)

    result, contours, hierarchy = cv2.findContours(ROIbin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    result = np.zeros(ROI.shape, np.uint8)
    cv2.drawContours(result, contours, -1, 255)
    cv2.imshow("contours", result)
    for i in range(len(contours)):
        rect = cv2.boundingRect(contours[i])
        #先判断是圆还是方
        if math.fabs(rect[2]-rect[3])>8:#是长方形，进行去毛刺处理
            temp = ROIbin[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
            temp = delete_jut(temp,int(temp.shape[1]*1/5),int(temp.shape[0]*3/5),1)
            ROIbin[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]=temp;
    cv2.imshow("newROI",ROIbin)

    cv2.waitKey(0)