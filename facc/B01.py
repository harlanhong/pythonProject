import cv2
import numpy as np
#去除指定大小的区域
from mpl_toolkits.mplot3d import Axes3D
import math
import matplotlib.pyplot as plt
import facc.HistUtil as histUtil
#用来强度分层
def intensitySlicing(img,i):
    cv2.imshow(str(i),img)
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

def RemoveSelectRegion(src, AreaHigh, AreaLow, CheckMode, NeiborMode):
    RemoveCount = 0
    # 新建一幅标签图像初始化为0像素点，为了记录每个像素点检验状态的标签，0代表未检查，1代表正在检查, 2代表检查不合格（需要反转颜色），3代表检查合格或不需检查
    # 初始化的图像全部为0，未检查
    PointLabel = np.zeros(src.shape, np.uint8)
    dst = np.zeros(src.shape, np.uint8)
    sp = src.shape
    if (CheckMode == 1):  # 去除小连通区域的白色点,除去白色的
        print("去除小连通域")
        for i in range(sp[0]):
            for j in range(sp[1]):
                if src[i, j] < 10:
                    PointLabel[i, j] = 3  # 将背景黑色点标记为合格，像素为3
    else:  # 除去黑色的
        print("去除孔洞")
        for i in range(sp[0]):
            for j in range(sp[1]):
                if src[i, j] > 10:
                    PointLabel[i, j] = 3  # 如果原图是白色区域，标记为合格，像素为3
    NeiborPos = []  # 将邻域压进容器
    NeiborPos.append((-1, 0))
    NeiborPos.append((1, 0))
    NeiborPos.append((0, -1))
    NeiborPos.append((0, 1))
    if NeiborMode == 1:
        print("Neighbor mode: 8邻域.")
        NeiborPos.append((-1, -1))
        NeiborPos.append((-1, 1))
        NeiborPos.append((1, -1))
        NeiborPos.append((1, 1))
    else:
        print("Neighbor mode: 4邻域.")
    NeihborCount = 4 + 4 * NeiborMode;
    CurrX = 0
    CurrY = 0
    for i in range(sp[0]):
        for j in range(sp[1]):
            if PointLabel[i, j] == 0:
                GrowBuffer = []
                GrowBuffer.append((i, j))
                PointLabel[i, j] = 1
                CheckResult = 0
                # 在这里说一下，python的for循环有点奇葩，就是范围是静态的，第一次获取到范围值后就不会改变了
                z = 0
                while z < len(GrowBuffer):
                    for q in range(NeihborCount):
                        CurrX = GrowBuffer[z][0] + NeiborPos[q][0]
                        CurrY = GrowBuffer[z][1] + NeiborPos[q][1]
                        if CurrX >= 0 and CurrX < sp[0] and CurrY >= 0 and CurrY < sp[1]:
                            if PointLabel[CurrX, CurrY] == 0:
                                GrowBuffer.append((CurrX, CurrY))
                                PointLabel[CurrX, CurrY] = 1
                    z += 1
                # 对整个连通域检查完
                if len(GrowBuffer) > AreaHigh or len(GrowBuffer) < AreaLow:
                    CheckResult = 2
                else:
                    CheckResult = 1
                    RemoveCount += 1

                for z in range(len(GrowBuffer)):
                    CurrX = GrowBuffer[z][0]
                    CurrY = GrowBuffer[z][1]
                    PointLabel[CurrX, CurrY] += CheckResult
    CheckMode = 255 * (1 - CheckMode)
    for i in range(sp[0]):
        for j in range(sp[1]):
            if PointLabel[i, j] == 2:
                dst[i, j] = CheckMode
            if PointLabel[i, j] == 3:
                dst[i, j] = src[i, j]

    return dst
#去除二值图像边缘的突出部
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

def inching(img,rect):
    x=0
    y=0
    height=0
    width =0
    #先对左列进行扫描
    #左列扫描区域
    left = img[rect[1]:rect[1]+rect[3],rect[0]-5:rect[0]+5]
    right = img[rect[1]:rect[1]+rect[3],rect[0]+rect[2]-5:rect[0]+rect[2]+5]
    up = img[rect[1]-3:rect[1]+3,rect[0]:rect[0]+rect[2]]
    down = img[rect[1]+rect[3]-3:rect[1]+rect[3]+3,rect[0]:rect[0]+rect[2]]
    anchor = (-1, -1);
    delta = 0;
    #自定义卷积核
    # kernel = np.zeros((1,3))
    # kernel[0,0]=-1;kernel[0,1]=1;kernel[0,2]=0
    # dst = cv2.filter2D(left,-1,kernel,anchor=anchor,delta=delta,borderType=cv2.BORDER_DEFAULT)
    #直接寻找靠近255一半的行或列
    #对列
    cols = np.sum(left, axis=0)
    cols = cols/left.shape[0]
    for i in range(len(cols)):
        if cols[i]>=200:
            x = i-5+rect[0]
            print(i)
            break;
    cols = np.sum(right, axis=0)
    cols = cols/right.shape[0]
    for i in range(len(cols)-1,-1,-1):
        if cols[i] >= 200:
            width = i - 5 + rect[0]+rect[2] -x
            print(i)
            break;
    rows = np.sum(up, axis=1)
    rows = rows/up.shape[1]
    for i in range(len(rows)):
        if rows[i] >= 200:
            y = i-3+rect[1]
            break;
    rows = np.sum(down, axis=1)
    rows = rows/down.shape[1]
    for i in range(len(rows)-1,-1,-1):
        if rows[i] >= 200:
            height = i-3+rect[1]+rect[3]-y;
            break;
    newRect = (x,y,width,height)
    return newRect

def avgGray(img):
    sp = img.shape
    sum = 0
    for i in range(sp[0]):
        for j in range(sp[1]):
            sum+=img[i,j]
    return sum/(sp[0]*sp[1])
def process(input):
    #进行截取是为了降低时间和去噪
    ROIBGR = input[143:891, 393:1239]
    #为了方便，我们新建一个与input一样大小的图片;
    cv2.imshow("ROIsrc",ROIBGR)
    ROIgray = cv2.cvtColor(ROIBGR, cv2.COLOR_BGR2GRAY)
    #强度分层
    #intensitySlicing(ROIgray)
    #ROIgray = cv2.medianBlur(ROIgray,3)
    #因为精度要求很高，所以一切改变图像形态的操作都不能进行
    ret, ROI = cv2.threshold(ROIgray, 100, 255, cv2.THRESH_BINARY)
    #ROI = RemoveSelectRegion(ROI, 500, 0, 0, 1)
    cv2.imshow("roiBinary",ROI)
    #第一次获取轮廓，用来矫正矩形
    result, contours, hierarchy = cv2.findContours(ROI, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros(ROI.shape, np.uint8)
    cv2.drawContours(result,contours,-1,255)
    cv2.imshow("contours",result)
    for i in range(len(contours)):
        rect = cv2.boundingRect(contours[i])
        #先判断是圆还是方
        if math.fabs(rect[2]-rect[3])>8:#是长方形，进行去毛刺处理
            temp = ROI[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
            temp = delete_jut(temp,int(temp.shape[1]*2/5),int(temp.shape[0]*2/5),1)
            ROI[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]=temp;
    cv2.imshow("newROI",ROI)
    # 第二次获取轮廓，用来确定中心点
    result, contours, hierarchy = cv2.findContours(ROI, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros(ROI.shape, np.uint8)
    cv2.drawContours(result, contours, -1, 255)
    cv2.imshow("contours2", result)
    colorlist = [(255,0,0),(0,255,0),(0,0,255),(0,0,0)]
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) < 100:
            continue
        rect = cv2.boundingRect(contours[i])
        cv2.rectangle(input, (rect[0]+393,rect[1]+143), (rect[0]+rect[2]+393,rect[1]+rect[3]+143), (0,0,255))
        if math.fabs(rect[2] - rect[3]) > 8:
            rect = inching(ROIgray,rect)
        #intensitySlicing(ROIgray[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]],i)
        cv2.rectangle(input, (rect[0]+393,rect[1]+143), (rect[0]+rect[2]+393,rect[1]+rect[3]+143), (0,255,0))
        cX = int(rect[0]+rect[2]/2) + 393
        cY = int(rect[1]+rect[3]/2) + 143
        cv2.line(input, (cX, cY - 10), (cX, cY + 10), colorlist[i])
        cv2.line(input, (cX - 10, cY), (cX + 10, cY), colorlist[i])

def unitTest():
    input = cv2.imread("B01/B01-0" + str(8) + ".bmp", 1)
    process(input)
    cv2.imshow("result",input)

def allTest():
    i = 1
    while i<10:
        print(i)
        input = cv2.imread("B01/B01-0" + str(i) + ".bmp", 1)
        process(input)
        cv2.imwrite("result/B01/B01-0" + str(i) + ".bmp",input)
        i+=1
if __name__ == '__main__':
    unitTest()
    cv2.waitKey(0)







