#encoding:utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt
pixl_threshold = 128;

def grayWorld(img):
    pc = img.shape
    avgR = 0
    avgB = 0
    avgG = 0
    for i in range(pc[0]):
        for j in range(pc[1]):
            avgR += img[i,j,0]
            avgG += img[i,j,1]
            avgB += img[i,j,2]
    avgG = avgG/(pc[0]*pc[1])
    avgR = avgR/(pc[0]*pc[1])
    avgB = avgB/(pc[0]*pc[1])
    avgGray = (avgB+avgR+avgG)/3
    ar = avgGray/avgR
    ag = avgGray/avgG
    ab = avgGray/avgB
    for i in range(pc[0]):
        for j in range(pc[1]):
            img[i,j] = [img[i,j,0]*ar,img[i,j,1]*ag,img[i,j,2]*ab]

    cv2.imshow("img",img)

def sobelEdge(srcImg):
    # Sobel边缘检测
    sobelX = cv2.Sobel(srcImg, cv2.CV_64F, 1, 0)  # x方向的梯度
    sobelY = cv2.Sobel(srcImg, cv2.CV_64F, 0, 1)  # y方向的梯度

    sobelX = np.uint8(np.absolute(sobelX))  # x方向梯度的绝对值
    sobelY = np.uint8(np.absolute(sobelY))  # y方向梯度的绝对值

    sobelCombined = cv2.bitwise_or(sobelX, sobelY)
    return sobelCombined

def thresholdImg(srcImg,keyTH):
    img = srcImg.copy()
    # global thresholding
    ret1, th1 = cv2.threshold(img, keyTH, 255, cv2.THRESH_BINARY_INV)
    # Otsu's thresholding
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ret1,th1

def detectFaces(srcImg):
    img = srcImg.copy()
    #print 1
    face_cascade = cv2.CascadeClassifier("data/haarcascades/haarcascade_frontalface_default.xml")
    test = face_cascade.load("data/haarcascades/haarcascade_frontalface_default.xml")
    print(test)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)#1.3and5  counts to the result of face recognation
    result = []
    for (x,y,width,height) in faces:
        result.append((x,y,x+width,y+height))
        #print result[0]
        print(len(result))
    cv2.rectangle(img, (result[0][0], result[0][1]), (result[0][0] + result[0][2], result[0][1] + result[0][3]), (255, 0, 0), 2)
    cv2.imshow("detectface",img)
    return result

def skinModel(srcImg):
    img = srcImg.copy()
    rows, cols, channels = img.shape
    # light compensation
    gamma = 0.95
    for r in range(rows):
        for c in range(cols):
            # get values of blue, green, red
            B = img.item(r, c, 0)
            G = img.item(r, c, 1)
            R = img.item(r, c, 2)
            # gamma correction
            B = int(B ** gamma)
            G = int(G ** gamma)
            R = int(R ** gamma)
            # set values of blue, green, red
            img.itemset((r, c, 0), B)
            img.itemset((r, c, 1), G)
            img.itemset((r, c, 2), R)
            ###############################################################################
    # convert color space from rgb to ycbcr
    imgYcc = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    # convert color space from bgr to rgb
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # prepare an empty image space
    imgSkin = np.zeros(img.shape, np.uint8)
    # copy original image
    imgSkin = img.copy()
    ################################################################################
    # define variables for skin rules
    Wcb = 46.97
    Wcr = 38.76
    WHCb = 14
    WHCr = 10
    WLCb = 23
    WLCr = 20
    Ymin = 16
    Ymax = 235
    Kl = 125
    Kh = 188
    WCb = 0
    WCr = 0
    CbCenter = 0
    CrCenter = 0
    ################################################################################
    for r in range(rows):
        for c in range(cols):
            # non-skin area if skin equals 0, skin area otherwise
            skin = 0
            ########################################################################
            # color space transformation
            # get values from ycbcr color space
            Y = imgYcc.item(r, c, 0)
            Cr = imgYcc.item(r, c, 1)
            Cb = imgYcc.item(r, c, 2)
            if Y < Kl:
                WCr = WLCr + (Y - Ymin) * (Wcr - WLCr) / (Kl - Ymin)
                WCb = WLCb + (Y - Ymin) * (Wcb - WLCb) / (Kl - Ymin)
                CrCenter = 154 - (Kl - Y) * (154 - 144) / (Kl - Ymin)
                CbCenter = 108 + (Kl - Y) * (118 - 108) / (Kl - Ymin)
            elif Y > Kh:
                WCr = WHCr + (Y - Ymax) * (Wcr - WHCr) / (Ymax - Kh)
                WCb = WHCb + (Y - Ymax) * (Wcb - WHCb) / (Ymax - Kh)
                CrCenter = 154 + (Y - Kh) * (154 - 132) / (Ymax - Kh)
                CbCenter = 108 + (Y - Kh) * (118 - 108) / (Ymax - Kh)
            if Y < Kl or Y > Kh:
                Cr = (Cr - CrCenter) * Wcr / WCr + 154
                Cb = (Cb - CbCenter) * Wcb / WCb + 108
                ########################################################################
            # skin color detection
            if Cb > 77 and Cb < 127 and Cr > 133 and Cr < 173:
                skin = 1
                # print 'Skin detected!'
            if 0 == skin:
                imgSkin.itemset((r, c, 0), 0)
                imgSkin.itemset((r, c, 1), 0)
                imgSkin.itemset((r, c, 2), 0)

                # display original image and skin image
    cv2.imshow("imgSkin",imgSkin)
    return imgSkin

def selectThreshold(img):
    hist_cv = cv2.calcHist([img], [0], None, [256], [0, 256])
    sum = 0
    for i in range(256):
        sum += hist_cv[i,0]
    temp = 0
    for i in range(256):
        temp += hist_cv[i,0];
        if temp/sum > 0.95:
            print(i)
            return i;

def cannyImg(img):
    img = cv2.GaussianBlur(img, (3, 3), 0)  # 高斯平滑处理原图像降噪
    canny = cv2.Canny(img, 20, 60)  # apertureSize默认为3
    return canny

def removeNoisy(srcImg):
    img = srcImg.copy()
    # OpenCV定义的结构元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # 膨胀图像
    dilated = cv2.dilate(img, kernel)

    # 腐蚀图像
    eroded = cv2.erode(dilated, kernel)
    # 显示腐蚀后的图像
    cv2.imshow("Eroded Image", eroded)
    return eroded

def removeBackground(srcImg):
    img = srcImg.copy()
    sp = img.shape
    mask = np.zeros((img.shape[0]+2,img.shape[1]+2),np.uint8)
    cv2.floodFill(img, mask, (5, 5), (255, 255, 255), (3, 3, 3), (3, 3, 3), 8)
    cv2.imshow("floodfill", img)
    return img


def hairComplete(img):

    return None



if __name__ == '__main__':
    srcImg = cv2.imread("img/1.jpg", 1);
    srcImgCopy = cv2.imread("img/1.jpg", 1);
    cv2.imshow("srcimg",srcImg)
    #grayWorld(srcImg)


    #采用XY方向上进行求导提取边缘
    srcImg = cv2.medianBlur(srcImg,3)
    srcImgCopy = cv2.medianBlur(srcImgCopy,3)

    sobelCombined = sobelEdge(srcImg)
    graySobel = cv2.cvtColor(sobelCombined,cv2.COLOR_RGB2GRAY)

    #根据图像的像素来选择一个合适的阈值
    keyTH = selectThreshold(graySobel)
    ret, th = thresholdImg(graySobel,keyTH)

    #去除背景（无相关操作）
    removeBackground(srcImgCopy)
    skinModel(srcImgCopy)

    #提取边缘


    canny = cannyImg(srcImgCopy)
    #因为与操作是针对255像素的，所以要进行一个反转
    ret, th2 = thresholdImg(th, 100)
    #两个边缘进行一个与操作，保留细节
    th3 = cv2.bitwise_or(canny,th2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    th3 = cv2.dilate(th3, kernel)
    th3 = cv2.erode(th3, kernel)
    th3 = cv2.dilate(th3, kernel)
    th3 = cv2.erode(th3, kernel)
    th3 = cv2.dilate(th3, kernel)
    th3 = cv2.erode(th3, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # 闭运算,去除孤立点
    th3 = cv2.morphologyEx(th3, cv2.MORPH_CLOSE, kernel)

    ret, th3 = thresholdImg(th3, 100)
    th3 = cv2.medianBlur(th3,3)
    th3 = cv2.medianBlur(th3,5)

    #为使得轮廓更加明显，用得到的图像与canny做一个与操作
    ret, th3 = thresholdImg(th3, 100)
    th3 = cv2.bitwise_or(canny, th3)
    ret, th3 = thresholdImg(th3, 100)
    th3 = cv2.erode(th3, kernel)
    cv2.imshow("canny",canny)
    cv2.imshow("result", th3)

    cv2.waitKey(0);

