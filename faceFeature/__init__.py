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

def thresholdImg(img):
    # global thresholding
    ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Otsu's thresholding
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ret2,th2

def detectFaces(img):
    #print 1
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    test = face_cascade.load('haarcascade_frontalface_default.xml')
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

if __name__ == '__main__':
    srcImg = cv2.imread("img/1.jpg", 1);
    srcImgCopy = cv2.imread("img/1.jpg", 1);
    cv2.imshow("srcimg",srcImg)
    #grayWorld(srcImg)
    sobelCombined = sobelEdge(srcImg)
    graySobel = cv2.cvtColor(sobelCombined,cv2.COLOR_RGB2GRAY)
    cv2.imshow("sobel",sobelCombined)
    cv2.imshow("graysobel",graySobel)
    print(graySobel.ndim)
    detectFaces(srcImgCopy)
    cv2.waitKey(0);

