import cv2
import numpy as np
import os
import tensorlayer as tl
import tensorflow as tf
import pylab
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from skinDetect import tensorlayerUtil
from faceFeature import HistUtil
def importData():
    src = []
    bin = []
    path1 = "Face_Dataset/Ground_Truth/GroundT_FacePhoto"  # 文件夹目录
    path2 = "Face_Dataset/Pratheepan_Dataset/FacePhoto"
    files = os.listdir(path2)  # 得到文件夹下的所有文件名称
    for file in files:  # 遍历文件夹
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            portion = os.path.splitext(file)  # 将文件名拆成名字和后缀
            if portion[1] != ".png":  # 关于后缀
                newname = portion[0] + ".png"
                os.rename(path2 + "/" + file, path2 + "/" + newname)  # 修改
            src.append(path2 + "/" + file)
            bin.append(path1 + "/" + file)
    path1 = "Face_Dataset/Ground_Truth/GroundT_FamilyPhoto"  # 文件夹目录
    path2 = "Face_Dataset/Pratheepan_Dataset/FamilyPhoto"
    files = os.listdir(path2)  # 得到文件夹下的所有文件名称
    for file in files:  # 遍历文件夹
        if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
            portion = os.path.splitext(file)  # 将文件名拆成名字和后缀
            if portion[1] != ".png":  # 关于后缀
                newname = portion[0] + ".png"
                os.rename(path2 + "/" + file, path2 + "/" + newname)  # 修改
            src.append(path2 + "/" + file)
            bin.append(path1 + "/" + file)
    return src,bin

def model(src,bin):
    sum = np.zeros((2,))
    size = len(src)
    CB=np.zeros((300,1))
    CR=np.zeros((300,1))
    crcb = np.zeros((300,300))
    count = 0
    for i in range(size):
        srcImg = cv2.imread(src[i])
        #cv2.imshow("src",srcImg)
        srcImg = cv2.cvtColor(srcImg, cv2.COLOR_BGR2YCrCb)
        binImg = cv2.imread(bin[i], 0)
        #cv2.imshow("bin",binImg)
        sp = srcImg.shape
        for x in range(sp[0]):
            for y in range(sp[1]):
                if binImg[x, y] > 200:
                    crcb[srcImg[x,y,1],srcImg[x,y,2]]+=1
    x = np.arange(300)
    y = np.arange(300)
    X, Y = np.meshgrid(x, y)
    Z=crcb
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.set_xlabel('cr')
    ax.set_ylabel('cb')
    ax.set_zlabel('z');
    plt.show()

    #彩色图
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    # plt.show()
    np.savetxt("crcb.txt",crcb)
    u = getParameter(crcb)
    print(u)
    return crcb

def getParameter(crcb):
    sp = crcb.shape
    crSum=0
    cbSum=0
    point = 0
    CB=[]
    CR =[]
    for i in range(sp[0]):
        for j in range(sp[1]):
            crSum += crcb[i,j]*i
            cbSum +=crcb[i,j]*j
            for time in range(int(crcb[i,j])):
                CB.append(j)
                CR.append(i)
            point+=crcb[i,j]

    y = [CB,CR]
    C = np.cov(y)
    m=np.matrix(np.array([[crSum / point], [cbSum / point]]))
    return C,m

def classify(X,m,C):
    return np.exp(-1/2*(X-m).T*C.I*(X-m))
def process(img,m,C):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
    sp = img.shape
    gaussian = np.zeros((sp[0],sp[1]),np.uint8)
    binary = np.zeros((sp[0],sp[1]))
    for x in range(sp[0]):
        for y in range(sp[1]):
            cr = img[x, y, 1]# / (img[x, y, 0] + img[x, y, 1] + img[x, y, 2])
            cb = img[x, y, 2] #/ (img[x, y, 0] + img[x, y, 1] + img[x, y, 2])
            X = np.array([[cr],[cb]])
            X = np.matrix(X)
            gaussian[x,y] = (classify(X,m,C))*255
    ret,binary = cv2.threshold(gaussian,40,255,cv2.THRESH_BINARY)
    return  binary,gaussian
def gaussModel(img):
    crcb = np.loadtxt("crcb.txt")
    # s,u = getParameter(crcb)
    C = np.loadtxt("s.txt")
    C = np.matrix(C)
    m = np.loadtxt("u.txt")
    m = np.matrix(m)
    m = np.reshape(m, [2, 1])
    return process(img, m, C)
if __name__ == '__main__':
    #数据导入
    # src,bin = importData()
    # print(src,bin)
    #获取模型数据
    #crcb = model(src,bin)
    img = cv2.imread("imageTailor/1 (169).jpg")
    faces = HistUtil.detectFaces(img)
    img1 = HistUtil.tailorImg(img,faces)
    img = HistUtil.removeBackground(img1,(0,0,0))
    binary,gauss = gaussModel(img);
    cv2.imshow("binary",binary)
    cv2.imshow("gauss",gauss)
    cv2.imshow("wrc",img1)
    cv2.imwrite("skeleton/b1.jpg",binary)
    cv2.imwrite("skeleton/g1.jpg",gauss)
    cv2.imwrite("skeleton/src1.jpg",img1)
    cv2.waitKey(0)







