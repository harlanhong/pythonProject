import cv2
import numpy as np
import math
from faceFeature.HistUtil import *
def detectFaces(srcImg):
    img = srcImg.copy()
    #print 1
    face_cascade = cv2.CascadeClassifier("data/haarcascades/haarcascade_frontalface_default.xml")
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)#1.3and5  counts to the result of face recognation
    return faces
def tailorImg(img,faces):
    x_up=[]
    y_up = []
    x_down=[]
    y_down=[]
    if len(faces)  == 1:
        for i,(x,y,w,h) in enumerate(faces):
            #注意不要越界
            new_x = x - int(w/3) if x - int(w/3)>0 else 0
            new_y = y - int(h/3) if y - int(h/3)>0 else 0
            new_w = w +int(2/3*w) if new_x+w+int(2/3*w)<img.shape[1] else img.shape[1]-new_x
            new_h = h +int(2/3*h) if new_y+h +int(2/3*h)<img.shape[0] else img.shape[0]-new_y
            new_img = img[new_y:new_y+new_h,new_x:new_x+new_w]
            return new_img

            #cv2.imshow("new_img",new_img)
    elif len(faces)>1:
        for i, (x, y, w, h) in enumerate(faces):
            x_up.append(x);
            y_up.append(y);
            x_down.append(x+w);
            y_down.append(y+h);
            # 注意不要越界
        x = min(x_up)
        y = min(y_up)
        w = max(x_down) - x
        h = max(y_down) - y

        new_x = x - int(w / 3) if x - int(w / 3) > 0 else 0
        new_y = y - int(h / 3) if y - int(h / 3) > 0 else 0
        new_w = w + int(2 / 3 * w) if new_x + w + int(2 / 3 * w) < img.shape[1] else img.shape[1] - new_x
        new_h = h + int(2 / 3 * h) if new_y + h + int(2 / 3 * h) < img.shape[0] else img.shape[0] - new_y
        new_img = img[new_y:new_y + new_h, new_x:new_x + new_w]
        cv2.imshow("tailor",new_img)
        return new_img
# 通过漫水填充去除背景
def removeBackground(srcImg,color = (255,255,255)):
        img = srcImg.copy()
        sp = img.shape
        mask1 = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)
        mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)
        cv2.floodFill(img, mask, (3, 3), color, (3, 3, 3), (3, 3, 3), 8)
        cv2.floodFill(img, mask1, (sp[1] - 3, 3), color, (3, 3, 3), (3, 3, 3), 8)
        return img
def myThreshold(imgGray,imgSkin,skinPoint,thresh):
    sp =  imgGray.shape
    dst = imgGray.copy()
    for i in range(sp[0]):
        for j in range(sp[1]):
            if imgSkin[i,j] >100:
                if imgGray[i,j]>thresh:
                    dst[i,j] = 255
                else:
                    dst[i,j] = 0
    return dst
#肤色检测
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
    # copy original image
    imgSkin = np.zeros((img.shape[0],img.shape[1]),np.uint8)
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
            #if Cb >= 80 and Cb <= 127 and Cr >= 141 and Cr <= 173:
            if Cb >= 80 and Cb <= 127 and Cr >= 141 and Cr <= 175:
                skin = 1
                # print 'Skin detected!'
            if 0 == skin:
                imgSkin[r,c] = 0
            else:
                imgSkin[r,c] = 255
                # display original image and skin image
    return imgSkin
def adaptiveThreshold(img):
    hist = myCalHist(img)
    histImg = DrawHist(hist, [255, 255, 255])
    cv2.imshow("histInit", histImg)
    min,max,hist = removeNoiseOfHist(hist)
    histImg = DrawHist(hist,[255,255,255])
    cv2.imshow("histIMG",histImg)
    #没有用处，可以不用，只是为了可视化
    #thresh = GetMinimumThreshold(hist)
    return min,max,0
def getFaceBySkin(skin,grayFace):
    sp = skin.shape
    result = grayFace.copy()
    for i in range(sp[0]):
        for j in range(sp[1]):
            if skin[i,j] != 255:
                result[i,j] = 255
    return result
def computeEdgesPoint(img):
    sp = img.shape
    point = 0
    sum =0
    for i in range(sp[0]):
        for j in range(sp[1]):
            if img[i,j]>100:
                point +=1
            sum +=1
    temp = point/sum
    if temp > 0 and temp<=0.05:
        return 5*temp
    elif temp> 0.05 and temp<=0.1:
        return 3*temp
    elif temp>0.1 and temp<=0.2:
        return 1.5*temp
    elif temp>0.3 and temp<=0.4:
        return 1.1*temp
    else:
        return temp
def removeNoiseOfHist(hist):
    result = hist.copy()
    counter = 0
    for i in range(255,-1,-1):
        if result[i] != 0:
            counter += 1
        result[i] = 0
        if counter == 10:
            break
    counter = 0
    for i in range(255):
        if result[i] !=0:
            counter+=1
        result[i] = 0
        if counter == 10:
            break

    for  i in range(255):
        if result[i]<50:
            result[i] = 0
        else:
            break
    for  i in range(255,-1,-1):
        if result[i]<50:
            result[i] = 0
        else:
            break
    for i in range(70):
        if result[i]>500:
            break
        result[i]=0
    min ,max =0,0
    for i in range(255):
        if result[i]!=0:
            min = i;
            break
    for i in range(255,-1,-1):
        if result[i]!=0:
            max = i;
            break
    return min,max,result
def AvgGray(imgSKIN,imgGRAY):
    sp = imgSKIN.shape
    brightness = 0
    count = 0
    for i in range(sp[0]):
        for j in range(sp[1]):
           if imgSKIN[i,j] >100:# and imgGRAY[i,j]<err_mean[1]+70:
               brightness += imgGRAY[i,j]
               count +=1
    if count !=0:
        return count,brightness/count
    else:
        return count,0
def partitionThreshold(imgSKIN,imgFace,imgCanny,patchSize):
    counter, theta = AvgGray(imgSKIN=imgSKIN, imgGRAY=imgFace)
    sp = imgFace.shape;
    divisionCount = math.floor(sp[1] / patchSize)
    imgFaceSegment = [None] * divisionCount
    imgSkinSegment = [None] * divisionCount
    for i in range(len(imgFaceSegment)):
        imgFaceSegment[i] = [0] * divisionCount
        imgSkinSegment[i] = [0] * divisionCount
    new_w = math.floor(sp[1]/divisionCount)
    new_h = math.floor(sp[0]/divisionCount)
    newGray = imgFace[0:new_h * divisionCount, 0:new_w * divisionCount]
    newSkin = imgSKIN[0:new_h * divisionCount, 0:new_w * divisionCount]
    newCanny = imgCanny[0:new_h * divisionCount, 0:new_w * divisionCount]
    dst = imgFace.copy()
    dst = dst[0:new_h * divisionCount, 0:new_w * divisionCount]
    for i in range(divisionCount):
        for j in range(divisionCount):
            imgFaceSegment[i][j] = imgFace[i*new_h:(i+1)*new_h,j*new_w:(j+1)*new_w]
            imgSkinSegment[i][j] = imgSKIN[i*new_h:(i+1)*new_h,j*new_w:(j+1)*new_w]
            skinpoint,alpha = AvgGray(imgSkinSegment[i][j],imgFaceSegment[i][j])
            if alpha !=0:
                min_th = min(alpha, theta)
                max_th = max(alpha, theta)
                # imgSegment[i][j] = myThreshold(imgSegment[i][j],imgSkinSegment[i][j],skinPoint,0.71*thresh) gamma = 0.63 lamda = 0.855 bata = 0.145
                edges = imgCanny[i * new_h:(i + 1) * new_h, j * new_w:(j + 1) * new_w]
                # 计算在这一小块中边缘点为多少个
                edgesPointPercent = computeEdgesPoint(edges)
                if alpha > theta:
                    # 尽量往上
                    imgFaceSegment[i][j] = myThreshold(imgFaceSegment[i][j], imgSkinSegment[i][j], skinpoint,
                                                   (1 + edgesPointPercent) * gamma * (lamda * max_th + bata * min_th))
                else:
                    # 尽量往下
                    imgFaceSegment[i][j] = myThreshold(imgFaceSegment[i][j], imgSkinSegment[i][j], skinpoint,
                                                   (1 + edgesPointPercent) * gamma * (lamda1 * max_th + bata1 * min_th))
    for i in range(divisionCount):
        for j in range(divisionCount):
            dst[i*new_h:(i+1)*new_h,j*new_w:(j+1)*new_w]=imgFaceSegment[i][j]
    cv2.imshow("initResult",dst)
    ret, dst = cv2.threshold(dst, 40, 255, cv2.THRESH_BINARY)
    return dst, newSkin, newGray

def process(img):
    img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)), interpolation=cv2.INTER_CUBIC)
    faces = detectFaces(srcImg=img)
    print(len(faces))
    imgFaceBGR = tailorImg(img=img,faces=faces)

    imgFaceBGR = removeBackground(imgFaceBGR)

    imgFaceGray = cv2.cvtColor(imgFaceBGR,cv2.COLOR_BGR2GRAY)
    #cv2.imshow("imgFaceGray", imgFaceGray)
    imgSkin = skinModel(imgFaceBGR)
    #cv2.imshow("imgskin",imgSkin)
    imgFaceByskin = getFaceBySkin(skin=imgSkin,grayFace=imgFaceGray)
    #cv2.imshow("imgfacebyskin",imgFaceByskin)
    #统计直方图,去除肤色噪点
    min,max,thresh = adaptiveThreshold(imgFaceByskin)
    sp = imgFaceByskin.shape
    for i in range(sp[0]):
        for j in range(sp[1]):
            if imgSkin[i,j]>100 and (imgFaceByskin[i,j]<min or imgFaceByskin[i,j]>max):
                imgSkin[i,j]=0
                imgFaceByskin[i,j] =255
    cv2.imshow("realSkin",imgFaceByskin)

    canny = cv2.Canny(imgFaceGray, 40, 120)

    imgFace_thresh,newSkin,newFace = partitionThreshold(imgSKIN=imgSkin,imgFace=imgFaceGray,imgCanny=canny,patchSize=15)
    imgFace_thresh = cv2.medianBlur(imgFace_thresh,3)

    #获取轮廓
    skeletonFace = removeBackground(newFace,(0,0,0))
    skeleton = getSkeleton(skeletonFace,2,3)
    result = cv2.bitwise_and(imgFace_thresh,skeleton)
    result = RemoveSelectRegion(result,100,0,0,0)
    cv2.imshow("result",result)
    return  result

def unitTest():
    img = cv2.imread("imageTailor/1 (28).jpg")
    img = cv2.medianBlur(img, 3)
    reslut = process(img)
def allTest():
    for i in range(1, 105):
        print(i)
        img = cv2.imread("img/" + str(i) + ".jpg")
        img = cv2.medianBlur(img, 3)
        reslut = process(img)
        cv2.imwrite("result/" + str(i) + ".jpg", reslut)

if __name__ == '__main__':
    gamma = 0.700
    lamda = 0.7000
    bata = 0.3000
    lamda1 = 0.35000
    bata1 = 0.65000
    unitTest()
    cv2.waitKey(0)