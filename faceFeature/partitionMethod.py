# encoding:utf-8
import cv2
import numpy as np
import math
import faceFeature.HistUtil as histUtil
import faceFeature.globalThresholdMothed as globalThresholdMothod
import faceFeature.nonSkinMethod as nonSkinMode
#检测人脸
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
#在人脸的基础上检测眼睛
def detectEyes(srcImg):
    src = srcImg.copy()
    faces = detectFaces(src)
    eye_cascade = cv2.CascadeClassifier('data/haarcascades_cuda/haarcascade_eye_tree_eyeglasses.xml')
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    result = [[]]
    for j,(x,y,w,h) in enumerate(faces):
        cv2.rectangle(src, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = src[y:y + h,x:x + w]  # 检测视频中脸部的眼睛，并用vector保存眼睛的坐标、大小（用矩形表示）
        eyes = eye_cascade.detectMultiScale(roi_gray) #眼睛检测
        #选择最大的两个框当成眼睛
        max = 0
        temp = []
        index = 0
        for i,(ex,ey,ew,eh) in enumerate(eyes):
            if ew*eh>max and ex<w and ey<h/2:
                max = ew*eh
                index = i
                temp = (ex+x,ey+y,ew,eh)
        if index != 0:
            result[j].append(temp)
            cv2.rectangle(src, (temp[0], temp[1]), (temp[0] + temp[2], temp[1] + temp[3]), (0, 255, 0), 2)
        max = 0
        temp = []
        #用来确认是否找到眼睛
        flag = 0
        for i,(ex,ey,ew,eh) in enumerate(eyes):
            if ew*eh>max and i != index and ex<w and ey<h/2:
                print(ex,ey,x+w,y+h/2)
                max = ew*eh
                flag = 1
                temp = (ex+x,ey+y,ew,eh)
        if flag == 1:
            result[j].append(temp)
            cv2.rectangle(src, (temp[0], temp[1]), (temp[0] + temp[2], temp[1] + temp[3]), (0, 255, 0), 2)
    #cv2.imshow("eye",src)
    return faces,result
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
    # prepare an empty image space
    imgSkin = np.zeros(img.shape, np.uint8)
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
    skinCounter=0
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
            if Cb >= 80 and Cb <= 127 and Cr >= 133 and Cr <= 177:
                skin = 1
                # print 'Skin detected!'
            if 0 == skin:
                imgSkin[r,c] = 0
            else:
                imgSkin[r,c] = 255
                skinCounter+=1
                # display original image and skin image
    return skinCounter,imgSkin
#通过漫水填充去除背景
def removeBackground(srcImg,color=(255,255,255)):
    img = srcImg.copy()
    sp = img.shape
    mask = np.zeros((img.shape[0]+2,img.shape[1]+2),np.uint8)
    mask1 = np.zeros((img.shape[0]+2,img.shape[1]+2),np.uint8)

    cv2.floodFill(img, mask, (3, 3), color, (3, 3, 3), (3, 3, 3), 8)

    cv2.floodFill(img,mask1,(sp[1]-3,3),color, (3, 3, 3), (3, 3, 3), 8)
    #img = myFloodFill(img,mask,(5,5),(3, 3, 3), (3, 3, 3), 8)
    #mask = np.zeros((img.shape[0]+2,img.shape[1]+2),np.uint8)
    #cv2.floodFill(img, mask, (5,img.shape[0]-5), (255, 255, 255), (3, 3, 3), (3, 3, 3), 8)
    cv2.imshow("floodfill", img)
    return img
#通过人脸检测裁剪人脸
def tailorImg(img,faces):
    if len(faces) == 1 or len(faces)>1:
        for i,(x,y,w,h) in enumerate(faces):
            #注意不要越界
            new_x = x - int(w/3) if x - int(w/3)>0 else 0
            new_y = y - int(h/3) if y - int(h/3)>0 else 0
            new_w = w +int(2/3*w) if new_x+w+int(2/3*w)<img.shape[1] else img.shape[1]-new_x
            new_h = h +int(2/3*h) if new_y+h +int(2/3*h)<img.shape[0] else img.shape[0]-new_y
            new_img = img[new_y:new_y+new_h,new_x:new_x+new_w]
            #cv2.imshow("new_img",new_img)
            return new_img
#计算肤色部分的平均灰度
def meanBrightness(imgSKIN,imgGRAY):
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
#计算全局平均肤色值（包括背景）
def computAvg(imgGray):
    sp = imgGray.shape
    avg = 0
    counter = 0
    for i in range(sp[0]):
        for j in range(sp[1]):
            avg+=imgGray[i,j]
            counter+=1
    return avg/counter
#获取最大的轮廓区域
def getMaxArea(new_skin):
    img,contours, hierarchy = cv2.findContours(new_skin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    maxValue = 0
    if len(contours) == 1:
        return 1,cv2.contourArea(contours[0])
    else:
        for i,contour in enumerate(contours):
            if cv2.contourArea(contour)>maxValue:
                maxValue = cv2.contourArea(contour)
    return len(contours),maxValue
#轮廓完善,因为计算量过大，所以不采用这种方法，而是采用
def skeletonComplete(imgResult,imgSKIN,imgFace):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    new_skin = imgSKIN
    new_skin = cv2.medianBlur(new_skin,3)
    new_skin = cv2.dilate(new_skin, kernel)
    new_skin = cv2.erode(new_skin, kernel)
    new_skin = cv2.dilate(new_skin, kernel)
    new_skin = cv2.erode(new_skin, kernel)
    new_skin = cv2.dilate(new_skin, kernel)
    new_skin = cv2.dilate(new_skin, kernel)
    new_skin = cv2.erode(new_skin, kernel)
    new_skin = cv2.erode(new_skin, kernel)
    new_skin = cv2.erode(new_skin, kernel)
    new_skin = cv2.dilate(new_skin, kernel)

    new_skin = RemoveSmallRegion(new_skin,3000,0,1)
    count,maxth = getMaxArea(new_skin)
    cv2.imshow("beforeRemoveLargestArea",new_skin)
    print("最大面积:",maxth,count)
    if count != 1:
        new_skin = RemoveSmallRegion(new_skin,maxth-1000,1,1)
    cv2.imshow("new_skin",new_skin)
    if count == 1 and maxth > 3/5*(new_skin.shape[0]*new_skin.shape[1]):
        print("轮廓面积过大，不认为是正常的轮廓！")
        result = imgResult
    else:
        #binary, contours, hierarchy = cv2.findContours(new_skin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        #cv2.drawContours(binary, contours, -1, (0, 0, 255), 3)
        #cv2.imshow("inver",binary)
        img = cv2.GaussianBlur(new_skin, (3, 3), 0)  # 高斯平滑处理原图像降噪
        canny = cv2.Canny(img, 50, 150)  # apertureSize默认为3
        ret,skeleton = cv2.threshold(canny,100,255,cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        skeleton = cv2.erode(skeleton, kernel)
        #skeleton = cv2.erode(skeleton, kernel)
        result = cv2.bitwise_and(skeleton,imgResult)
    cv2.imshow("skeletonComplete",result)
    return result
#去除二值图像边缘的突出部
def delete_jut(src,uthreshold,vthreshold,type):
    threshold = 0
    dst = src.copy()
    sp = src.shape
    height = sp[0]
    width = sp[1]
    k = 0
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
    return dst
#对图片进行处理
#去除小区域的
def RemoveSmallRegion(src,AreaLimit,CheckMode,NeiborMode):
    RemoveCount = 0
    # 新建一幅标签图像初始化为0像素点，为了记录每个像素点检验状态的标签，0代表未检查，1代表正在检查, 2代表检查不合格（需要反转颜色），3代表检查合格或不需检查
    # 初始化的图像全部为0，未检查
    PointLabel = np.zeros(src.shape,np.uint8)
    dst = np.zeros(src.shape, np.uint8)
    sp = src.shape
    if(CheckMode == 1):#去除小连通区域的白色点,除去白色的
        print("去除小连通域")
        for i in range(sp[0]):
            for j in range(sp[1]):
                if src[i,j]<10:
                    PointLabel[i,j] = 3#将背景黑色点标记为合格，像素为3
    else: #除去黑色的
        print("去除孔洞")
        for i in range(sp[0]):
            for j in range(sp[1]):
                if src[i,j]>10:
                    PointLabel[i,j] = 3#如果原图是白色区域，标记为合格，像素为3
    NeiborPos = [] #将邻域压进容器
    NeiborPos.append((-1,0))
    NeiborPos.append((1, 0))
    NeiborPos.append((0, -1))
    NeiborPos.append((0, 1))
    if NeiborMode ==1:
        print("Neighbor mode: 8邻域.")
        NeiborPos.append((-1, -1))
        NeiborPos.append((-1, 1))
        NeiborPos.append((1, -1))
        NeiborPos.append((1, 1))
    else:
        print("Neighbor mode: 4邻域.")
    NeihborCount = 4+4*NeiborMode;
    CurrX = 0
    CurrY =0
    for i in range(sp[0]):
        for j in range(sp[1]):
            if PointLabel[i,j] == 0:
                GrowBuffer = []
                GrowBuffer.append((i,j))
                PointLabel[i,j] = 1
                CheckResult = 0
                #在这里说一下，python的for循环有点奇葩，就是范围是静态的，第一次获取到范围值后就不会改变了
                z=0
                while z<len(GrowBuffer):
                    for q in range(NeihborCount):
                        CurrX = GrowBuffer[z][0]+NeiborPos[q][0]
                        CurrY = GrowBuffer[z][1]+NeiborPos[q][1]
                        if CurrX >=0 and CurrX < sp[0] and CurrY >=0 and CurrY<sp[1]:
                            if PointLabel[CurrX,CurrY] == 0:
                                GrowBuffer.append((CurrX,CurrY))
                                PointLabel[CurrX,CurrY] = 1
                    z += 1
                #对整个连通域检查完
                if len(GrowBuffer)>AreaLimit:
                    CheckResult = 2
                else:
                    CheckResult = 1
                    RemoveCount +=1
                for z in range(len(GrowBuffer)):
                    CurrX = GrowBuffer[z][0]
                    CurrY = GrowBuffer[z][1]
                    PointLabel[CurrX,CurrY] += CheckResult
    CheckMode = 255*(1-CheckMode)
    for i in range(sp[0]):
        for j in range(sp[1]):
            if PointLabel[i,j] == 2:
                dst[i,j] = CheckMode
            if PointLabel[i,j] == 3:
                dst[i,j] = src[i,j]

    return dst
#去除指定大小的区域
def RemoveSelectRegion(src,AreaHigh,AreaLow,CheckMode,NeiborMode):
    RemoveCount = 0
    # 新建一幅标签图像初始化为0像素点，为了记录每个像素点检验状态的标签，0代表未检查，1代表正在检查, 2代表检查不合格（需要反转颜色），3代表检查合格或不需检查
    # 初始化的图像全部为0，未检查
    PointLabel = np.zeros(src.shape,np.uint8)
    dst = np.zeros(src.shape, np.uint8)
    sp = src.shape
    if(CheckMode == 1):#去除小连通区域的白色点,除去白色的
        print("去除小连通域")
        for i in range(sp[0]):
            for j in range(sp[1]):
                if src[i,j]<10:
                    PointLabel[i,j] = 3#将背景黑色点标记为合格，像素为3
    else: #除去黑色的
        print("去除孔洞")
        for i in range(sp[0]):
            for j in range(sp[1]):
                if src[i,j]>10:
                    PointLabel[i,j] = 3#如果原图是白色区域，标记为合格，像素为3
    NeiborPos = [] #将邻域压进容器
    NeiborPos.append((-1,0))
    NeiborPos.append((1, 0))
    NeiborPos.append((0, -1))
    NeiborPos.append((0, 1))
    if NeiborMode ==1:
        print("Neighbor mode: 8邻域.")
        NeiborPos.append((-1, -1))
        NeiborPos.append((-1, 1))
        NeiborPos.append((1, -1))
        NeiborPos.append((1, 1))
    else:
        print("Neighbor mode: 4邻域.")
    NeihborCount = 4+4*NeiborMode;
    CurrX = 0
    CurrY =0
    for i in range(sp[0]):
        for j in range(sp[1]):
            if PointLabel[i,j] == 0:
                GrowBuffer = []
                GrowBuffer.append((i,j))
                PointLabel[i,j] = 1
                CheckResult = 0
                #在这里说一下，python的for循环有点奇葩，就是范围是静态的，第一次获取到范围值后就不会改变了
                z=0
                while z<len(GrowBuffer):
                    for q in range(NeihborCount):
                        CurrX = GrowBuffer[z][0]+NeiborPos[q][0]
                        CurrY = GrowBuffer[z][1]+NeiborPos[q][1]
                        if CurrX >=0 and CurrX < sp[0] and CurrY >=0 and CurrY<sp[1]:
                            if PointLabel[CurrX,CurrY] == 0:
                                GrowBuffer.append((CurrX,CurrY))
                                PointLabel[CurrX,CurrY] = 1
                    z += 1
                #对整个连通域检查完
                if len(GrowBuffer)> AreaHigh or len(GrowBuffer) < AreaLow:
                    CheckResult = 2
                else:
                    CheckResult = 1
                    RemoveCount +=1

                for z in range(len(GrowBuffer)):
                    CurrX = GrowBuffer[z][0]
                    CurrY = GrowBuffer[z][1]
                    PointLabel[CurrX,CurrY] += CheckResult
    CheckMode = 255*(1-CheckMode)
    for i in range(sp[0]):
        for j in range(sp[1]):
            if PointLabel[i,j] == 2:
                dst[i,j] = CheckMode
            if PointLabel[i,j] == 3:
                dst[i,j] = src[i,j]

    return dst
#阈值化
def myThreshold(imgGray,imgSkin,skinPoint,thresh,globalAvg=0,localAvg=0):
    #对每一个局部进行阈值处理，在这里，对各个因素加以考虑，下面列举几种情况
    #是皮肤且应该阈值化为白色的但是因为肤色偏暗
    #是皮肤但是因为肤色模型的原因，检测为非皮肤，应该阈值化为白色
    #非皮肤但是被肤色模型判定为肤色，例如眉毛、额头上的头发，这些应该阈值化为黑色
    #非皮肤且肤色模型检测正确，但是因为灰度值很高，很难阈值化为黑色
    sp =  imgGray.shape
    dst = imgGray.copy()
    #print(thresh,globalAvg,localAvg)
    if globalAvg<145:
        for i in range(sp[0]):
            for j in range(sp[1]):
                if imgSkin[i,j] >100:
                    if imgGray[i,j]>thresh:# and imgGray[i,j]>globalAvg-25:
                        dst[i,j] = 255
                    else:
                        dst[i,j] = 0
    else:
        for i in range(sp[0]):
            for j in range(sp[1]):
                if imgSkin[i,j] >100:
                    if imgGray[i,j]>thresh and imgGray[i,j]>(globalAvg+localAvg)/2.25:
                        dst[i,j] = 255
                    else:
                        dst[i,j] = 0
    return dst

#计算图片中边缘点的个数
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
#分区结合边缘检测的方法来进行二值化
def divisionThreshold(imgSKIN,imgFace,imgCanny,imgHair,imgBGR):
    useless,theta = meanBrightness(imgSKIN=imgSKIN, imgGRAY=imgFace)
    print("平均亮度",theta)
    sp = imgSKIN.shape
    divisionCount = math.floor(sp[1]/6)
    #初始化数组大小
    imgSegment = [None] * divisionCount
    imgSkinSegment = [None] * divisionCount
    for i in range(len(imgSegment)):
        imgSegment[i] = [0] * divisionCount
        imgSkinSegment[i] = [0] * divisionCount

    new_w = math.floor(sp[1]/divisionCount)
    new_h = math.floor(sp[0]/divisionCount)
    dst = imgSKIN.copy()
    dst = dst[0:new_h*divisionCount,0:new_w*divisionCount]
    newSkin = imgSKIN[0:new_h*divisionCount,0:new_w*divisionCount]
    newFace = imgFace[0:new_h*divisionCount,0:new_w*divisionCount]
    #newHair = imgHair[0:new_h*divisionCount,0:new_w*divisionCount]
    newBGR = imgBGR[0:new_h*divisionCount,0:new_w*divisionCount]
    for i in range(divisionCount):
        for j in range(divisionCount):
            imgSegment[i][j] = imgFace[i*new_h:(i+1)*new_h,j*new_w:(j+1)*new_w]
            imgSkinSegment[i][j] = imgSKIN[i*new_h:(i+1)*new_h,j*new_w:(j+1)*new_w]
            skinPoint,alpha = meanBrightness(imgSKIN=imgSkinSegment[i][j],imgGRAY=imgSegment[i][j])
            if alpha !=0:
                min_th = min(alpha,theta)
                max_th = max(alpha,theta)
                #imgSegment[i][j] = myThreshold(imgSegment[i][j],imgSkinSegment[i][j],skinPoint,0.71*thresh) gamma = 0.63 lamda = 0.855 bata = 0.145
                edges = imgCanny[i*new_h:(i+1)*new_h,j*new_w:(j+1)*new_w]
                #计算在这一小块中边缘点为多少个
                edgesPointPercent = computeEdgesPoint(edges)
                if alpha>theta:
                    #尽量往上
                    imgSegment[i][j] = myThreshold(imgSegment[i][j],imgSkinSegment[i][j],skinPoint,(1+edgesPointPercent)*gamma*(lamda*max_th+bata*min_th),globalAvg=theta,localAvg=alpha)
                else:
                    #尽量往下
                    imgSegment[i][j] = myThreshold(imgSegment[i][j],imgSkinSegment[i][j],skinPoint,(1+edgesPointPercent)*gamma*(lamda1*max_th+bata1*min_th),globalAvg=theta,localAvg=alpha)
    for i in range(divisionCount):
        for j in range(divisionCount):
            dst[i*new_h:(i+1)*new_h,j*new_w:(j+1)*new_w]=imgSegment[i][j]
    ret,dst = cv2.threshold(dst,150,255,cv2.THRESH_BINARY)
    return dst,newSkin,newFace,newBGR
#头发处理
def hairProcess(imageSrc):
    img = imageSrc.copy()
    sp = img.shape
    imgRGB = img.copy()
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow("hairhaha",img)
    x = 0
    y =0
    while img[x,y] >100 :
        y+=1
        x+=1
    mask = np.zeros((imgRGB.shape[0] + 2, img.shape[1] + 2), np.uint8)
    cv2.floodFill(imgRGB,mask,(x+2,y+3),(0,0,0),(3,3,3),(3,3,3),8)
    print("1",x,y)
    x1=0
    y1=img.shape[1]-1
    while img[x1,y1] > 100:
        x1+=2;
        y1-=2;
    mask = np.zeros((imgRGB.shape[0] + 2, img.shape[1] + 2), np.uint8)
    cv2.floodFill(imgRGB, mask, (y1 - 2,x1 + 2),(0, 0, 0),(3,3,3),(3,3,3), 8)
    print("2",x1,y1)
    imghair = cv2.cvtColor(imgRGB,cv2.COLOR_BGR2GRAY);
    return imghair
#去除肤色噪声
def removeSkinNoise(imgGray,imgBin):
    img = imgGray.copy()
    sp = imgGray.shape
    for i in range(sp[0]):
        for j in range(sp[1]):
            if imgBin[i,j]<100:
                img[i,j]=255
    hist = histUtil.myCalHist(img)
    histImg = histUtil.DrawHist(hist, [255, 255, 255])
    cv2.imshow("histInit", histImg)
    min,max,hist =histUtil.removeNoiseOfHist(hist)
    histImg = histUtil.DrawHist(hist,[255,255,255])
    cv2.imshow("histIMG",histImg)
    for i in range(sp[0]):
        for j in range(sp[1]):
            if imgBin[i, j] > 100:
                if imgGray[i,j]<min or imgGray[i,j]>max:
                    imgBin[i,j] = 0
    return imgBin


#总的图片处理过程

def processImg(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    if img.shape[1]>800 or img.shape[0]>800:
        img = cv2.resize(img,(int(img.shape[1]/2) ,int(img.shape[0]/2)),interpolation=cv2.INTER_CUBIC)
    #cv2.imshow("img",img)
    #人脸检测
    faces = detectFaces(img)
    if len(faces)<1:
        print("人脸识别失败，采用全局阈值")
        return globalThresholdMothod.globalThreshod(img)
    #获取到的人物头像前景图
    imgFace = tailorImg(img,faces)
    cv2.imshow("img", imgFace)
    #获取imgFace的肤色图
    imgFace = removeBackground(imgFace)

    skinCounter,imgFace_Skin = skinModel(imgFace)
    cv2.imshow("skin", imgFace_Skin)

    if skinCounter<imgFace_Skin.shape[0]*imgFace_Skin.shape[1]/5:
        print(skinCounter,imgFace_Skin.shape[0]*imgFace_Skin.shape[1])
        print("肤色检测：少于1/5区域，肤色模型沦陷，采用非肤色模式")
        result = nonSkinMode.processImg(img)
        return result;
    else:

        #去除肤色噪点
        imgFace_Gray = cv2.cvtColor(imgFace, cv2.COLOR_BGR2GRAY)

        imgFace_Skin = removeSkinNoise(imgFace_Gray,imgFace_Skin)
        ####################################################

        #harlan
        #对肤色图进行修剪

        imgFace_Skin = RemoveSelectRegion(imgFace_Skin,5000,0,0,1)

        #imgHair = hairProcess(imgFace)
        #cv2.imshow("skinTemp", imgFace_Skin)


        #harlan=======================================
        cv2.imshow("imgFace_gray",imgFace_Gray)
        canny = cv2.Canny(imgFace_Gray,40,120)
        cv2.imshow("canny",canny)
        imgFace_thresh,newSkin,newFace,newBGR = divisionThreshold(imgSKIN=imgFace_Skin,imgFace=imgFace_Gray,imgCanny=canny,imgHair= None,imgBGR = imgFace)
        cv2.imshow("face_threshold",imgFace_thresh)
        #轮廓补充
        #result = skeletonComplete(imgFace_thresh,imgSKIN=newSkin,imgFace=newFace)
        skeletonFace = removeBackground(newBGR, (0, 0, 0))
        skeleton = histUtil.getSkeleton(skeletonFace, 1, 1)
        skeleton = RemoveSmallRegion(skeleton,200,0,0)

        cv2.imshow("skeleton",skeleton)
        result = cv2.bitwise_and(imgFace_thresh, skeleton)
        return result
#整套图片处理
def createResult():
    i = 1
    while i <= 907:
        print(i)
        img = cv2.imread("imageTailor/1 (" + str(i) + ").jpg", 1)
        #img = cv2.imread("img/"+str(i)+".jpg")
        i = i + 1
        if img is None:
            continue;
        img = cv2.medianBlur(img,3)
        dst = processImg(img)
        dst = cv2.medianBlur(dst, 3)
        dst = RemoveSelectRegion(dst, 20, 0, 0, 1)
        dst = delete_jut(dst, 1, 1, 1)
        dst = delete_jut(dst, 1, 1, 0)
        cv2.imwrite("result/" + str(i-1) + ".jpg", dst)

#单个图片处理
def unitTest():
    #img = cv2.imread("imageTailor/1 (1).jpg",1)
    img = cv2.imread("img/1.jpg")
    img = cv2.medianBlur(img, 3)
    dst = processImg(img)

    dst = cv2.medianBlur(dst, 3)
    dst = RemoveSelectRegion(dst, 20, 0, 0, 1)
    #dst = delete_jut(dst, 1, 1, 1)
    #dst = delete_jut(dst, 1, 1, 0)
    cv2.imshow("result",dst)
    print(dst.shape)
if __name__ == '__main__':
    gamma = 0.700
    lamda = 0.7000
    bata = 0.3000
    lamda1 = 0.35000
    bata1 = 0.65000
    unitTest()
    cv2.waitKey(0)